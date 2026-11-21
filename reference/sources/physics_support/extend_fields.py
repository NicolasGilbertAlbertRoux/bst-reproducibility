'Release residual-derived field directions with the same native kinetic law.\n\nOne new S3 tangent direction per atom. No force, mass or observed angle is\nfitted. Quaternion multiplication preserves the unit field at finite amplitude.'
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import hashlib,json,time
import numpy as np
from fields_and_references import Configuration,rotate,dump
from cotransported_references import CoTransportedConfiguration
from dynamics_full import NodalKinetic,center_jets,jacobian_jet,ROOT
from refit_dependencies import skew

def product(a,b):
    return np.column_stack([a[:,0]*b[:,0]-np.einsum('ni,ni->n',a[:,1:],b[:,1:]),
        a[:,0,None]*b[:,1:]+b[:,0,None]*a[:,1:]+np.cross(a[:,1:],b[:,1:])])

class EnrichedConfiguration(CoTransportedConfiguration):
    def __init__(self):
        super().__init__()
        path=ROOT/'resultats/CO_INITIAL_DIRECTIONS_NODALES_R345.npz';z=np.load(path)
        assert np.max(abs(z['initial_state'][:87]-self.q0))<1e-13
        self.mode_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        self.K=[np.ascontiguousarray(z[f'K_{i}']) for i in range(5)]
        self.u=np.zeros(5);self.enriched_q0=np.r_[self.q0,np.zeros(5)]
        refresh=self.state.refresh
        def enrich(x):
            refresh(x)
            self.base_fields=[n.copy() for n in self.state.fields]
            self.base_tangents=self.state.tangent;self.base_pull_S=self.state.pull_S
            self.r=[];self.ru=[];self.ruu=[]
            for u,K in zip(self.u,self.K,strict=True):
                norm=np.linalg.norm(K,axis=1);angle=norm*u
                r=np.column_stack([np.cos(angle),K*(u*np.sinc(angle/np.pi))[:,None]])
                ru=np.column_stack([-norm*np.sin(angle),K*np.cos(angle)[:,None]])
                self.r.append(r);self.ru.append(ru);self.ruu.append(-norm[:,None]**2*r)
            self.state.fields=[self.apply(n,r) for n,r in zip(self.base_fields,self.r,strict=True)]
            self.state.tangent=[self.apply(n,r) for n,r in zip(self.base_tangents,self.r,strict=True)]
            self.state.pull_S=self.apply(self.base_pull_S,self.r[0]);self.state.m.fields=self.state.fields
            # The action on a deformed field is recomputed, never inherited
            # from the old field or merely rotated self-energy covector.
            self.model.action.rows=[self.model.action.action(n,self.model.h(i)) for i,n in enumerate(self.state.fields)]
            self.state.m.self_energies=[row[0] for row in self.model.action.rows]
            defect=max(np.max(abs(np.sum(n*n,axis=1)-1.)) for n in self.state.fields)
            assert defect<4e-12,defect
        self.state.refresh=enrich
    def apply(self,n,r):
        R=self.target_rotation
        return rotate(product(rotate(n,R.T),r),R)
    def potential(self,q):
        self.u=np.asarray(q[87:92]);E,g=super().potential(q[:87])
        extra=np.array([np.sum(raw*self.apply(base,ru)) for (i,raw),base,ru in zip(self.raw_total,self.base_fields,self.ru,strict=True)])
        g=np.r_[g,extra];self.last['gradient']=g.copy()
        return E,g
    def jets(self,q):
        saved_t=self.state.tangent;saved_z=self.state.pull_S
        self.state.tangent=self.base_tangents;self.state.pull_S=self.base_pull_S
        try:old=Configuration.jets(self,q[:87])
        finally:self.state.tangent=saved_t;self.state.pull_S=saved_z
        result=[]
        for i,j in enumerate(old):
            new={key:(self.apply(value,self.r[i]) if key in ['a','aa','z','az','zz'] else value) for key,value in j.items()}
            new['u']=self.apply(self.base_fields[i],self.ru[i]);new['uu']=self.apply(self.base_fields[i],self.ruu[i])
            new['at']=self.apply(j['a'],self.ru[i])
            if i==0:new['zu']=self.apply(j['z'],self.ru[i])
            result.append(new)
        return result

class EnrichedDynamics:
    def __init__(self):
        self.configuration=EnrichedConfiguration();self.nq=92;self.kernel=NodalKinetic(self.configuration.beta)
        self.keep=np.r_[np.arange(3),np.arange(6,98)];self.calls=0
    def prepare(self,q):
        c=self.configuration;self.U,self.gU=c.potential(q);self.jets=c.jets(q)
        self.X,self.JX,self.HX=center_jets(c.state.m,q[60:72]);self.frames=c.frames
        self.Jpsi,self.dJpsi=jacobian_jet(q[84:87]);self.frame_jets=[jacobian_jet(t) for t in q[72:84].reshape(4,3)]
        self.E=[];self.sl=[];self.G=np.zeros((98,98))
        for i,(n,j,O) in enumerate(zip(c.state.fields,self.jets,self.frames,strict=True)):
            A,C,G=self.kernel.mass(n,c.model.h(i));a=j['a'];u=j['u'];parts=j['parts']
            aa=np.einsum('ni,nij,nj->n',a,A,a);at=np.einsum('ni,nij,nj->n',a,A,u);uu=float(np.einsum('ni,nij,nj->',u,A,u))
            ca=np.einsum('ni,nij->nj',a,C);cu=np.einsum('ni,nij->j',u,C)
            if i==0:
                Z=j['Z'];z=j['z'];zz=np.einsum('ni,nij,nj->n',z,A,z);za=np.einsum('ni,nij,nj->n',z,A,a);zu=np.einsum('ni,nij,nj->n',z,A,u)
                cz=np.einsum('ni,nij->nj',z,C)
                F0=np.block([[Z.T@(zz[:,None]*Z),Z.T@(za[:,None]*parts)],[parts.T@(za[:,None]*Z),parts.T@(aa[:,None]*parts)]])
                cross=np.r_[Z.T@zu,parts.T@at];B0=np.vstack([Z.T@cz,parts.T@ca]);sl=slice(0,52)
            else:F0=parts.T@(aa[:,None]*parts);cross=parts.T@at;B0=parts.T@ca;sl=slice(50+2*i,52+2*i)
            F=np.block([[F0,cross[:,None]],[cross[None,:],np.array([[uu]])]]);B=np.vstack([B0,cu])
            p=len(F);local=np.block([[G,B.T],[B,F]]);E=np.zeros((9+p,98))
            E[:3,90:93]=-self.Jpsi;E[3:6,:3]=O.T;E[6:9,:3]=-O.T@skew(self.X[i]);E[6:9,3:6]=O.T
            if i:
                k=78+3*(i-1);E[3:6,k:k+3]=O.T@self.frame_jets[i-1][0]
                k=66+3*(i-1);E[6:9,k:k+3]=O.T@self.JX[i-1]
            E[9:-1,6+sl.start:6+sl.stop]=np.eye(sl.stop-sl.start);E[-1,6+87+i]=1.
            self.G+=E.T@local@E;self.E.append(E);self.sl.append(sl)
        self.G=.5*(self.G+self.G.T);T=self.G[3:6,3:6];C=self.G[np.ix_(self.keep,np.arange(3,6))]
        self.R=-np.linalg.solve(T,C.T);self.M=self.G[np.ix_(self.keep,self.keep)]+C@self.R;self.M=.5*(self.M+self.M.T)
        return self.M
    def kinetic(self,q,v,prepared=False):
        if not prepared:self.prepare(q)
        Q=0.;g=np.zeros(92);gv=np.zeros(98);target=np.zeros(3);c=self.configuration
        for i,(n,j,E,O,sl) in enumerate(zip(c.state.fields,self.jets,self.E,self.frames,self.sl,strict=True)):
            local=E@v;adot=j['parts']@v[6+sl.stop-2:6+sl.stop];udot=v[93+i]
            rate=j['a']*adot[:,None]+j['u']*udot;zdot=j['Z']@v[6:56] if i==0 else None
            if i==0:rate+=j['z']*zdot[:,None]
            value,gn,gt,gl=self.kernel.quadratic(n,c.model.h(i),rate,local[:9]);Q+=value
            da=j['aa']*adot[:,None]+j['at']*udot+(j['az']*zdot[:,None] if i==0 else 0.)
            ca=np.einsum('ni,ni->n',gn,j['a'])+np.einsum('ni,ni->n',gt,da)
            g[sl.stop-2:sl.stop]+=j['parts'].T@ca
            of=j['at']*adot[:,None]+j['uu']*udot+(j['zu']*zdot[:,None] if i==0 else 0.)
            g[87+i]+=float(np.sum(gn*j['u'])+np.sum(gt*of))
            localcov=np.r_[gl,np.zeros(sl.stop-sl.start+1)]
            localcov[-3:-1]=j['parts'].T@np.einsum('ni,ni->n',gt,j['a']);localcov[-1]=float(np.sum(gt*j['u']))
            if i==0:
                dz=j['az']*adot[:,None]+j['zz']*zdot[:,None]+j['zu']*udot
                cz=np.einsum('ni,ni->n',gn,j['z'])+np.einsum('ni,ni->n',gt,dz)
                g[:50]+=j['Z'].T@cz;localcov[9:59]=j['Z'].T@np.einsum('ni,ni->n',gt,j['z'])
            gv+=E.T@localcov;target+=np.sum(np.cross(n[:,1:],gn[:,1:])+np.cross(rate[:,1:],gt[:,1:]),axis=0)
            if i:
                ucov=O@gl[6:9];center=slice(60+3*(i-1),60+3*i);cdot=v[6+center.start:6+center.stop]
                g[center]+=self.JX[i-1].T@np.cross(ucov,v[:3])+np.einsum('cab,b,c->a',self.HX[i-1],cdot,ucov)
                frame=slice(72+3*(i-1),72+3*i);tdot=v[6+frame.start:6+frame.stop];Jl,dJl=self.frame_jets[i-1]
                force=O@(np.cross(gl[3:6],local[3:6])+np.cross(gl[6:9],local[6:9]))
                g[frame]+=Jl.T@force+np.einsum('acb,b,c->a',dJl,tdot,O@gl[3:6])
            g[84:87]-=np.einsum('acb,b,c->a',self.dJpsi,v[90:93],gl[:3])
        g[84:87]+=self.Jpsi.T@target
        return float(Q),g,gv
    def evaluate(self,y,diagnostics=False):
        q=y[:92];moment=np.r_[y[-3:],y[92:184]];self.prepare(q)
        v=np.linalg.solve(self.M,moment);full=np.zeros(98);full[self.keep]=v;full[3:6]=self.R@v
        Q,gQ,gv=self.kinetic(q,full,prepared=True);T=.5*float(moment@v);gH=self.gU-.5*gQ
        qdef=abs(Q-2*T)/max(1.,abs(Q));mdef=float(np.max(abs(.5*gv-self.G@full)))
        assert qdef<3e-10 and mdef<2e-7*max(1.,np.max(abs(moment))),(qdef,mdef)
        dy=np.r_[v[3:],-gH,np.cross(moment[:3],v[:3])];self.calls+=1
        self.last={'H_native':float(self.U+T),'U_native':float(self.U),'T_native':T,'q':q.copy(),
            'configuration_velocity':v[3:].copy(),'spatial_velocity':v[:3].copy(),'translation_velocity':full[3:6].copy(),
            'canonical_momenta':moment[3:].copy(),'spatial_body_momentum':moment[:3].copy(),'Hamiltonian_configuration_gradient':gH.copy(),
            'native_Q_mass_relative_defect':qdef,'native_momentum_inf_defect':mdef,'geometry':self.configuration.last['geometry'],
            'target_potential_torque':np.zeros(3),'time_is_native_not_SI':True}
        if diagnostics:
            eig=np.linalg.eigvalsh(self.M);self.last['mass_eigenvalue_range']=eig[[0,-1]].tolist();self.last['mass_condition']=float(eig[-1]/eig[0])
            Jdot=np.einsum('acb,a->cb',self.dJpsi,dy[84:87]);ell=-np.linalg.solve(self.Jpsi.T,y[92+84:92+87])
            elldot=-np.linalg.solve(self.Jpsi.T,dy[92+84:92+87]+Jdot.T@ell)
            self.last['old_target_momentum']=ell.tolist();self.last['target_momentum_time_derivative']=elldot.tolist()
            self.last['Hamiltonian_time_derivative']=float(gH@dy[:92]+v[3:]@dy[92:184]+v[:3]@dy[-3:])
        return dy,self.last['H_native']
    def initial(self):
        q=self.configuration.enriched_q0.copy();self.prepare(q);v=np.zeros(95)
        w=np.array(self.configuration.row['common_rotation_fit']['candidates'][0]['velocity_native']);v[:3]=w[3:6];v[87:90]=-w[:3]
        moment=self.M@v
        return np.r_[q,moment[3:],moment[:3]]

def controls():
    start=time.time();d=EnrichedDynamics();y=d.initial();dy,H=d.evaluate(y,True);initial=d.last.copy()
    assert initial['mass_eigenvalue_range'][0]>1e-10
    np.savez_compressed(ROOT/'resultats/INITIAL_ENRICHI_R345.npz',state=y,mass=d.M,rhs=dy)
    dump(ROOT/'resultats/CONTROLES_ENRICHIS_R345.json',{'revision':345,'full':False,'initial':initial,'physical_validation':False})
    print('Enriched mass positive',initial['mass_eigenvalue_range'],'condition',initial['mass_condition'],flush=True)
    q=y[:92].copy();q[87:92]=[.03,-.02,.01,.025,-.015];q[84:87]=[.17,-.11,.09]
    E,g=d.configuration.potential(q);checks=[]
    for name,sl in [('fields',slice(0,60)),('new_field_modes',slice(87,92)),('target_and_references',slice(84,87))]:
        v=np.zeros(92);v[sl]=np.sin(np.arange(len(v[sl]))*.31+.23);v/=np.linalg.norm(v);exact=float(g@v)
        eps=1e-5;fd=(d.configuration.potential(q+eps*v)[0]-d.configuration.potential(q-eps*v)[0])/(2*eps)
        error=abs(exact-fd);assert error<8e-5*max(1.,abs(exact)),(name,exact,fd,error)
        checks.append({'family':'potential_'+name,'step':eps,'analytic':exact,'finite_difference':fd,'absolute_defect':error})
    velocity=np.sin(np.arange(98)*.37+.2)*.013;d.prepare(q);Q,gQ,gv=d.kinetic(q,velocity,True)
    for name,sl in [('fields',slice(0,60)),('new_field_modes',slice(87,92)),('centers',slice(60,72)),('frames',slice(72,84)),('target',slice(84,87))]:
        v=np.zeros(92);v[sl]=np.sin(np.arange(len(v[sl]))*.31+.23);v/=np.linalg.norm(v);exact=float(gQ@v);eps=1e-5
        fd=(d.kinetic(q+eps*v,velocity)[0]-d.kinetic(q-eps*v,velocity)[0])/(2*eps)
        error=abs(exact-fd);assert error<3e-5*max(1.,abs(exact)),(name,exact,fd,error)
        checks.append({'family':'kinetic_'+name,'step':eps,'analytic':exact,'finite_difference':fd,'absolute_defect':error})
        print('Enriched kinetic derivative',name,'passed',flush=True)
    d.evaluate(y,True);gH=d.last['Hamiltonian_configuration_gradient'].copy()
    v=np.zeros(187);v[87:92]=np.sin(np.arange(5)*.31+.23);v/=np.linalg.norm(v);exact=float(gH@v[:92]);eps=1e-5
    fd=(d.evaluate(y+eps*v)[1]-d.evaluate(y-eps*v)[1])/(2*eps);error=abs(exact-fd);assert error<1e-4*max(1.,abs(exact))
    checks.append({'family':'Hamiltonian_new_modes','step':eps,'analytic':exact,'finite_difference':fd,'absolute_defect':error})
    d.evaluate(y,True);assert max(abs(np.array(d.last['target_momentum_time_derivative'])))<2e-8
    dump(ROOT/'resultats/CONTROLES_ENRICHIS_R345.json',{'revision':345,'full':True,'initial':initial,'checks':checks,
        'new_modes':5,'same_potential_and_kinetic_law':True,'modes_from_unresolved_EL_residual_not_angle_fit':True,
        'physical_validation':False,'seconds':time.time()-start,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print('Enriched native derivative controls passed; seconds',round(time.time()-start,1),flush=True)

if __name__=='__main__':controls()
