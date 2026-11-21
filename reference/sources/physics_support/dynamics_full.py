'Finite-field Hamiltonian dynamics, using the received native action.\n\n87 shape/centre/frame/reference coordinates and one spatial SO(3) momentum.\nTranslation is reduced at P=0 after all intrinsic/rigid cross terms are built.\nNo fictitious field mass, velocity average or fixed target momentum is added.\nThis finite conditional realization is not a continuum or physical validation.'
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,ctypes,hashlib,json,time,itertools
import numpy as np
from fields_and_references import Configuration,dump
from geometry_conditional import left_jacobian
from refit_dependencies import ROOT,ARRAY,load_c,Gram,Kinetic,skew

def jacobian_jet(t):
    r=float(np.linalg.norm(t));K=skew(t);K2=K@K
    if r<1e-3:
        A=.5-r*r/24+r**4/720;B=1/6-r*r/120+r**4/5040
        A1=-1/12+r*r/180-r**4/6720;B1=-1/60+r*r/1260-r**4/60480
    else:
        A=(1-np.cos(r))/r**2;B=(r-np.sin(r))/r**3
        A1=(r*np.sin(r)-2*(1-np.cos(r)))/r**4
        B1=(3*np.sin(r)-r*(2+np.cos(r)))/r**5
    J=np.eye(3)+A*K+B*K2
    d=np.array([A*skew(e)+B*(skew(e)@K+K@skew(e))+A1*t[a]*K+B1*t[a]*K2 for a,e in enumerate(np.eye(3))])
    return J,d

def center_jets(m,c):
    'X, dX/dc, d2X/dc2, with c=(log radius,tangent coordinates).'
    X=[np.zeros(3)];J=[];H=[]
    for d0,axes,rb,ci in zip(m.D0,m.tangents,m.rbase,c.reshape(-1,3),strict=True):
        w=axes.T@ci[1:];t=float(np.linalg.norm(w));r=rb*np.exp(ci[0])
        if t<1e-3:
            S=1-t*t/6+t**4/120;C=-1/3+t*t/30-t**4/840;D=1/15-t*t/210+t**4/7560
        else:
            S=np.sin(t)/t;C=(t*np.cos(t)-np.sin(t))/t**3;D=((3-t*t)*np.sin(t)-3*t*np.cos(t))/t**5
        direction=d0*np.cos(t)+w*S;pos=r*direction
        dw=np.column_stack([-d0*S*w[a]+np.eye(3)[a]*S+w*C*w[a] for a in range(3)])
        ddw=np.empty((3,3,3))
        for a in range(3):
            for b in range(3):
                ddw[:,a,b]=-d0*(C*w[a]*w[b]+S*(a==b))+np.eye(3)[a]*C*w[b]+np.eye(3)[b]*C*w[a]+w*(D*w[a]*w[b]+C*(a==b))
        ji=np.column_stack([pos,r*dw@axes.T]);hi=np.empty((3,3,3));hi[:,0,0]=pos
        hi[:,0,1:]=ji[:,1:];hi[:,1:,0]=ji[:,1:]
        hi[:,1:,1:]=r*np.einsum('cab,ia,jb->cij',ddw,axes,axes)
        X.append(pos);J.append(ji);H.append(hi)
    return np.array(X),np.array(J),np.array(H)

class NodalKinetic:
    def __init__(self,beta,alpha=1.):
        source=ROOT/'sources/cinetique_complete_R343.c';self.alpha=alpha;self.beta=beta
        self.qfun,self.hash=load_c(source,'bst_general_q',[ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ARRAY,ARRAY,ARRAY,ARRAY,ARRAY,ARRAY,ctypes.POINTER(ctypes.c_double)])
        self.mfun,_=load_c(source,'bst_general_mass',[ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ARRAY,ARRAY,ARRAY,ARRAY])
    def quadratic(self,n,h,t,v):
        gn=np.zeros_like(n);gt=np.zeros_like(n);gv=np.zeros(9);Q=ctypes.c_double()
        status=self.qfun(round(len(n)**(1/3)),h,self.alpha,self.beta,np.ascontiguousarray(n),np.ascontiguousarray(t),np.ascontiguousarray(v),gn,gt,gv,ctypes.byref(Q))
        if status:raise ValueError('Kinetic gnomonic chart unknown')
        return Q.value,gn,gt,gv
    def mass(self,n,h):
        A=np.zeros((len(n),4,4));C=np.zeros((len(n),4,9));G=np.zeros((9,9))
        if self.mfun(round(len(n)**(1/3)),h,self.alpha,self.beta,np.ascontiguousarray(n),A,C,G):raise ValueError('Kinetic mass chart unknown')
        return A,C,G

def numpy_general(n,h,t,v,beta):
    'Independent scalar eight-corner quadrature, arbitrary intrinsic rate.'
    N=round(len(n)**(1/3));f=n.reshape(N,N,N,4);rate=t.reshape(N,N,N,4)
    xyz=np.moveaxis(np.indices((N,N,N)),0,-1)*h-.5*(N-1)*h;total=0.
    for bits in itertools.product([0,1],repeat=3):
        axes=tuple(k for k,b in enumerate(bits) if b)
        z=np.flip(f,axis=axes) if axes else f;rt=np.flip(rate,axis=axes) if axes else rate;p=np.flip(xyz,axis=axes) if axes else xyz
        base=z[:-1,:-1,:-1];ds=[]
        for a,nb in enumerate([z[1:,:-1,:-1],z[:-1,1:,:-1],z[:-1,:-1,1:]]):
            scalar=np.sum(base*nb,axis=-1);assert scalar.min()>0
            ds.append((-1)**bits[a]*(nb/scalar[...,None]-base)/h)
        d=np.stack(ds,axis=-2);ad=v[6:]+np.cross(v[3:6],p[:-1,:-1,:-1]);u=rt[:-1,:-1,:-1].copy()
        u[...,1:]-=np.cross(v[:3],base[...,1:]);u-=np.einsum('...a,...ac->...c',ad,d)
        tr=np.sum(d*d,axis=(-1,-2));norm=np.sum(u*u,axis=-1);pd=np.einsum('...c,...ac->...a',u,d)
        total+=h**3/8*np.sum((1+beta*tr)*norm-beta*np.sum(pd*pd,axis=-1))
    return float(total)

class Dynamics:
    def __init__(self):
        self.configuration=Configuration();self.nq=87;self.kernel=NodalKinetic(self.configuration.beta)
        self.keep=np.r_[np.arange(3),np.arange(6,93)];self.translation=np.arange(3,6);self.calls=0
    def prepare(self,q):
        c=self.configuration;self.U,self.gU=c.potential(q);self.jets=c.jets(q)
        self.X,self.JX,self.HX=center_jets(c.state.m,q[60:72]);self.frames=c.frames
        self.Jpsi,self.dJpsi=jacobian_jet(q[84:87]);self.frame_jets=[jacobian_jet(t) for t in q[72:84].reshape(4,3)]
        self.E=[];self.sl=[];self.G=np.zeros((93,93));self.local_grams=[]
        for i,(n,j,O) in enumerate(zip(c.state.fields,self.jets,self.frames,strict=True)):
            A,C,G=self.kernel.mass(n,c.model.h(i));parts=j['parts'];a=j['a']
            aa=np.einsum('ni,nij,nj->n',a,A,a);cross_a=np.einsum('ni,nij->nj',a,C)
            if i==0:
                Z=j['Z'];z=j['z'];zz=np.einsum('ni,nij,nj->n',z,A,z);za=np.einsum('ni,nij,nj->n',z,A,a)
                cross_z=np.einsum('ni,nij->nj',z,C)
                F=np.block([[Z.T@(zz[:,None]*Z),Z.T@(za[:,None]*parts)],[parts.T@(za[:,None]*Z),parts.T@(aa[:,None]*parts)]])
                B=np.vstack([Z.T@cross_z,parts.T@cross_a]);sl=slice(0,52)
            else:F=parts.T@(aa[:,None]*parts);B=parts.T@cross_a;sl=slice(50+2*i,52+2*i)
            p=F.shape[0];local=np.block([[G,B.T],[B,F]]);E=np.zeros((9+p,93))
            E[:3,6+84:6+87]=-self.Jpsi
            E[3:6,:3]=O.T;E[6:9,:3]=-O.T@skew(self.X[i]);E[6:9,3:6]=O.T
            if i:
                k=6+72+3*(i-1);E[3:6,k:k+3]=O.T@self.frame_jets[i-1][0]
                k=6+60+3*(i-1);E[6:9,k:k+3]=O.T@self.JX[i-1]
            E[9:,6+sl.start:6+sl.stop]=np.eye(p)
            self.G+=E.T@local@E;self.E.append(E);self.sl.append(sl);self.local_grams.append(G)
        self.G=.5*(self.G+self.G.T);T=self.G[3:6,3:6];C=self.G[np.ix_(self.keep,self.translation)]
        self.R=-np.linalg.solve(T,C.T);self.M=self.G[np.ix_(self.keep,self.keep)]+C@self.R;self.M=.5*(self.M+self.M.T)
        return self.M
    def kinetic(self,q,v,prepared=False,gradient=True):
        if not prepared:self.prepare(q)
        c=self.configuration;v=np.asarray(v);Q=0.;g=np.zeros(87);gv=np.zeros(93);target=np.zeros(3)
        for i,(n,j,E,O,sl) in enumerate(zip(c.state.fields,self.jets,self.E,self.frames,self.sl,strict=True)):
            local=E@v;adot=j['parts']@v[6+sl.stop-2:6+sl.stop];t=j['a']*adot[:,None]
            zdot=j['Z']@v[6:56] if i==0 else None
            if i==0:t+=j['z']*zdot[:,None]
            value,gn,gt,gl=self.kernel.quadratic(n,c.model.h(i),t,local[:9]);Q+=value
            da=j['aa']*adot[:,None]+(j['az']*zdot[:,None] if i==0 else 0.)
            ca=np.einsum('ni,ni->n',gn,j['a'])+np.einsum('ni,ni->n',gt,da)
            g[sl.stop-2:sl.stop]+=j['parts'].T@ca
            vt=j['parts'].T@np.einsum('ni,ni->n',gt,j['a'])
            localcov=np.r_[gl,np.zeros(sl.stop-sl.start)];localcov[-2:]=vt
            if i==0:
                dz=j['az']*adot[:,None]+j['zz']*zdot[:,None]
                cz=np.einsum('ni,ni->n',gn,j['z'])+np.einsum('ni,ni->n',gt,dz)
                g[:50]+=j['Z'].T@cz;localcov[9:59]=j['Z'].T@np.einsum('ni,ni->n',gt,j['z'])
            gv+=E.T@localcov
            target+=np.sum(np.cross(n[:,1:],gn[:,1:])+np.cross(t[:,1:],gt[:,1:]),axis=0)
            if i:
                omega=v[:3];ucov=O@gl[6:9];center=slice(60+3*(i-1),60+3*i);cdot=v[6+center.start:6+center.stop]
                g[center]+=self.JX[i-1].T@np.cross(ucov,omega)+np.einsum('cab,b,c->a',self.HX[i-1],cdot,ucov)
                frame=slice(72+3*(i-1),72+3*i);tdot=v[6+frame.start:6+frame.stop];Jl,dJl=self.frame_jets[i-1]
                force=O@(np.cross(gl[3:6],local[3:6])+np.cross(gl[6:9],local[6:9]))
                g[frame]+=Jl.T@force+np.einsum('acb,b,c->a',dJl,tdot,O@gl[3:6])
            g[84:87]-=np.einsum('acb,b,c->a',self.dJpsi,v[90:93],gl[:3])
        g[84:87]+=self.Jpsi.T@target
        return float(Q),g,gv
    def evaluate(self,y,diagnostics=False):
        y=np.asarray(y);q=y[:87];moment=np.r_[y[174:177],y[87:174]];self.prepare(q)
        v=np.linalg.solve(self.M,moment);full=np.zeros(93);full[self.keep]=v;full[3:6]=self.R@v
        Q,gQ,gvel=self.kinetic(q,full,prepared=True);T=.5*float(moment@v);scale=max(1.,abs(Q))
        massdef=abs(Q-2*T)/scale;momentdef=float(np.max(abs(.5*gvel-self.G@full)))
        if massdef>3e-10 or momentdef>2e-7*max(1.,np.max(abs(moment))):raise ValueError(('Kinetic embedding mismatch',massdef,momentdef))
        gH=self.gU-.5*gQ;mu=moment[:3];dy=np.r_[v[3:],-gH,np.cross(mu,v[:3])];self.calls+=1
        self.last={'H_native':float(self.U+T),'U_native':float(self.U),'T_native':T,'time_is_native_not_SI':True,
                   'q':q.copy(),'canonical_momenta':moment[3:].copy(),'spatial_body_momentum':mu.copy(),
                   'configuration_velocity':v[3:].copy(),'spatial_velocity':v[:3].copy(),'translation_velocity':full[3:6].copy(),
                   'Hamiltonian_configuration_gradient':gH.copy(),'target_potential_torque':self.configuration.last['target_reference_torque_native'].copy(),
                   'geometry':self.configuration.last['geometry'],'native_Q_mass_relative_defect':massdef,
                   'native_momentum_inf_defect':momentdef,'translation_momentum_inf_defect':float(np.max(abs((self.G@full)[3:6])))}
        if diagnostics:
            eig=np.linalg.eigvalsh(self.M);self.last['mass_eigenvalue_range']=eig[[0,-1]].tolist();self.last['mass_condition']=float(eig[-1]/eig[0])
            Jdot=np.einsum('acb,a->cb',self.dJpsi,dy[84:87]);ell=-np.linalg.solve(self.Jpsi.T,y[87+84:174])
            elldot=-np.linalg.solve(self.Jpsi.T,dy[87+84:174]+Jdot.T@ell)
            self.last['old_target_momentum']=ell.tolist();self.last['old_target_momentum_time_derivative']=elldot.tolist()
            self.last['target_reaction_balance_inf_defect']=float(np.max(abs(elldot-self.configuration.last['target_reference_torque_native'])))
            self.last['spatial_Casimir_time_derivative']=float(2*mu@dy[174:177])
            self.last['Hamiltonian_time_derivative']=float(gH@dy[:87]+v[3:]@dy[87:174]+v[:3]@dy[174:177])
        return dy,self.last['H_native']
    def initial(self):
        q=self.configuration.q0.copy();self.prepare(q);v=np.zeros(90);w=np.array(self.configuration.row['common_rotation_fit']['candidates'][0]['velocity_native'])
        v[:3]=w[3:6];v[3+84:3+87]=-w[:3];moment=self.M@v
        return np.r_[q,moment[3:],moment[:3]]

def controls():
    start=time.time();d=Dynamics();q=d.configuration.q0.copy();d.prepare(q);c=d.configuration;n=c.state.fields[0];j=d.jets[0];h=c.model.h(0)
    vlocal=np.array([.17,-.11,.13,.023,-.031,.017,.08,-.03,.04]);rate=j['a']*.013+j['z']*.017
    Q,gn,gt,gv=d.kernel.quadratic(n,h,rate,vlocal);npQ=numpy_general(n,h,rate,vlocal,c.beta);npdef=abs(Q-npQ)/max(1.,abs(Q));assert npdef<3e-11
    zero=np.zeros_like(n);Q0,g0,_,v0=d.kernel.quadratic(n,h,zero,vlocal);old,oldg,_=Kinetic(c.beta)(n,h,vlocal)
    assert abs(Q0-old)<3e-9*max(1.,abs(old));assert np.max(abs(g0-oldg))<3e-9
    gram=Gram(c.beta)(n,h);gramdef=float(np.max(abs(d.local_grams[0]-gram)));assert gramdef<3e-9
    localchecks=[]
    for name,tv,vv in [('intrinsic',j['a']*.023,np.zeros(9)),('rigid',np.zeros_like(n),np.sin(np.arange(9)+.23)*.037)]:
        exact=float(np.sum(gt*tv)+gv@vv)
        for eps in [2e-5,1e-5]:
            fd=(d.kernel.quadratic(n,h,rate+eps*tv,vlocal+eps*vv)[0]-d.kernel.quadratic(n,h,rate-eps*tv,vlocal-eps*vv)[0])/(2*eps)
            err=abs(exact-fd);assert err<3e-7*max(1.,abs(exact));localchecks.append({'family':name,'step':eps,'analytic':exact,'finite_difference':fd,'absolute_defect':err})
    eig=np.linalg.eigvalsh(d.M);assert eig[0]>1e-10,('Unqualified full kinetic rank',eig[:8])
    base={'independent_numpy_Q_relative_defect':npdef,'legacy_Q_absolute_defect':abs(Q0-old),'legacy_gradient_inf_defect':float(np.max(abs(g0-oldg))),
          'legacy_Gram_inf_defect':gramdef,'local_velocity_checks':localchecks,'mass_eigenvalues':eig.tolist(),'mass_condition':float(eig[-1]/eig[0])}
    dump(ROOT/'resultats/CONTROLES_MASSE_R343.json',{'revision':343,'full':False,'base':base})
    print('General mass / nodal controls passed; condition',base['mass_condition'],'seconds',round(time.time()-start,1),flush=True)
    qn=q.copy();qn[72:84]=np.tile([.12,-.08,.10],4);qn[84:87]=[.17,-.11,.09]
    # Nonzero rates expose Hessian and moving-frame terms absent in rigid tests.
    v=np.sin(np.arange(93)*.37+.2)*.013;d.prepare(qn);Q,g,gvel=d.kinetic(qn,v,prepared=True);checks=[]
    for name,sl in [('fields',slice(0,60)),('centers',slice(60,72)),('frames',slice(72,84)),('target_reference',slice(84,87))]:
        direction=np.zeros(87);direction[sl]=np.sin(np.arange(len(direction[sl]))*.31+.23);direction/=np.linalg.norm(direction);exact=float(g@direction)
        for eps in [2e-5,1e-5]:
            qp=d.kinetic(qn+eps*direction,v)[0];qm=d.kinetic(qn-eps*direction,v)[0];fd=(qp-qm)/(2*eps);err=abs(fd-exact)
            assert err<3e-5*max(1.,abs(exact)),(name,eps,exact,fd,err)
            checks.append({'family':name,'step':eps,'analytic':exact,'finite_difference':fd,'absolute_defect':err})
        print('General kinetic configuration derivative',name,'passed',flush=True)
    y=d.initial();dy,H=d.evaluate(y,diagnostics=True);initial=d.last.copy();assert initial['target_reaction_balance_inf_defect']<2e-6
    # Native H directional differences at fixed canonical momenta.
    hchecks=[];gH=d.last['Hamiltonian_configuration_gradient'].copy()
    for name,sl in [('fields',slice(0,60)),('centers',slice(60,72)),('frames',slice(72,84)),('target_reference',slice(84,87))]:
        direction=np.zeros(177);direction[sl]=np.sin(np.arange(len(direction[sl]))*.31+.23);direction/=np.linalg.norm(direction);exact=float(gH@direction[:87])
        eps=1e-5;ep=d.evaluate(y+eps*direction)[1];em=d.evaluate(y-eps*direction)[1];fd=(ep-em)/(2*eps);err=abs(fd-exact)
        assert err<1e-4*max(1.,abs(exact)),(name,exact,fd,err)
        hchecks.append({'family':name,'step':eps,'analytic':exact,'finite_difference':fd,'absolute_defect':err})
    d.evaluate(y,diagnostics=True)
    out={'revision':343,'full':True,'base':base,'nonzero_intrinsic_rate_configuration_checks':checks,'Hamiltonian_checks':hchecks,
         'initial':initial,'source_sha256':d.kernel.hash,'finite_coordinates':87,'reduced_mass_dimension':90,'canonical_plus_LiePoisson_state_dimension':177,
         'same_alpha_beta_action_in_potential_and_kinetic':True,'translation_reduced_after_intrinsic_cross_terms':True,
         'target_reference_reactions_included':True,'physical_validation':False,'seconds':time.time()-start}
    dump(ROOT/'resultats/CONTROLES_MASSE_R343.json',out);np.savez_compressed(ROOT/'resultats/INITIAL_DYNAMIQUE_R343.npz',state=y,mass=d.M,rhs=dy)
    print('Full R343 controls passed; seconds',round(time.time()-start,1),'target reaction',initial['target_reaction_balance_inf_defect'],flush=True)

if __name__=='__main__':
    controls()
