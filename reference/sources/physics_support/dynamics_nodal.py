'Unrestricted finite-grid field EL dynamics, held received references.\n\nState: five unit nodal S3 fields and their tangent rates, 24 centre/frame\ncoordinates, 30 body/relative velocities. The common target coordinate is\nremoved as an EXACT redundancy of unrestricted fields with HELD references.\nThe kinetic metric and potential are the received native formulas. No\nreference rotor, new mass, force, fitted geometry, or clock is introduced.'
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from scipy.spatial.transform import Rotation
from fields_and_references import Configuration,ROOT,dump
from dynamics_full import NodalKinetic,center_jets,jacobian_jet
from refit_dependencies import skew
from solve_acceleration_nodal import constrained_inverse

class FullNodalDynamics:
    def __init__(self,derivative_step=2e-5):
        self.c=Configuration();self.qbase=self.c.q0.copy();self.c.potential(self.qbase)
        self.base_fields=np.array(self.c.state.fields);self.shape=self.base_fields.shape;self.nfield=self.base_fields.size
        self.kernel=NodalKinetic(self.c.beta);self.eps=derivative_step;self.calls=0;self.current_fields=self.base_fields
        self.h=[self.c.model.h(i) for i in range(5)]
        def refresh(x):
            self.c.state.x=np.asarray(x).copy();self.c.state.fields=list(self.current_fields);self.c.state.m.fields=self.c.state.fields
            self.c.model.action.rows=[self.c.model.action.action(n,h) for n,h in zip(self.current_fields,self.h,strict=True)]
            self.c.state.m.self_energies=[row[0] for row in self.c.model.action.rows]
        self.c.state.refresh=refresh
    def pack(self,n,t,x,v):return np.r_[np.asarray(n).ravel(),np.asarray(t).ravel(),x,v]
    def unpack(self,y):
        m=self.nfield
        return y[:m].reshape(self.shape),y[m:2*m].reshape(self.shape),y[2*m:2*m+24],y[2*m+24:]
    def project(self,y):
        n,t,x,v=self.unpack(y);norm=np.linalg.norm(n,axis=2);normal=n/norm[:,:,None]
        tangent=t-normal*np.sum(normal*t,axis=2)[:,:,None]
        defect=max(float(np.max(abs(norm-1))),float(np.max(abs(np.sum(normal*t,axis=2)))))
        return self.pack(normal,tangent,x,v),defect
    def initial(self):
        initial=json.loads((ROOT/'resultats/CONTROLES_ENRICHIS_R345.json').read_text())['initial']
        qdot=np.array(initial['configuration_velocity']);assert np.max(abs(qdot[:84]))<1e-12
        angular=qdot[84:87];t=np.zeros_like(self.base_fields);t[:,:,1:]=np.cross(angular,self.base_fields[:,:,1:])
        v=np.r_[initial['spatial_velocity'],initial['translation_velocity'],qdot[60:84]]
        return self.pack(self.base_fields,t,self.qbase[60:84],v)
    def geometry(self,x):
        X,JX,HX=center_jets(self.c.state.m,x[:12]);theta=x[12:].reshape(4,3)
        frames=np.concatenate([self.c.O0[:1],np.array([Rotation.from_rotvec(t).as_matrix()@O for t,O in zip(theta,self.c.O0[1:],strict=True)])])
        jets=[jacobian_jet(t) for t in theta];Es=[]
        for i,O in enumerate(frames):
            E=np.zeros((9,30));E[3:6,:3]=O.T;E[6:9,:3]=-O.T@skew(X[i]);E[6:9,3:6]=O.T
            if i:
                j=i-1;E[6:9,6+3*j:9+3*j]=O.T@JX[j];E[3:6,18+3*j:21+3*j]=O.T@jets[j][0]
            Es.append(E)
        return X,JX,HX,frames,jets,Es
    def kinetic(self,n,t,x,v,gradients=False):
        X,JX,HX,frames,jets,Es=self.geometry(x);Q=0.;pr=np.zeros(30);pi=[];gns=[];gq=np.zeros(24)
        for i,(field,rate,O,E,h) in enumerate(zip(n,t,frames,Es,self.h,strict=True)):
            local=E@v;value,gn,gt,gl=self.kernel.quadratic(field,h,rate,local);Q+=value;pi.append(.5*gt);pr+=.5*E.T@gl
            if gradients:
                gns.append(.5*gn)
                if i:
                    j=i-1;ucov=O@gl[6:9];cdot=v[6+3*j:9+3*j];tdot=v[18+3*j:21+3*j];Jl,dJl=jets[j]
                    gq[3*j:3*j+3]+=.5*(JX[j].T@np.cross(ucov,v[:3])+np.einsum('cab,b,c->a',HX[j],cdot,ucov))
                    force=O@(np.cross(gl[3:6],local[3:6])+np.cross(gl[6:9],local[6:9]))
                    gq[12+3*j:15+3*j]+=.5*(Jl.T@force+np.einsum('acb,b,c->a',dJl,tdot,O@gl[3:6]))
        return {'T':.5*Q,'pi':pi,'p':pr,'gn':gns,'gq':gq,'E':Es}
    def potential(self,n,x):
        self.current_fields=n;q=self.qbase.copy();q[60:84]=x;U,g=self.c.potential(q)
        return U,[row[1] for row in self.c.raw_total],g[60:84],self.c.last['geometry']
    def value(self,y):
        y,defect=self.project(y);n,t,x,v=self.unpack(y);k=self.kinetic(n,t,x,v);U,g,gq,geometry=self.potential(n,x)
        return {'H_native':float(U+k['T']),'U_native':U,'T_native':k['T'],'body_momentum':k['p'][:6],
                'spatial_angular_Casimir_native':float(k['p'][:3]@k['p'][:3]),'constraint_projection_defect':defect,
                'geometry':geometry,'maximum_field_rate_native':float(np.max(np.linalg.norm(t,axis=2)))}
    def evaluate(self,y,diagnostics=False):
        start=time.time();y,projection_defect=self.project(y);n,t,x,v=self.unpack(y);base=self.kinetic(n,t,x,v,True)
        U,gU,gUq,geometry=self.potential(n,x);eps=self.eps;speed=np.linalg.norm(t,axis=2);angle=eps*speed
        co=np.cos(angle)[:,:,None];si=(eps*np.sinc(angle/np.pi))[:,:,None]
        plus=self.kinetic(n*co+t*si,t*co-n*(speed*speed)[:,:,None]*si,x+eps*v[6:],v)
        minus=self.kinetic(n*co-t*si,t*co+n*(speed*speed)[:,:,None]*si,x-eps*v[6:],v)
        r_r=(plus['p']-minus['p'])/(2*eps);p=base['p'];r_r[:3]-=np.cross(p[:3],v[:3])+np.cross(p[3:6],v[3:6]);r_r[3:6]-=np.cross(p[3:6],v[:3])
        r_r[6:]+=-base['gq']+gUq
        D=np.zeros((30,30));S=np.zeros_like(D);rhs=-r_r.copy();cache=[];nodal=[]
        for i,(field,E,h) in enumerate(zip(n,base['E'],self.h,strict=True)):
            r=(plus['pi'][i]-minus['pi'][i])/(2*eps)-base['gn'][i]+gU[i]
            r-=field*np.sum(field*r,axis=1)[:,None];nodal.append(r)
            A,C,G=self.kernel.mass(field,h);H=constrained_inverse(A,field);B=C@E;Di=E.T@G@E
            D+=Di;S+=Di-np.einsum('nca,ncb->ab',B,H@B);rhs+=np.einsum('nca,nc->a',B,np.einsum('nij,nj->ni',H,r))
            cache.append((A,C,H,E))
        S=.5*(S+S.T)
        # Cholesky is a rank/positivity gate, not a regularizer.
        L=np.linalg.cholesky(S);acc=np.linalg.solve(L.T,np.linalg.solve(L,rhs));field_acc=[];balance=r_r+D@acc;maximum=0.
        for field,rate,r,(A,C,H,E) in zip(n,t,nodal,cache,strict=True):
            B=C@E;at=-np.einsum('nij,nj->ni',H,r+B@acc);balance+=np.einsum('nca,nc->a',B,at)
            left=r+np.einsum('nij,nj->ni',A,at)+B@acc;left-=field*np.sum(field*left,axis=1)[:,None]
            maximum=max(maximum,float(np.max(abs(left))));field_acc.append(at-field*np.sum(rate*rate,axis=1)[:,None])
        rigid_defect=float(np.max(abs(balance)));assert maximum<2e-7 and rigid_defect<3e-6,(maximum,rigid_defect)
        deriv=self.pack(t,np.array(field_acc),v[6:],acc);Hdot=0.
        for gu,gt,pi,rate,at in zip(gU,base['gn'],base['pi'],t,field_acc,strict=True):Hdot+=float(np.sum((gu+gt)*rate)+np.sum(pi*at))
        Hdot+=float((gUq+base['gq'])@v[6:]+base['p']@acc)
        self.calls+=1;self.last={'H_native':float(U+base['T']),'U_native':U,'T_native':base['T'],
            'Hamiltonian_time_derivative_native':Hdot,'field_EL_inf_native':maximum,'rigid_EL_inf_native':rigid_defect,
            'body_momentum':p[:6],'spatial_angular_Casimir_native':float(p[:3]@p[:3]),'geometry':geometry,
            'maximum_field_acceleration_native':float(np.max(np.linalg.norm(field_acc,axis=2))),
            'constraint_projection_defect':projection_defect,'RHS_seconds':time.time()-start,'held_received_references':True,
            'free_field_tangent_components':len(n)*len(n[0])*3,'physical_validation':False}
        if diagnostics:self.last['Schur_eigenvalue_range_native']=np.linalg.eigvalsh(S)[[0,-1]]
        return deriv

def controls():
    d=FullNodalDynamics();y=d.initial();dy=d.evaluate(y,True);first=dict(d.last)
    print('Full nodal RHS',first['RHS_seconds'],'seconds','Hdot',first['Hamiltonian_time_derivative_native'],'field EL',first['field_EL_inf_native'],flush=True)
    assert abs(first['Hamiltonian_time_derivative_native'])<2e-5
    d.eps=1e-5;fine=d.evaluate(y,True);second=dict(d.last);difference=float(np.max(abs(dy-fine)))
    n,t,x,v=d.unpack(y);reverse=d.pack(n,-t,x,-v);rev=d.evaluate(reverse);rn,rt,rx,rv=d.unpack(rev);fn,ft,fx,fv=d.unpack(fine)
    reversal=max(float(np.max(abs(rn+fn))),float(np.max(abs(rt-ft))),float(np.max(abs(rx+fx))),float(np.max(abs(rv-fv))))
    assert difference<3e-5 and reversal<2e-7,(difference,reversal)
    value=d.value(y);assert abs(value['H_native']-first['H_native'])<3e-10
    # Independent directional H difference along the computed full-field flow.
    h=1e-5;fd=(d.value(y+h*fine)['H_native']-d.value(y-h*fine)['H_native'])/(2*h)
    assert abs(fd-second['Hamiltonian_time_derivative_native'])<3e-5,(fd,second['Hamiltonian_time_derivative_native'])
    result={'revision':345,'full':True,'initial':first,'refined_initial':second,
        'two_steps': [2e-5,1e-5],'rhs_step_refinement_inf_native':difference,'velocity_reversal_RHS_inf_defect':reversal,
        'directional_H_finite_difference_native':fd,'same_H_replay_absolute_defect':abs(value['H_native']-first['H_native']),
        'no_reference_auxiliary_assumption_needed_for_held_preparation':True,
        'same_native_action_coefficients_and_initial_physical_field_rates':True,'no_mass_added_or_eigenvalue_clipped':True,
        'pointwise_full_grid_dynamics_not_empirical_validation':True,'physical_validation':False,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dump(ROOT/'resultats/CONTROLES_DYNAMIQUE_NODALE_R345.json',result)
    np.savez_compressed(ROOT/'resultats/INITIAL_NODAL_R345.npz',state=y,rhs=fine)
    print('Full nodal initial controls passed; FD difference',difference,'reversal',reversal,'H FD',fd,flush=True)

if __name__=='__main__':controls()
