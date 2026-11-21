'Current finite-field embedding, its first/second jets and reference torques.\n\nThe common target orientation psi is now a relative configuration variable.\nReader reference fields remain fixed in the received body recipe. Its momentum\nis not declared conserved when those references exert a torque.'
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,time
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from refit_dependencies import ReactiveContacts,install,ROOT
from precision_sum import AccurateSumAction
from geometry_conditional import left_jacobian
from mix import dump as _dump

def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value

def dump(path,value):return _dump(path,plain(value))

def rotate(a,R):
    z=a.copy();z[:,1:]=a[:,1:]@R.T;return z

class Configuration:
    def __init__(self):
        self.row=json.loads((ROOT/'resultats/SELECTION_RECALAGE_R341.json').read_text())['row']
        p=self.row['common_rotation_fit']['candidates'][0]
        self.beta=self.row['beta'];self.gamma=p['gamma'];self.kangle=p['kangle']
        self.model=install(ReactiveContacts('SF4','current','mobile'),self.beta,self.kangle,self.gamma)
        self.model.action.action=AccurateSumAction(self.beta)
        self.state=self.model.state;self.O0=np.concatenate([np.eye(3)[None],self.model.frames.copy()])
        self.ns=self.model.nstate;self.nx=len(self.state.x);assert self.ns==60 and self.nx==72
        self.nq=87;self.psi_slice=slice(84,87);self.frame_slice=slice(72,84)
        self.q0=np.r_[self.row['full_coordinates'],np.zeros(15)]
        self.psi=np.zeros(3);self.target_rotation=np.eye(3);self.raw=deque(maxlen=10)
        self._refresh=self.state.refresh;self._pull=self.state.pull
        def refresh(x):
            self._refresh(x)
            R=self.target_rotation
            self.state.fields=[rotate(a,R) for a in self.state.fields]
            self.state.tangent=[rotate(a,R) for a in self.state.tangent]
            self.state.pull_S=rotate(self.state.pull_S,R)
            self.state.m.fields=self.state.fields
            if self.model.action.record:
                self.model.action.rows=[(E,rotate(g,R)) for E,g in self.model.action.rows]
        def pull(i,g):
            self.raw.append((i,g.copy()))
            return self._pull(i,g)
        self.state.refresh=refresh;self.state.pull=pull
        self.last=None
    def potential(self,q):
        q=np.asarray(q,float);self.psi=q[self.psi_slice];self.target_rotation=Rotation.from_rotvec(self.psi).as_matrix()
        theta=q[self.frame_slice].reshape(4,3)
        self.frames=np.concatenate([self.O0[:1],np.array([Rotation.from_rotvec(t).as_matrix()@O for t,O in zip(theta,self.O0[1:],strict=True)])])
        self.model.frames=self.frames[1:].copy();self.raw.clear()
        E,g=self.model.evaluate(q[:self.nx]);assert [i for i,g in self.raw]==list(range(5))*2
        # ReactiveContacts pulls the five complete field covectors, then the
        # five self-energy covectors for its separate reduction diagnostic.
        # Keep the complete covectors, including contact/reference reactions.
        self.raw_total=list(self.raw)[:5]
        grad=np.zeros(self.nq);grad[:self.nx]=g
        fworld=np.array([-O@np.array(c['F_frame_conjugate_torque_native']) for O,c in zip(self.frames[1:],self.model.last['contacts'],strict=True)])
        grad[self.frame_slice]=np.concatenate([left_jacobian(t).T@f for t,f in zip(theta,fworld,strict=True)])
        torque=sum((np.sum(np.cross(n[:,1:],gn[:,1:]),axis=0) for n,(i,gn) in zip(self.state.fields,self.raw_total,strict=True)),np.zeros(3))
        grad[self.psi_slice]=left_jacobian(self.psi).T@torque
        self.last={'potential_energy_native':float(E),'gradient':grad.copy(),'target_reference_torque_native':torque.copy(),'geometry':self.state.m.geometry(self.model.last)}
        return float(E),grad
    def jets(self,q):
        'n_a,n_z,n_aa,n_az,n_zz in the received local finite embedding.'
        jets=[];mask=self.state.m.trial.mask;R=self.target_rotation
        for i,(original,template) in enumerate(zip(self.state.m.trial.fields,self.state.m.trial.templates,strict=True)):
            symbol,b,norm,parts=template;coordinates=q[50:52] if i==0 else q[50+2*i:52+2*i]
            amp=parts@coordinates;angle=norm*amp
            base=original*np.cos(angle)[:,None]+b*(amp*np.sinc(angle/np.pi))[:,None]
            a=-original*(norm*np.sin(angle))[:,None]+b*np.cos(angle)[:,None]
            aa=-norm[:,None]**2*base
            a[~mask]=0;aa[~mask]=0
            if i:
                jets.append({'a':rotate(a,R),'aa':rotate(aa,R),'parts':parts});continue
            radius=np.linalg.norm(base[:,1:],axis=1);active=mask&(radius>1e-12)
            unit=np.divide(base[:,1:],radius[:,None],out=np.zeros_like(base[:,1:]),where=radius[:,None]>1e-12)
            ra=np.einsum('ij,ij->i',unit,a[:,1:])
            raa=np.divide(np.sum(a[:,1:]**2,axis=1)+np.sum(base[:,1:]*aa[:,1:],axis=1)-ra**2,radius,out=np.zeros_like(radius),where=radius>1e-12)
            ua=np.divide(a[:,1:]-unit*ra[:,None],radius[:,None],out=np.zeros_like(unit),where=radius[:,None]>1e-12)
            uaa=np.divide(aa[:,1:]-unit*raa[:,None]-2*ua*ra[:,None],radius[:,None],out=np.zeros_like(unit),where=radius[:,None]>1e-12)
            polar=np.column_stack([-radius,base[:,0,None]*unit]);pa=np.column_stack([-ra,a[:,0,None]*unit+base[:,0,None]*ua])
            paa=np.column_stack([-raa,aa[:,0,None]*unit+2*a[:,0,None]*ua+base[:,0,None]*uaa])
            for p in [polar,pa,paa]:p[~active]=0
            turn=self.state.Z@q[:50];co=np.cos(turn)[:,None];si=np.sin(turn)[:,None]
            Na=a*co+pa*si;Nz=-base*si+polar*co;Naa=aa*co+paa*si;Naz=-a*si+pa*co;Nzz=-(base*co+polar*si)
            for t in [Nz,Naz,Nzz]:t[~active]=0
            assert np.max(abs(rotate(Na,R)-self.state.tangent[0]))<3e-12
            assert np.max(abs(rotate(Nz,R)-self.state.pull_S))<3e-12
            jets.append({'a':rotate(Na,R),'z':rotate(Nz,R),'aa':rotate(Naa,R),'az':rotate(Naz,R),'zz':rotate(Nzz,R),'parts':parts,'Z':self.state.Z})
        return jets

def controls():
    started=time.time();m=Configuration();q=m.q0.copy();E,g=m.potential(q);initial=m.last.copy();rows=[]
    for name,sl in [('fields',slice(0,60)),('centers',slice(60,72)),('frames',slice(72,84)),('target_reference',slice(84,87))]:
        v=np.zeros(87);v[sl]=np.sin(np.arange(len(v[sl]))*.31+.23);v/=np.linalg.norm(v);exact=float(g@v)
        for h in [2e-5,1e-5]:
            ep,_=m.potential(q+h*v);em,_=m.potential(q-h*v);fd=(ep-em)/(2*h);err=abs(fd-exact)
            assert err<8e-5*max(1.,abs(exact)),(name,h,err)
            rows.append({'family':name,'step':h,'analytic':exact,'finite_difference':fd,'absolute_defect':err})
    qn=q.copy();qn[72:84]=np.tile([.12,-.08,.10],4);qn[84:87]=[.17,-.11,.09]
    in_field,gn=m.potential(qn);v=np.zeros(87);v[72:]=np.sin(np.arange(15)*.31+.23);v/=np.linalg.norm(v);exact=float(gn@v)
    for h in [2e-5,1e-5]:
        fd=(m.potential(qn+h*v)[0]-m.potential(qn-h*v)[0])/(2*h);err=abs(fd-exact);assert err<8e-5*max(1.,abs(exact));rows.append({'family':'nonzero_rotation_charts','step':h,'analytic':exact,'finite_difference':fd,'absolute_defect':err})
    m.potential(q);j=m.jets(q);jetchecks=[]
    for i in [0,1]:
        v=np.zeros(87);sl=slice(0,52) if i==0 else slice(52,54);v[sl]=np.sin(np.arange(sl.stop-sl.start)*.31+.23);v/=np.linalg.norm(v)
        adot=j[i]['parts']@(v[50:52] if i==0 else v[52:54]);zdot=j[0]['Z']@v[:50] if i==0 else None
        expected=j[i]['aa']*adot[:,None]+(j[0]['az']*zdot[:,None] if i==0 else 0.)
        for h in [2e-5,1e-5]:
            m.potential(q+h*v);jp=m.jets(q+h*v)[i]['a'];m.potential(q-h*v);jm=m.jets(q-h*v)[i]['a'];fd=(jp-jm)/(2*h)
            err=float(np.max(abs(fd-expected)));assert err<2e-6
            jetchecks.append({'atom':i,'jet':'a_derivative','step':h,'maximum_absolute_defect':err})
    m.potential(q)
    out={'revision':343,'initial':initial,'potential_gradient_checks':rows,'field_second_jet_checks':jetchecks,
         'target_orientation_is_relative_to_held_references':True,'internal_momentum_not_assumed_conserved':True,
         'same_potential_at_psi_zero_as_R342':True,'physical_validation':False,'seconds':time.time()-started}
    dump(ROOT/'resultats/CONTROLES_REFERENCES_ET_JETS_R343.json',out)
    print('R343 reference/jet controls passed; target torque',initial['target_reference_torque_native'],'seconds',round(time.time()-started,1),flush=True)

if __name__=='__main__':controls()
