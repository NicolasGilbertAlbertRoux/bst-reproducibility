'Joint Routh relaxation: shapes, independent preparations, centres, momenta.\n\nTwo common SO3 factors, P=0, fixed prepared momentum magnitudes, no angular\ntarget in the objective. Exact R235 metric at each new state, not frozen M.\nThis is a conditional native reduction; physical source/clock remain open.'
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,ctypes,hashlib,json,shutil,subprocess,sys,time
import numpy as np
from rotations_spatial import SpatialQuadratic,centers_and_jacobian,embeddings,schur,skew,ROOT
from contacts_reactive import ReactiveContacts
from angles_internal import move,pullback
from mix import dump

class FastGram:
    def __init__(self):
        source=ROOT/'sources/metrique_rapide_R339.c';dep=source.with_name('metrique_rotations_R339.c');self.hash=hashlib.sha256(source.read_bytes()+dep.read_bytes()).hexdigest()
        target=source.with_name('gram_rapide_R339_'+self.hash[:12]+('.dylib' if sys.platform=='darwin' else '.so'))
        if not target.exists():
            cc=shutil.which('cc') or shutil.which('clang') or shutil.which('gcc');assert cc
            subprocess.run([cc,'-O3',*(['-dynamiclib'] if sys.platform=='darwin' else ['-shared','-fPIC']),str(source),'-lm','-o',str(target)],check=True)
        a=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');self.f=ctypes.CDLL(str(target)).bst_rotations_gram
        self.f.argtypes=[ctypes.c_int,ctypes.c_double,a,a];self.f.restype=ctypes.c_int
    def __call__(self,n,h):
        G=np.zeros((9,9))
        if self.f(round(len(n)**(1/3)),h,np.ascontiguousarray(n),G):raise ValueError('Kinetic card unknown')
        return G

class Routh:
    def __init__(self,branch,protocol='grids_alignees_stiffness_fixed',gamma_override=None):
        record=json.loads((ROOT/'results'/f'SF4_{branch}_BILAN_ROTATIONS_R339.json').read_text());row=next(r for r in record['protocols'] if r['protocol']==protocol)
        candidate=row['common_target_and_spatial_rotation']['coherent_velocity_Euler_candidates'][0];self.mu0=np.array(candidate['momentum_native']).reshape(2,3)
        self.magnitudes=np.linalg.norm(self.mu0,axis=1);self.directions=self.mu0/self.magnitudes[:,None];self.axes=[]
        for d in self.directions:
            t=np.cross(d,np.eye(3)[np.argmin(abs(d))]);t/=np.linalg.norm(t);self.axes.append(np.stack([t,np.cross(d,t)]))
        self.axes=np.array(self.axes);self.gamma=candidate['common_contact_stiffness_factor'] if gamma_override is None else gamma_override
        self.protocol=protocol;self.model=ReactiveContacts('SF4',law='fixed' if 'fixed' in protocol else 'current',F_reader='held' if 'fixed' in protocol else 'mobile',stiffness_factor=self.gamma)
        self.state=self.model.state;self.ns=self.model.nstate;self.nq=len(self.state.x);self.branch=branch
        self.x0=np.r_[row['potential']['full_coordinates'],np.zeros(4)];self.frames=np.tile(np.eye(3),(5,1,1)) if 'fixed' in protocol else np.concatenate([np.eye(3)[None],self.model.frames])
        self.fast=FastGram();self.kernel=SpatialQuadratic();self.calls=0;self.last=None
        self.bounds=np.vstack([self.state.system.bounds,np.tile([-np.pi,np.pi],(4,1))]);self.history=[];self.started=time.time()
    def momenta(self,x):
        angles=x[self.nq:].reshape(2,2);v=np.einsum('ia,iac->ic',angles,self.axes)
        return self.magnitudes[:,None]*move(self.directions,v),v
    def evaluate(self,x):
        energy,gU=self.model.evaluate(x[:self.nq]);s=self.state;m=s.m;X,J=centers_and_jacobian(s);N=len(X);embedding=embeddings(X,self.frames)
        B=np.zeros((3*N+6,9));B[:3*N,:3]=np.tile(np.eye(3),(N,1));B[3*N:3*N+3,3:6]=np.eye(3);B[-3:,-3:]=np.eye(3)
        E=[e@B for e in embedding];raw=[self.fast(n,self.model.h(i)) for i,n in enumerate(s.fields)];G=sum(e.T@g@e for e,g in zip(E,raw,strict=True));I=schur(G)
        eig=np.linalg.eigvalsh(I)
        if eig[0]<1e-9:raise ValueError('Reduced metric that is singular or unqualified')
        mu,orientation=self.momenta(x);w=np.linalg.solve(I,mu.ravel());u=-np.linalg.solve(G[-3:,-3:],G[-3:,:-3]@w);y=np.r_[w,u]
        kinetic=.5*w@mu.ravel();gT=np.zeros(self.nq);qsum=0.
        for i,(n,Gi,Ei) in enumerate(zip(s.fields,raw,E,strict=True)):
            vi=Ei@y;q,gn,_=self.kernel(n,self.model.h(i),vi);qsum+=q;sl=slice(0,52) if i==0 else slice(50+2*i,52+2*i);gT[sl]+=.5*s.pull(i,gn)
            if i:
                gu=2*(Gi@vi)[6:];gx=np.cross(self.frames[i]@gu,w[3:6]);gT[self.ns+3*(i-1):self.ns+3*i]+=.5*J[i-1].T@gx
        assert abs(qsum-2*kinetic)<2e-9*max(1.,abs(qsum)),(qsum,kinetic)
        gradient=np.zeros(len(x));gradient[:self.nq]=gU-gT
        back=pullback(self.directions,orientation,self.magnitudes[:,None]*w.reshape(2,3));gradient[self.nq:]=np.einsum('iac,ic->ia',self.axes,back).ravel()
        torque=np.cross(w.reshape(2,3),mu);self.calls+=1
        self.last={'effective_energy_native':float(energy+kinetic),'potential_energy_native':float(energy),'kinetic_energy_native':float(kinetic),
            'full_coordinates':x[:self.nq].tolist(),'momentum_chart_coordinates':x[self.nq:].tolist(),'prepared_momentum_magnitudes_native':self.magnitudes.tolist(),
            'momenta_native':mu.ravel().tolist(),'velocities_native':w.tolist(),'translation_velocity_native':u.tolist(),
            'state_force_inf_native':float(max(abs(gradient[:self.ns]))),'center_force_inf_native':float(max(abs(gradient[self.ns:self.nq]))),
            'Euler_torque_inf_native':float(np.max(abs(torque))),'momentum_orientation_gradient_inf':float(max(abs(gradient[self.nq:]))),
            'geometry':m.geometry(self.model.last),'reduced_metric_eigenvalue_range':eig[[0,-1]].tolist(),
            'gradient_native':gradient.tolist(),'common_contact_stiffness_factor':float(self.gamma),
            'maximum_nodal_turn':float(np.max(abs(s.Z@x[:50]))),'active_algorithmic_bounds':[i for i,(z,(lo,hi)) in enumerate(zip(x,self.bounds,strict=True)) if min(z-lo,hi-z)<1e-7]}
        return float(energy+kinetic),gradient
    def controls(self,x):
        E,g=self.evaluate(x);rows=[]
        for name,sl in [('field',slice(0,self.ns)),('centers',slice(self.ns,self.nq)),('momentum_orientation',slice(self.nq,len(x)))]:
            v=np.zeros(len(x));v[sl]=np.sin(np.arange(len(v[sl]))*.31+.21);v/=np.linalg.norm(v);exact=float(g@v)
            for h in [2e-5,1e-5]:
                plus,_=self.evaluate(x+h*v);minus,_=self.evaluate(x-h*v);fd=(plus-minus)/(2*h);error=abs(fd-exact)
                assert error<5e-5*max(1.,abs(exact)),(name,h,exact,fd,error)
                rows.append({'coordinate_family':name,'step':h,'analytic':exact,'finite_difference':fd,'absolute_defect':error})
        self.evaluate(x);return rows
    def relax(self,maxiter):
        x=self.x0.copy();energy,g=self.evaluate(x);initial=self.last.copy();scale=np.r_[np.full(50,.10),np.full(self.ns-50,.15),np.full(self.nq-self.ns,.12),np.full(4,.1)]
        z=x/scale;gz=g*scale;inverse=np.eye(len(x));invalid=0;message='Research budget achieved; retained status.'
        for iteration in range(maxiter):
            if max(abs(g[:self.nq]))<1e-6 and self.last['Euler_torque_inf_native']<1e-7:message='Native balance required in the declared reduction.';break
            direction=-inverse@gz
            if direction@gz>=0:direction=-gz;inverse=np.eye(len(x))
            step=direction*scale;alpha=min(1.,.08/max(float(np.max(abs(self.state.Z@step[:50]))),1e-30),.06/max(float(max(abs(step[50:self.ns]))),1e-30),.08/max(float(max(abs(step[self.ns:]))),1e-30))
            accepted=False
            for attempt in range(20):
                trial=np.clip((z+alpha*direction)*scale,self.bounds[:,0],self.bounds[:,1]);zt=trial/scale
                try:and_field,gt=self.evaluate(trial)
                except ValueError:invalid+=1;alpha*=.5;continue
                displacement=zt-z
                if and_field<=energy+1e-4*float(gz@displacement):accepted=True;break
                alpha*=.5
            if not accepted:message='Not acceptable not found; condition and unknown retained.';break
            difference=gt*scale-gz;product=float(displacement@difference)
            if product>1e-12*np.linalg.norm(displacement)*np.linalg.norm(difference):
                rho=1/product;A=np.eye(len(x))-rho*np.outer(displacement,difference);inverse=A@inverse@A.T+rho*np.outer(displacement,displacement)
            z=zt;x=trial;energy=and_field;g=gt;gz=gt*scale;entry={'iteration':iteration+1,'evaluations':self.calls,**self.last};self.history.append(entry)
            dump(ROOT/'results'/f'SF4_{self.branch}_{self.protocol}_REPRISE_ROUTH_R339.json',{'revision':339,'initial':initial,'history':self.history,'full':False,'physical_status':'UNKNOWN','physical_region_excluded':False})
            print(self.branch,self.protocol,'Routh',iteration+1,'E',round(energy,6),'g',round(self.last['state_force_inf_native'],6),'gc',round(self.last['center_force_inf_native'],6),'angle',round(self.last['geometry']['equatorial_angle_deg'],4),'s',round(time.time()-self.started,1),flush=True)
        self.evaluate(x)
        return {'initial':initial,'result':self.last,'message':message,'iterations':len(self.history),'invalid_numeric_trials':invalid,'history':self.history,'physical_status':'UNKNOWN','physical_region_excluded':False,
            'conditional_prepared_momenta_not_physical_spin_charge_or_clock':True,'all_field_F_frame_and_source_equations_not_solved':True,'no_angular_target_or_lagrange_holding_force':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('--branch',default='souple');p.add_argument('--protocol',default='grids_alignees_stiffness_fixed');p.add_argument('--maxiter',type=int,default=45);p.add_argument('--check-only',action='store_true');a=p.parse_args();m=Routh(a.branch,a.protocol)
    # The fast all-mode integrator must agree with the independently polarized
    # quadratic matrices on every atom at the inherited point.
    m.evaluate(m.x0);fastchecks=[]
    for i,n in enumerate(m.state.fields):
        slow=np.load(ROOT/'results'/f'SF4_{a.branch}_ATOME_{i}_ROTATIONS_R339.npz')['primitive_metric'];fast=m.fast(n,m.model.h(i));error=float(np.linalg.norm(fast-slow)/np.linalg.norm(slow));assert error<3e-11,error;fastchecks.append(error)
    checks=m.controls(m.x0);dump(ROOT/'results'/f'SF4_{a.branch}_{a.protocol}_CONTROLES_ROUTH_R339.json',{'fast_Gram_relative_defects_by_atom':fastchecks,'full_Routh_gradient_checks':checks})
    print('Routh full native gradient controls passed',len(checks),'s',round(time.time()-m.started,1),flush=True)
    if a.check_only:return
    result=m.relax(a.maxiter);result.update({'revision':339,'branch':a.branch,'protocol':a.protocol,'gradient_controls':checks,'fast_Gram_checks':fastchecks,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'fast_C_source_sha256':m.fast.hash,'seconds':time.time()-m.started})
    dump(ROOT/'results'/f'SF4_{a.branch}_{a.protocol}_ROUTH_R339.json',result)

if __name__=='__main__':main()
