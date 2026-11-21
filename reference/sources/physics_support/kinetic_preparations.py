'Recompute native inertia and its state derivatives at the R336 preparations.\n\nThis tests specified co-transported internal-rotation preparations, not a\nphysical electric charge or an already constructed molecular trajectory.'
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,ctypes,hashlib,json,shutil,subprocess,sys,time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from relax_states import JointStates
from mix import dump
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sources'))
from kinetic_native import kinetic_blocks
from dynamics_profile import Metric as LegacyMetric

class Quadratic:
    def __init__(self):
        source=ROOT/'sources/metrique_et_gradient_R337.c';self.hash=hashlib.sha256(source.read_bytes()).hexdigest()
        suffix='.dylib' if sys.platform=='darwin' else '.so';target=source.with_name('metrique_R337_'+self.hash[:12]+suffix)
        if not target.exists():
            compiler=shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')
            if compiler is None:raise RuntimeError('Compiler C required')
            mode=['-dynamiclib'] if sys.platform=='darwin' else ['-shared','-fPIC']
            subprocess.run([compiler,'-O3',*mode,str(source),'-lm','-o',str(target)],check=True)
        array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        f=ctypes.CDLL(str(target)).bst_metric_q_gradient
        f.argtypes=[ctypes.c_int,ctypes.c_double,array,array,array,array,ctypes.POINTER(ctypes.c_double)];f.restype=ctypes.c_int;self.f=f
    def __call__(self,field,h,velocity):
        field=np.ascontiguousarray(field);velocity=np.asarray(velocity,float);gradient=np.zeros_like(field);q=ctypes.c_double()
        if self.f(round((len(field))**(1/3)),h,field,np.ascontiguousarray(velocity[:3]),np.ascontiguousarray(velocity[3:]),gradient,ctypes.byref(q)):
            raise ValueError('Inoperable kinetic gnomonic card')
        return q.value,gradient

class State:
    def __init__(self,label):
        if label=='SF4':row=json.loads((ROOT/'resultats/INVERSION_ADMISSIBLE_R336.json').read_text())['rows'][0]
        else:row=next(r for r in json.loads((ROOT/'resultats/INVERSION_PREPARATIONS_R336.json').read_text())['rows'] if r['label']=='SF6')
        self.label=label;point=row['free_centers_result'];self.recipe={**row['recipe'],'center_start':point['center_coordinates']}
        self.system=JointStates(label,self.recipe['kangle']);self.x=self.system.initial(self.recipe)
        self.refresh(self.x)
    def refresh(self,x):
        self.x=np.asarray(x).copy()
        self.energy,self.gradient=self.system.evaluate(self.x);self.m=self.system.m;self.fields=[f.copy() for f in self.m.fields]
        self.Z=self.system.Z.copy();self.parts=[];self.tangent=[];self.pull_S=None
        qS=self.x[50:52];qF=self.x[52:self.system.nstate].reshape(self.m.N,2)
        for i,(original,template) in enumerate(zip(self.m.trial.fields,self.m.trial.templates,strict=True)):
            symbol,b,norm,parts=template;coords=qS if i==0 else qF[i-1]
            amp=parts@coords;theta=norm*amp
            base=original*np.cos(theta)[:,None]+b*(amp*np.sinc(theta/np.pi))[:,None];base[~self.m.trial.mask]=original[~self.m.trial.mask]
            derivative=-original*(norm*np.sin(theta))[:,None]+b*np.cos(theta)[:,None];derivative[~self.m.trial.mask]=0
            if i==0:
                radius=np.linalg.norm(base[:,1:],axis=1);active=self.m.trial.mask&(radius>1e-12)
                unit=np.divide(base[:,1:],radius[:,None],out=np.zeros_like(base[:,1:]),where=radius[:,None]>1e-12)
                polar=np.column_stack([-radius,base[:,0,None]*unit]);polar[~active]=0;self.Z[~active]=0;turn=self.Z@self.x[:50]
                shape_tangent=-base*np.sin(turn)[:,None]+polar*np.cos(turn)[:,None];shape_tangent[~active]=0
                dr=np.einsum('ij,ij->i',unit,derivative[:,1:]);of=np.divide(derivative[:,1:]-unit*dr[:,None],radius[:,None],out=np.zeros_like(unit),where=radius[:,None]>1e-12)
                depolar=np.column_stack([-dr,derivative[:,0,None]*unit+base[:,0,None]*of]);depolar[~active]=0
                derivative=derivative*np.cos(turn)[:,None]+depolar*np.sin(turn)[:,None]
                self.pull_S=shape_tangent
            self.parts.append(parts);self.tangent.append(derivative)
    def pull(self,i,g):
        q=np.einsum('ij,ij->i',g,self.tangent[i])@self.parts[i]
        return np.r_[self.Z.T@np.einsum('ij,ij->i',g,self.pull_S),q] if i==0 else q
    def field_at(self,x,i):
        self.system.evaluate(x);return self.system.m.fields[i].copy()

def primitive(kernel,state,i,need_derivative=True):
    n=state.fields[i];h=float(state.m.d[('S' if i==0 else 'F')+'_h']);p=52 if i==0 else 2
    G=np.zeros((6,6));D=np.zeros((p,6,6));values=[];grads=[]
    for j in range(6):
        velocity=np.eye(6)[j];q,g=kernel(n,h,velocity);G[j,j]=q;values.append(q)
        pull=state.pull(i,g) if need_derivative else np.zeros(p);D[:,j,j]=pull;grads.append(pull)
    for j in range(6):
        for k in range(j+1,6):
            q,g=kernel(n,h,np.eye(6)[j]+np.eye(6)[k]);value=.5*(q-values[j]-values[k]);G[j,k]=G[k,j]=value
            pull=.5*(state.pull(i,g)-grads[j]-grads[k]) if need_derivative else np.zeros(p);D[:,j,k]=D[:,k,j]=pull
    return G,D

def assemble(primitives):
    count=len(primitives);size=3*count;G=np.zeros((size+3,size+3));embeddings=[]
    for i,local in enumerate(primitives):
        E=np.zeros((6,size+3));E[:3,3*i:3*i+3]=np.eye(3);E[3:,size:]=np.eye(3);G+=E.T@local@E;embeddings.append(E)
    C=G[:size,size:];Ti=np.linalg.inv(G[size:,size:]);I=G[:size,:size]-C@Ti@C.T
    return (I+I.T)/2,G,embeddings

def reduce_derivatives(raw,derivatives):
    I,G,embeddings=assemble(raw);size=len(I);C=G[:size,size:];Ti=np.linalg.inv(G[size:,size:]);rows=[]
    for E,local in zip(embeddings,derivatives,strict=True):
        for D in local:
            H=E.T@D@E;dC=H[:size,size:]
            value=H[:size,:size]-dC@Ti@C.T-C@Ti@dC.T+C@Ti@H[size:,size:]@Ti@C.T
            rows.append((value+value.T)/2)
    return I,G,np.array(rows)

def svec(M):
    i,j=np.triu_indices(len(M));v=M[i,j].copy();v[i!=j]*=np.sqrt(2);return v
def unsvec(v,n):
    i,j=np.triu_indices(n);value=v.copy();value[i!=j]/=np.sqrt(2);M=np.zeros((n,n));M[i,j]=value;M[j,i]=value;return M
def skew(v):return np.array([[0.,-v[2],v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]])

def pressure_fits(I,D,g,blocks):
    A=.5*np.array([svec(v) for v in D]);u,s,vh=np.linalg.svd(A,full_matrices=False);cut=1e-10*max(s[0],1.)
    keep=s>cut;unrestricted=(vh[keep].T/s[keep])@(u[:,keep].T@g);residual=g-A@unrestricted
    # PSD is a necessary relaxation, not a chosen physical ensemble.
    lipschitz=max(float(s[0]**2),1e-30);z=np.zeros(A.shape[1]);y=z.copy();t=1.;trace=[]
    for iteration in range(6000):
        step=y-A.T@(A@y-g)/lipschitz;candidate=unsvec(step,len(I));eigen,Q=np.linalg.eigh(candidate)
        nxt=svec((Q*np.maximum(eigen,0.))@Q.T);tn=(1+np.sqrt(1+4*t*t))/2
        y=nxt+(t-1)/tn*(nxt-z);delta=np.linalg.norm(nxt-z);z=nxt;t=tn
        if iteration%200==0:trace.append({'iteration':iteration,'relative_force_residual':float(np.linalg.norm(A@z-g)/np.linalg.norm(g))})
        if iteration>100 and delta<1e-12*max(np.linalg.norm(z),1.):break
    covariance=unsvec(z,len(I));ev,V=np.linalg.eigh(covariance);order=np.argsort(ev)[::-1];seeds=[]
    for k in order[:min(4,len(I))]:
        if ev[k]>0:seeds.append(np.sqrt(ev[k])*V[:,k])
    seeds.append(np.sqrt(max(np.linalg.norm(covariance),1e-8))*np.sin(np.arange(len(I))+.31)/np.sqrt(len(I)))
    norm=max(np.linalg.norm(g),1e-30)
    def equations(w,derivative=False):
        force=.5*np.einsum('i,aij,j->a',w,D,w)-g
        momentum=I@w;torque=[];J=[]
        for i in range(blocks):
            sl=slice(3*i,3*i+3);torque.extend(np.cross(w[sl],momentum[sl]))
            jac=skew(w[sl])@I[sl,:];jac[:,sl]-=skew(momentum[sl]);J.append(jac)
        value=np.r_[force,torque]/norm
        jacobian=np.vstack([np.einsum('aij,j->ai',D,w),np.vstack(J)])/norm
        return jacobian if derivative else value
    candidates=[]
    for initial in seeds:
        sol=least_squares(equations,initial,jac=lambda w:equations(w,True),max_nfev=1000,gtol=1e-12,ftol=1e-12,xtol=1e-12)
        w=sol.x;force=g-.5*np.einsum('i,aij,j->a',w,D,w);mu=I@w
        torque=np.concatenate([np.cross(w[3*i:3*i+3],mu[3*i:3*i+3]) for i in range(blocks)])
        candidates.append({'velocity_native':w.tolist(),'momentum_native':mu.tolist(),'relative_state_force_residual':float(np.linalg.norm(force)/norm),
            'maximum_state_force_residual_native':float(max(abs(force))),'relative_Euler_torque_residual':float(np.linalg.norm(torque)/norm),
            'kinetic_energy_native':float(.5*w@I@w),'residual_vector_native':force.tolist(),'optimizer_evaluations':sol.nfev})
    candidates.sort(key=lambda c:c['relative_state_force_residual']**2+c['relative_Euler_torque_residual']**2)
    report={'rotation_coordinates':len(I),'force_coordinates':len(g),'linear_symmetric_pressure_rank':int(sum(keep)),
        'unrestricted_symmetric_pressure_relative_residual':float(np.linalg.norm(residual)/norm),
        'left_span_witness':(residual/max(np.linalg.norm(residual),1e-30)).tolist(),
        'left_span_witness_matrix_projection_norm':float(np.linalg.norm(A.T@residual)/max(np.linalg.norm(residual),1e-30)),
        'PSD_relaxation_relative_residual':float(np.linalg.norm(A@z-g)/norm),'PSD_relaxation_eigenvalues':ev.tolist(),
        'PSD_relaxation_is_not_a_physical_ensemble_or_orbit':True,'PSD_iterations':iteration+1,'PSD_history':trace,
        'coherent_velocity_with_Euler_tests':candidates,'minimum_rotation_mass_eigenvalue':float(np.linalg.eigvalsh(I).min()),
        'physical_state_or_causal_clock_not_selected':True,'physical_region_excluded':False}
    return report,covariance

def controls(kernel,state):
    legacy=LegacyMetric(ROOT/'sources/metrique_courbe_R323.c');reports=[]
    for i in [0,1]:
        n=state.fields[i];h=float(state.m.d[('S' if i==0 else 'F')+'_h']);v=np.array([.17,-.11,.13,.08,-.03,.04])
        q,g=kernel(n,h,v);t=np.zeros_like(n);t[:,1:]=-np.cross(v[:3],n[:,1:]);old=legacy(n,t,h,np.zeros(3));vector=np.r_[1.,v[3:]]
        defect=abs(q-vector@old@vector)/max(abs(q),1e-30);assert defect<3e-11,defect
        probes=np.zeros((3,*tuple(state.m.d['grid_shape']),4));grid=n.reshape((*tuple(state.m.d['grid_shape']),4))
        for axis in range(3):probes[axis,...,1:]=-np.cross(np.eye(3)[axis],grid[...,1:])
        matrix,_=kinetic_blocks(grid,h,probes,np.eye(3));defect2=abs(q-v@matrix@v)/max(abs(q),1e-30);assert defect2<3e-11,defect2
        direction=np.zeros(state.system.nstate)
        if i==0:direction[:52]=np.sin(np.arange(52)*.31+.2)
        else:direction[52:54]=[.6,-.8]
        direction/=np.linalg.norm(direction);x=state.x.copy();expected=float(state.pull(i,g)@(direction[:52] if i==0 else direction[52:54]));fd=[]
        for step in [2e-5,1e-5]:
            plus=x.copy();minus=x.copy();plus[:state.system.nstate]+=step*direction;minus[:state.system.nstate]-=step*direction
            qp,_=kernel(state.field_at(plus,i),h,v);qm,_=kernel(state.field_at(minus,i),h,v);observed=(qp-qm)/(2*step)
            error=abs(observed-expected);assert error<3e-5*max(1.,abs(expected)),(i,step,error)
            fd.append({'step':step,'analytic':expected,'finite_difference':observed,'absolute_defect':error})
        Q=Rotation.from_rotvec([.24,-.17,.31]).as_matrix();rot=n.copy();rot[:,1:]=n[:,1:]@Q.T
        qr,_=kernel(rot,h,np.r_[Q@v[:3],v[3:]]);cov=abs(qr-q)/max(abs(q),1e-30);assert cov<3e-11,cov
        reports.append({'atom_index':i,'original_C_metric_relative_defect':defect,'independent_R235_numpy_relative_defect':defect2,'native_target_rotation_covariance_defect':cov,'analytic_field_gradient_checks':fd})
    state.system.evaluate(state.x)
    return reports

def run(label,pilot=False):
    started=time.time();state=State(label);kernel=Quadratic();checks=controls(kernel,state)
    print(label,'independent scalar and gradient controls passed',flush=True)
    if pilot:
        result={'label':label,'checks':checks,'pilot_only':True};dump(ROOT/'results'/f'{label}_PILOTE_CINETIQUE_R337.json',result);return result
    raw=[];derivatives=[]
    for i in range(len(state.fields)):
        G,D=primitive(kernel,state,i);raw.append(G);derivatives.append(D)
        np.savez_compressed(ROOT/'results'/f'{label}_ATOME_{i}_METRIQUE_R337.npz',primitive_metric=G,state_derivatives=D)
        print(label,'atomic metric derivatives',i+1,'/',len(state.fields),'elapsed',round(time.time()-started,1),flush=True)
    I,G,D=reduce_derivatives(raw,derivatives);B=np.tile(np.eye(3),(len(state.fields),1));Ic=B.T@I@B;Dc=np.einsum('pi,apq,qj->aij',B,D,B)
    g=state.gradient[:state.system.nstate];common,Cov=pressure_fits(Ic,Dc,g,1);independent,Covi=pressure_fits(I,D,g,len(state.fields))
    result={'revision':337,'label':label,'scope':'instantaneous necessary balance for fixed-shape internal rotations with co-transported references and boundary; R326/R235 one-field normalisation',
        'source_recipe':state.recipe,'native_potential_energy':state.energy,'maximum_potential_state_gradient_native':float(max(abs(g))),
        'center_potential_gradient_inf':float(max(abs(state.gradient[state.system.nstate:]))),'independent_native_controls':checks,
        'common_SO3_rotation':common,'independent_atomic_SO3_rotations_conditional_extension':independent,
        'same_action_no_new_material_gain':True,'translation_momentum_reduced_after_assembly':True,
        'source_connection_alone_does_not_change_relative_coordinates_R250':True,'R250_odd_two_face_numeric_matrices_not_reused':True,
        'no_new_physical_charge_or_source_identification':True,'no_periodic_trajectory_or_stability_proven':True,
        'no_new_angular_target_or_observation_tolerance':True,'physical_alignment_status':'UNKNOWN','physical_region_excluded':False,
        'numerical_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'native_metric_source_sha256':kernel.hash,'seconds':time.time()-started}
    np.savez_compressed(ROOT/'results'/f'{label}_OPERATEURS_MAINTIEN_R337.npz',full_atomic_rotation_translation_metric=G,
        reduced_atomic_rotation_metric=I,reduced_metric_state_derivatives=D,potential_state_gradient=g,common_rotation_metric=Ic,
        common_rotation_metric_derivatives=Dc,PSD_relaxed_common_velocity_tensor=Cov,PSD_relaxed_individual_velocity_tensor=Covi)
    dump(ROOT/'results'/f'{label}_MAINTIEN_CINETIQUE_R337.json',result)
    print(json.dumps({'label':label,'common_residual':common['coherent_velocity_with_Euler_tests'][0]['relative_state_force_residual'],
        'independent_residual':independent['coherent_velocity_with_Euler_tests'][0]['relative_state_force_residual'],
        'independent_Euler_defect':independent['coherent_velocity_with_Euler_tests'][0]['relative_Euler_torque_residual'],'seconds':result['seconds']}),flush=True)
    return result
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--labels',nargs='+',default=['SF4','SF6']);parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    for label in args.labels:run(label,args.pilot)
