'R339: native spatial/orbital, target and translation metric on R338 fields.\n\nNecessary instantaneous tests, not an orbit or physical validation. The old\nR261/R262 kinematic embedding is used with newly integrated one-field R235\nmatrices. All derivatives include the orbital centres and the P=0 Schur term.'
import os
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,ctypes,hashlib,json,shutil,subprocess,sys,time,itertools
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from kinetic_preparations import ROOT,State,Quadratic,svec,unsvec,skew
from contacts_reactive import ReactiveContacts
from angles_internal import pullback
from mix import dump

class SpatialQuadratic:
    def __init__(self):
        source=ROOT/'sources/metrique_rotations_R339.c';self.hash=hashlib.sha256(source.read_bytes()).hexdigest()
        target=source.with_name('rotations_R339_'+self.hash[:12]+('.dylib' if sys.platform=='darwin' else '.so'))
        if not target.exists():
            cc=shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')
            if cc is None: raise RuntimeError('Compiler C required')
            subprocess.run([cc,'-O3',*(['-dynamiclib'] if sys.platform=='darwin' else ['-shared','-fPIC']),str(source),'-lm','-o',str(target)],check=True)
        a=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');self.f=ctypes.CDLL(str(target)).bst_rotations
        self.f.argtypes=[ctypes.c_int,ctypes.c_double,a,a,a,a,a,ctypes.POINTER(ctypes.c_double)];self.f.restype=ctypes.c_int
    def __call__(self,n,h,v,center=None):
        n=np.ascontiguousarray(n);g=np.zeros_like(n);gc=np.zeros(3);q=ctypes.c_double()
        c=np.zeros(3) if center is None else np.ascontiguousarray(center,dtype=float)
        if self.f(round(len(n)**(1/3)),float(h),n,c,np.ascontiguousarray(v,dtype=float),g,gc,ctypes.byref(q)):
            raise ValueError('Inoperable gnomonic map; UNKNOWN, no physical region excluded')
        return q.value,g,gc

def numpy_quadratic(n,h,v,c):
    'Independent eight-corner quadrature with actual corner coordinates.'
    N=round(len(n)**(1/3));f=n.reshape(N,N,N,4)
    xyz=np.moveaxis(np.indices((N,N,N)),0,-1)*h-.5*(N-1)*h+c;total=0.
    for bits in itertools.product([0,1],repeat=3):
        axes=tuple(k for k,b in enumerate(bits) if b);z=np.flip(f,axis=axes) if axes else f
        p=np.flip(xyz,axis=axes) if axes else xyz;base=z[:-1,:-1,:-1];ds=[]
        for a,nb in enumerate([z[1:,:-1,:-1],z[:-1,1:,:-1],z[:-1,:-1,1:]]):
            scalar=np.sum(base*nb,axis=-1);assert scalar.min()>0
            ds.append((-1)**bits[a]*(nb/scalar[...,None]-base)/h)
        d=np.stack(ds,axis=-2);ad=v[6:]+np.cross(v[3:6],p[:-1,:-1,:-1])
        u=np.zeros_like(base);u[...,1:]=-np.cross(v[:3],base[...,1:]);u-=np.einsum('...a,...ac->...c',ad,d)
        tr=np.sum(d*d,axis=(-1,-2));norm=np.sum(u*u,axis=-1);pd=np.einsum('...c,...ac->...a',u,d)
        total+=h**3/8*np.sum((1+tr)*norm-np.sum(pd*pd,axis=-1))
    return float(total)

def controls(kernel,state):
    legacy=Quadratic();rows=[];v=np.array([.17,-.11,.13,.023,-.031,.017,.08,-.03,.04]);c=np.array([.4,-.3,.7])
    for i in [0,1]:
        n=state.fields[i];h=float(state.m.d[('S' if i==0 else 'F')+'_h']);q,g,gc=kernel(n,h,v,c)
        qnp=numpy_quadratic(n,h,v,c);defect=abs(q-qnp)/max(1.,abs(q));assert defect<3e-11,defect
        v0=v.copy();v0[3:6]=0;q0,g0,_=kernel(n,h,v0,c);qo,go=legacy(n,h,np.r_[v[:3],v[6:]])
        d0=abs(q0-qo)/max(1.,abs(q0));dg=float(np.max(abs(g0-go)));assert d0<3e-11 and dg<2e-10,(d0,dg)
        direction=np.sin(np.arange(52 if i==0 else 2)*.31+.2);direction/=np.linalg.norm(direction)
        full=np.zeros(len(state.x));sl=slice(0,52) if i==0 else slice(52,54);full[sl]=direction;exact=float(state.pull(i,g)@direction);fd=[]
        for step in [2e-5,1e-5]:
            qp=kernel(state.field_at(state.x+step*full,i),h,v,c)[0];qm=kernel(state.field_at(state.x-step*full,i),h,v,c)[0]
            observed=(qp-qm)/(2*step);error=abs(exact-observed);assert error<3e-5*max(1.,abs(exact)),(i,step,error)
            fd.append({'step':step,'analytic':exact,'finite_difference':observed,'absolute_defect':error})
        d=np.array([.3,-.4,.2]);cmb=[]
        for step in [2e-5,1e-5]:
            observed=(kernel(n,h,v,c+step*d)[0]-kernel(n,h,v,c-step*d)[0])/(2*step);error=abs(observed-gc@d)
            assert error<2e-5*max(1.,abs(gc@d)),error
            cmb.append({'step':step,'analytic':float(gc@d),'finite_difference':observed,'absolute_defect':error})
        shift=np.array([.19,-.14,.11]);vs=v.copy();vs[6:]+=np.cross(v[3:6],shift)
        qs,gs,gcs=kernel(n,h,vs,c-shift);origin=abs(qs-q)/max(1.,abs(q));assert origin<3e-11,origin
        R=Rotation.from_rotvec([.24,-.17,.31]).as_matrix();rn=n.copy();rn[:,1:]=n[:,1:]@R.T;rv=v.copy();rv[:3]=R@v[:3]
        qr,_,_=kernel(rn,h,rv,c);target=abs(qr-q)/max(1.,abs(q));assert target<3e-11,target
        rows.append({'atom_index':i,'independent_numpy_scalar_relative_defect':defect,'R337_zero_spatial_limit_relative_defect':d0,
            'R337_zero_spatial_gradient_inf_defect':dg,'field_gradient_checks':fd,'center_gradient_checks':cmb,
            'origin_shift_relative_defect':origin,'target_SO3_covariance_relative_defect':target})
    state.system.evaluate(state.x);return rows

def primitive(kernel,state,i):
    n=state.fields[i];h=float(state.m.d[('S' if i==0 else 'F')+'_h']);p=52 if i==0 else 2;G=np.zeros((9,9));D=np.zeros((p,9,9));eye=np.eye(9)
    for j in range(9):
        q,g,_=kernel(n,h,eye[j]);G[j,j]=q;D[:,j,j]=state.pull(i,g)
    for j in range(9):
        for k in range(j+1,9):
            q,g,_=kernel(n,h,eye[j]+eye[k]);G[j,k]=G[k,j]=.5*(q-G[j,j]-G[k,k]);D[:,j,k]=D[:,k,j]=.5*(state.pull(i,g)-D[:,j,j]-D[:,k,k])
    return G,D

def centers_and_jacobian(state):
    m=state.m;c=state.x[state.system.nstate:].reshape(m.N,3);r=m.rbase*np.exp(c[:,0]);w=np.einsum('ia,iac->ic',c[:,1:],m.tangents)
    geometry=m.state(c.ravel());X=np.vstack([np.zeros(3),geometry['R']]);J=np.zeros((m.N,3,3))
    for j in range(m.N):
        J[j,:,0]=X[j+1]
        for a in range(3):
            g=np.zeros_like(X[1:]);g[j,a]=r[j];back=pullback(m.D0,w,g)
            J[j,a,1:]=m.tangents[j]@back[j]
    return X,J

def embeddings(X,frames):
    count=len(X);size=3*count+3;out=[]
    for i,(x,O) in enumerate(zip(X,frames,strict=True)):
        E=np.zeros((9,size+3));E[:3,3*i:3*i+3]=np.eye(3)
        E[3:6,3*count:size]=np.linalg.det(O)*O.T
        E[6:,3*count:size]=-O.T@skew(x);E[6:,size:]=O.T;out.append(E)
    return out

def schur(G,H=None):
    size=len(G)-3;C=G[:size,size:];Mi=np.linalg.inv(G[size:,size:]);I=G[:size,:size]-C@Mi@C.T;I=.5*(I+I.T)
    if H is None:return I
    dC=H[:,:size,size:];D=H[:,:size,:size]-dC@Mi@C.T-C@Mi@np.swapaxes(dC,1,2)+C@Mi@H[:,size:,size:]@Mi@C.T
    return I,.5*(D+np.swapaxes(D,1,2))

def assemble(state,raw,derivatives,frames):
    X,J=centers_and_jacobian(state);E=embeddings(X,frames);N=len(X);size=3*N+6;G=np.zeros((size,size));H=np.zeros((len(state.x),size,size))
    for i,(Gi,Di,Ei) in enumerate(zip(raw,derivatives,E,strict=True)):
        G+=Ei.T@Gi@Ei;sl=slice(0,52) if i==0 else slice(50+2*i,52+2*i)
        H[sl]+=Ei.T@Di@Ei
        if i:
            for a in range(3):
                of=np.zeros_like(Ei);of[6:,3*N:3*N+3]=-frames[i].T@skew(J[i-1,:,a])
                H[state.system.nstate+3*(i-1)+a]+=of.T@Gi@Ei+Ei.T@Gi@of
    I,D=schur(G,H);B=np.zeros((len(I),6));B[:3*N,:3]=np.tile(np.eye(3),(N,1));B[3*N:,3:]=np.eye(3)
    return I,D,G,B,E

def geometry_controls(state,raw,derivatives,frames,I,D,G):
    X,J=centers_and_jacobian(state);N=len(X);shift=np.array([.22,-.18,.13]);Es=embeddings(X-shift,frames)
    Gs=sum(E.T@M@E for E,M in zip(Es,raw,strict=True));Is=schur(Gs);origin=float(np.linalg.norm(I-Is)/np.linalg.norm(I));assert origin<3e-11,origin
    v=np.cos(np.arange(len(I))*.23+.31)*.013;u=-np.linalg.solve(G[-3:,-3:],G[-3:,:-3]@v);y=np.r_[v,u]
    momentum=float(np.linalg.norm(G[-3:]@y));assert momentum<1e-9,momentum
    # Proper and improper passive changes of the molecular basis. Axial Ω
    # needs det(A); target rotations remain in their independent target basis.
    covariance=[]
    for name,A in [('proper',Rotation.from_rotvec([.31,-.19,.24]).as_matrix()),('improper',np.diag([-1.,1.,1.]))]:
        Ea=embeddings(X@A.T,np.array([A@O for O in frames]));Ga=sum(E.T@M@E for E,M in zip(Ea,raw,strict=True))
        L=np.eye(len(G));L[3*N:3*N+3,3*N:3*N+3]=np.linalg.det(A)*A;L[-3:,-3:]=A
        defect=float(np.linalg.norm(L.T@Ga@L-G)/np.linalg.norm(G));assert defect<3e-11,defect
        wrong=[]
        for e,O in zip(Ea,[A@O for O in frames],strict=True):
            z=e.copy();z[3:6,3*N:3*N+3]=O.T;wrong.append(z)
        Gw=sum(E.T@M@E for E,M in zip(wrong,raw,strict=True));negative=float(np.linalg.norm(L.T@Gw@L-G)/np.linalg.norm(G))
        if name=='improper':assert negative>1e-4,negative
        covariance.append({'basis':name,'full_Gram_relative_defect':defect,'omitted_axial_determinant_negative_control_relative_defect':negative})
    # Centre derivatives of the assembled P=0 kinetic energy, not a static
    # radial-constraint identity. Compare the exact centre embedding twice.
    direction=np.zeros(len(state.x));direction[state.system.nstate:]=np.sin(np.arange(3*state.m.N)*.41+.3);direction/=np.linalg.norm(direction)
    expected=float(np.einsum('i,aij,j,a',v,D,v,direction));checks=[];x=state.x.copy()
    for step in [2e-5,1e-5]:
        mats=[]
        for sign in [1,-1]:
            state.x=x+sign*step*direction;xx,_=centers_and_jacobian(state);es=embeddings(xx,frames);gg=sum(e.T@m@e for e,m in zip(es,raw,strict=True));mats.append(schur(gg))
        observed=float(v@(mats[0]-mats[1])@v/(2*step));error=abs(expected-observed);assert error<3e-6*max(1.,abs(expected)),error
        checks.append({'step':step,'analytic':expected,'finite_difference':observed,'absolute_defect':error})
    state.x=x
    return {'origin_P0_Schur_relative_defect':origin,'eliminated_translation_momentum_norm':momentum,
        'passive_full_Gram_covariance_checks':covariance,'Schur_orbital_centre_derivative_checks':checks}

def potential(state,law,reader):
    model=ReactiveContacts(state.label,law=law,F_reader=reader);energy,total=model.evaluate(state.x);base=np.zeros_like(total);base[:model.nstate]=model.reduction['own_state_gradient']
    m=model.m;c=state.x[model.nstate:].reshape(m.N,3);r=model.last['radii'];D=model.last['D'];w=np.einsum('ia,iac->ic',c[:,1:],m.tangents)
    basec=np.zeros_like(c);basec[:,0]=r*np.array([t['pair_radial_derivative'] for t in model.last['contacts']]);back=pullback(m.D0,w,model.reduction['Gram_gradient_direction'])
    basec[:,1:]=np.einsum('iac,ic->ia',m.tangents,back);base[model.nstate:]=basec.ravel()
    return base,total-base,model.result(state.x)

def pressure_fit(I,D,base,contact,nstate,blocks,maxeval=700):
    'Linear span, PSD+gamma relaxation, then actual single-velocity Euler test.'
    eig,Q=np.linalg.eigh(I);assert eig[0]>1e-9,(eig[0],eig[-1]);L=(Q/np.sqrt(eig))@Q.T
    d=np.einsum('ij,ajk,kl->ail',L.T,D,L);A=.5*np.array([svec(v) for v in d]);norm=max(float(np.linalg.norm(base)),1e-30)
    sx=norm/max(float(np.linalg.norm(A,2)),1e-30);sg=norm/max(float(np.linalg.norm(contact)),1e-30)
    F=np.column_stack([sx*A,-sg*contact])/norm;g=base/norm;U,s,V=np.linalg.svd(F,full_matrices=False);keep=s>1e-10*max(s[0],1.)
    z0=(V[keep].T/s[keep])@(U[:,keep].T@g);residual=g-F@z0;lip=max(float(s[0]**2),1e-30)
    z=np.zeros(F.shape[1]);y=z.copy();t=1.;history=[]
    for iteration in range(7000):
        step=y-F.T@(F@y-g)/lip;ev,P=np.linalg.eigh(unsvec(step[:-1],len(I)));nxt=np.r_[svec((P*np.maximum(ev,0.))@P.T),max(step[-1],0.)]
        tn=(1+np.sqrt(1+4*t*t))/2;y=nxt+(t-1)/tn*(nxt-z);delta=np.linalg.norm(nxt-z);z=nxt;t=tn
        if iteration%500==0:history.append({'iteration':iteration,'normalized_native_force_residual':float(np.linalg.norm(F@z-g))})
        if iteration>200 and delta<1e-11*max(1.,np.linalg.norm(z)):break
    tensor=unsvec(z[:-1],len(I));ev,P=np.linalg.eigh(tensor);seeds=[]
    for k in np.argsort(ev)[::-1][:3]:
        if ev[k]>0:seeds.append(np.r_[np.sqrt(ev[k])*P[:,k],z[-1]])
    seeds.append(np.r_[.05*np.sin(np.arange(len(I))+.31),max(z[-1],.01)])
    W=np.sqrt(sx)*L
    def equations(a,jac=False):
        w=W@a[:-1];gamma=sg*a[-1];mu=I@w;force=.5*np.einsum('i,aij,j->a',w,D,w)-base-gamma*contact
        torque=[];J=[]
        for i in range(blocks):
            sl=slice(3*i,3*i+3);torque.extend(np.cross(w[sl],mu[sl]));j=skew(w[sl])@I[sl];j[:,sl]-=skew(mu[sl]);J.append(j)
        if not jac:return np.r_[force,torque]/norm
        jf=np.column_stack([np.einsum('aij,j->ai',D,w)@W,-sg*contact]);jt=np.column_stack([np.vstack(J)@W,np.zeros(3*blocks)])
        return np.vstack([jf,jt])/norm
    candidates=[]
    for seed in seeds:
        sol=least_squares(equations,seed,jac=lambda a:equations(a,True),bounds=(np.r_[np.full(len(I),-np.inf),0.],np.full(len(I)+1,np.inf)),
            max_nfev=maxeval,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        w=W@sol.x[:-1];gamma=sg*sol.x[-1];mu=I@w;res=base+gamma*contact-.5*np.einsum('i,aij,j->a',w,D,w)
        torque=np.concatenate([np.cross(w[3*i:3*i+3],mu[3*i:3*i+3]) for i in range(blocks)])
        candidates.append({'velocity_native':w.tolist(),'momentum_native':mu.tolist(),'common_contact_stiffness_factor':float(gamma),
            'state_force_inf_native':float(max(abs(res[:nstate]))),'center_force_inf_native':float(max(abs(res[nstate:]))),
            'Euler_torque_inf_native':float(max(abs(torque))),'force_norm_divided_by_base_norm':float(np.linalg.norm(res)/norm),
            'Euler_norm_divided_by_base_norm':float(np.linalg.norm(torque)/norm),'kinetic_energy_native':float(.5*w@mu),
            'native_force_residual':res.tolist(),'native_Euler_torque_residual':torque.tolist(),'optimizer_evaluations':sol.nfev,'optimizer_status':int(sol.status)})
    candidates.sort(key=lambda c:c['force_norm_divided_by_base_norm']**2+c['Euler_norm_divided_by_base_norm']**2)
    witness=residual/max(float(np.linalg.norm(residual)),1e-30)
    out={'rotation_coordinates':len(I),'force_coordinates':len(base),'minimum_reduced_metric_eigenvalue':float(eig[0]),
        'maximum_reduced_metric_eigenvalue':float(eig[-1]),'unrestricted_symmetric_pressure_and_gain_rank':int(sum(keep)),
        'unrestricted_linear_normalized_residual':float(np.linalg.norm(residual)),'linear_left_witness':witness.tolist(),
        'left_witness_map_projection_norm':float(np.linalg.norm(F.T@witness)),
        'PSD_nonnegative_gain_relaxation_normalized_residual':float(np.linalg.norm(F@z-g)),
        'PSD_relaxed_gain':float(sg*z[-1]),'PSD_iterations':iteration+1,'PSD_history':history,
        'PSD_is_necessary_relaxation_not_ensemble_or_orbit':True,'coherent_velocity_Euler_candidates':candidates,
        'gain_zero_is_only_boundary_of_positive_law_family':True,'no_physical_interval_or_clock_selected':True,
        'native_normalized_residual_is_not_experimental_error_rate':True,'physical_status':'UNKNOWN','physical_region_excluded':False}
    return out,sx*tensor

def run(branch,controls_only=False):
    started=time.time();name='PORTS_COMMON' if branch=='conservee' else 'JETS1';state=State('SF4')
    source=ROOT/'results'/f'SF4_AFFINEMENT_{name}_R338.json';point=json.loads(source.read_text())['result'];state.refresh(np.array(point['full_coordinates']))
    kernel=SpatialQuadratic();checks=controls(kernel,state);print(branch,'scalar, field and centre controls passed',round(time.time()-started,1),flush=True)
    if controls_only:
        dump(ROOT/'results'/f'SF4_{branch}_CONTROLES_R339.json',{'controls':checks,'kernel_sha256':kernel.hash});return
    raw=[];derivatives=[]
    for i in range(len(state.fields)):
        cache=ROOT/'results'/f'SF4_{branch}_ATOME_{i}_ROTATIONS_R339.npz'
        if cache.exists():
            saved=np.load(cache);assert str(saved['kernel_sha256'])==kernel.hash and str(saved['point_sha256'])==hashlib.sha256(source.read_bytes()).hexdigest();G,D=saved['primitive_metric'],saved['field_derivatives']
        else:
            G,D=primitive(kernel,state,i);np.savez_compressed(cache,primitive_metric=G,field_derivatives=D,kernel_sha256=kernel.hash,point_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        raw.append(G);derivatives.append(D);print(branch,'atomic spatial metric',i+1,'/',len(state.fields),'elapsed',round(time.time()-started,1),flush=True)
    rows=[]
    for protocol,law,reader in [('grids_alignees_stiffness_fixed','fixed','held'),('frames_coorientes_stiffness_reactive','current','mobile')]:
        model=ReactiveContacts('SF4',law=law,F_reader=reader);frames=np.tile(np.eye(3),(len(state.fields),1,1)) if reader=='held' else np.concatenate([np.eye(3)[None],model.frames])
        I,D,G,B,E=assemble(state,raw,derivatives,frames);geo=geometry_controls(state,raw,derivatives,frames,I,D,G)
        base,contact,potential_record=potential(state,law,reader);Ic=B.T@I@B;Dc=np.einsum('pi,apq,qj->aij',B,D,B)
        common,C=pressure_fit(Ic,Dc,base,contact,state.system.nstate,2);print(branch,protocol,'common',common['coherent_velocity_Euler_candidates'][0]['state_force_inf_native'],flush=True)
        independent,Ci=pressure_fit(I,D,base,contact,state.system.nstate,len(state.fields)+1)
        np.savez_compressed(ROOT/'results'/f'SF4_{branch}_{protocol}_OPERATEURS_R339.npz',metric_with_translation=G,reduced_metric=I,
            reduced_metric_derivatives=D,common_embedding=B,common_metric=Ic,common_metric_derivatives=Dc,potential_base_gradient=base,
            potential_contact_gradient=contact,frames=frames,common_PSD_relaxation=C,independent_PSD_relaxation=Ci,full_coordinates=state.x)
        rows.append({'protocol':protocol,'potential':potential_record,'kinematic_controls':geo,'common_target_and_spatial_rotation':common,
            'independent_target_and_common_spatial_rotations_conditional':independent})
        print(branch,protocol,'independent',independent['coherent_velocity_Euler_candidates'][0]['state_force_inf_native'],'elapsed',round(time.time()-started,1),flush=True)
    out={'revision':339,'branch':branch,'scope':'instantaneous necessary equilibrium at prescribed R338 SF4 fields/centres with global spatial/orbital rotation, target rotations and P=0 translation reduction',
        'source_calibration_file':source.name,'source_calibration_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'independent_native_controls':checks,'protocols':rows,
        'source_python_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_C_sha256':kernel.hash,
        'R261_R262_kinematic_embedding_recomputed_with_current_one_field_action':True,'old_numeric_matrices_or_two_face_factors_not_reused':True,
        'reactive_and_fixed_potential_variants_kept_separate':True,'individual_F_frames_prepared_not_selected_or_freely_equilibrated':True,
        'no_full_relative_frame_or_shape_trajectory_solved':True,'no_new_tolerance_or_experimental_target':True,
        'no_physical_region_excluded':True,'no_domain_or_contract_closed':True,'physical_alignment_status':'UNKNOWN','seconds':time.time()-started}
    dump(ROOT/'results'/f'SF4_{branch}_BILAN_ROTATIONS_R339.json',out);return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--branches',nargs='+',choices=['conservee','souple'],default=['conservee','souple']);p.add_argument('--controls-only',action='store_true');args=p.parse_args()
    for branch in args.branches:run(branch,args.controls_only)
