'Retune received coefficients after field changes, transporting their effects.\n\nalpha=1, beta>0 in E2+beta E4 and the SAME kinetic completion. Native current\ncontact curvature is recomputed; gamma and angular weight k are jointly fit\nwith prepared rotations. No new molecular force or observation tolerance.'
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,ctypes,hashlib,json,shutil,subprocess,sys,time
import numpy as np
from scipy.optimize import least_squares
from kinetic_preparations import ROOT,State,svec,unsvec,skew
from rotations_spatial import SpatialQuadratic,primitive,assemble,numpy_quadratic
from refit_rotations import FastGram,Routh
from contacts_reactive import ReactiveContacts,RecordingAction
from angles_internal import pullback
from mix import dump

def load_c(source,function,arguments,dependencies=()):
    digest=hashlib.sha256(source.read_bytes()+b''.join(p.read_bytes() for p in dependencies)).hexdigest()
    target=source.with_name(source.stem+'_'+digest[:12]+('.dylib' if sys.platform=='darwin' else '.so'))
    if not target.exists():
        cc=shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')
        if not cc:raise RuntimeError('Compiler C required')
        subprocess.run([cc,'-O3',*(['-dynamiclib'] if sys.platform=='darwin' else ['-shared','-fPIC']),str(source),'-lm','-o',str(target)],check=True)
    f=getattr(ctypes.CDLL(str(target)),function);f.argtypes=arguments;f.restype=ctypes.c_int
    return f,digest

ARRAY=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
class Action:
    def __init__(self,beta=1.,alpha=1.):
        self.alpha=float(alpha);self.beta=float(beta)
        self.f,self.hash=load_c(ROOT/'sources/action_parametree_R331.c','bst_action_params',[ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ARRAY,ARRAY,ARRAY])
    def __call__(self,n,h):
        g=np.zeros_like(n);out=np.zeros(3)
        if self.f(round(len(n)**(1/3)),h,self.alpha,self.beta,np.ascontiguousarray(n),g,out):raise ValueError('UNKNOWN energy map')
        return float(out[0]),g

class Kinetic:
    def __init__(self,beta=1.,alpha=1.):
        self.alpha=float(alpha);self.beta=float(beta)
        self.f,self.hash=load_c(ROOT/'sources/metrique_parametree_R340.c','bst_rotations_params',
            [ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ARRAY,ARRAY,ARRAY,ARRAY,ARRAY,ctypes.POINTER(ctypes.c_double)])
    def __call__(self,n,h,v,center=None):
        g=np.zeros_like(n);gc=np.zeros(3);q=ctypes.c_double();c=np.zeros(3) if center is None else np.ascontiguousarray(center,dtype=float)
        if self.f(round(len(n)**(1/3)),h,self.alpha,self.beta,np.ascontiguousarray(n),c,np.ascontiguousarray(v,dtype=float),g,gc,ctypes.byref(q)):raise ValueError('UNKNOWN kinetic card')
        return q.value,g,gc

class Gram:
    def __init__(self,beta=1.,alpha=1.):
        self.alpha=float(alpha);self.beta=float(beta)
        self.f,self.hash=load_c(ROOT/'sources/metrique_rapide_parametree_R340.c','bst_rotations_gram_params',
            [ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ARRAY,ARRAY],(ROOT/'sources/metrique_parametree_R340.c',))
    def __call__(self,n,h):
        G=np.zeros((9,9))
        if self.f(round(len(n)**(1/3)),h,self.alpha,self.beta,np.ascontiguousarray(n),G):raise ValueError('UNKNOWN kinetic card')
        return G

def install(model,beta,k=0.,gamma=1.):
    model.action.action=Action(beta);model.m.weight=float(k);model.factor=float(gamma)
    return model

def load_state(name,label='SF4'):
    s=State(label)
    if label=='SF6':return s
    if name=='R339_libere':point=json.loads((ROOT/'resultats/SF4_souple_grilles_alignees_raideur_fixe_ROUTH_R339.json').read_text())['result']
    else:
        tag='PORTS_COMMON' if name=='R338_conservee' else 'JETS1';point=json.loads((ROOT/'results'/f'SF4_AFFINEMENT_{tag}_R338.json').read_text())['result']
    s.refresh(np.array(point['full_coordinates']));return s

def force_components(model,x):
    model.m.weight=0.;model.factor=1.;E,total=model.evaluate(x);s=model.state;m=model.m;ns=model.nstate
    base=np.zeros_like(total);base[:ns]=model.reduction['own_state_gradient'];c=x[ns:].reshape(m.N,3);r=model.last['radii'];D=model.last['D'];w=np.einsum('ia,iac->ic',c[:,1:],m.tangents)
    gc=np.zeros_like(c);gc[:,0]=r*np.array([row['pair_radial_derivative'] for row in model.last['contacts']]);base[ns:]=gc.ravel()
    contact=total-base;diff=D@D.T-m.target;gram=np.zeros_like(total);back=pullback(m.D0,w,2*diff@D)
    gc=np.zeros_like(c);gc[:,1:]=np.einsum('iac,ic->ia',m.tangents,back);gram[ns:]=gc.ravel()
    return base,np.column_stack([contact,gram]),{'atomic_self_energies':model.last['atomic_self_energies'],'base_potential_energy':float(sum(model.last['atomic_self_energies'])+model.last['well_energy']),
        'contact_energy_at_gain1':model.last['contact_energy'],'Gram_energy_at_weight1':.5*float(np.sum(diff*diff)),
        'contact_kappas_at_gain1':[r['kappa'] for r in model.last['contacts']],'geometry':m.geometry(model.last)}

def fit(I,D,base,C,nstate,blocks=2,fixed_gains=None,starts=3):
    'Profile prepared velocities and nonnegative received coefficients.'
    eig,Q=np.linalg.eigh(I)
    if eig[0]<=1e-9:raise ValueError('Unqualified metro')
    L=(Q/np.sqrt(eig))@Q.T;d=np.einsum('ij,ajk,kl->ail',L.T,D,L);A=.5*np.array([svec(z) for z in d])
    target=base.copy()
    if fixed_gains is not None:target+=C@fixed_gains
    norm=max(float(np.linalg.norm(target)),1e-30);sx=norm/max(float(np.linalg.norm(A,2)),1e-30)
    sg=norm/np.maximum(np.linalg.norm(C,axis=0),1e-30);p=A.shape[1];F=sx*A/norm
    if fixed_gains is None:F=np.column_stack([F,-C*sg[None,:]/norm])
    U,s,V=np.linalg.svd(F,full_matrices=False);keep=s>1e-10*max(s[0],1.);g=target/norm;zls=(V[keep].T/s[keep])@(U[:,keep].T@g);linear_res=g-F@zls
    lips=max(float(s[0]**2),1e-30);z=np.zeros(F.shape[1]);y=z.copy();t=1.
    for iteration in range(4500):
        step=y-F.T@(F@y-g)/lips;ev,P=np.linalg.eigh(unsvec(step[:p],len(I)));nxt=svec((P*np.maximum(ev,0.))@P.T)
        if fixed_gains is None:nxt=np.r_[nxt,np.maximum(step[p:],0.)]
        tn=(1+np.sqrt(1+4*t*t))/2;y=nxt+(t-1)/tn*(nxt-z);delta=np.linalg.norm(nxt-z);z=nxt;t=tn
        if iteration>200 and delta<1e-11*max(1.,np.linalg.norm(z)):break
    tensor=unsvec(z[:p],len(I));ev,P=np.linalg.eigh(tensor);W=np.sqrt(sx)*L;seeds=[]
    for k in np.argsort(ev)[::-1][:starts]:
        if ev[k]>0:seeds.append(np.r_[np.sqrt(ev[k])*P[:,k],z[p:]] if fixed_gains is None else np.sqrt(ev[k])*P[:,k])
    if not seeds:seeds=[np.r_[np.ones(len(I))*.01,np.ones(2)*.01] if fixed_gains is None else np.ones(len(I))*.01]
    def equation(a,jac=False):
        w=W@a[:len(I)];gain=sg*a[len(I):] if fixed_gains is None else np.array(fixed_gains);mu=I@w
        force=.5*np.einsum('i,aij,j->a',w,D,w)-base-C@gain;torques=[];jt=[]
        for b in range(blocks):
            sl=slice(3*b,3*b+3);torques.extend(np.cross(w[sl],mu[sl]));J=skew(w[sl])@I[sl];J[:,sl]-=skew(mu[sl]);jt.append(J)
        if not jac:return np.r_[force,torques]/norm
        jf=np.einsum('aij,j->ai',D,w)@W;Jt=np.vstack(jt)@W
        if fixed_gains is None:jf=np.column_stack([jf,-C*sg[None,:]]);Jt=np.column_stack([Jt,np.zeros((3*blocks,2))])
        return np.vstack([jf,Jt])/norm
    records=[]
    for seed in seeds:
        bound=(np.r_[np.full(len(I),-np.inf),np.zeros(2)],np.full(len(I)+2,np.inf)) if fixed_gains is None else (-np.inf,np.inf)
        opt=least_squares(equation,seed,jac=lambda a:equation(a,True),bounds=bound,max_nfev=550,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        w=W@opt.x[:len(I)];gain=sg*opt.x[len(I):] if fixed_gains is None else np.array(fixed_gains);force=base+C@gain-.5*np.einsum('i,aij,j->a',w,D,w);mu=I@w
        torque=np.concatenate([np.cross(w[3*b:3*b+3],mu[3*b:3*b+3]) for b in range(blocks)])
        records.append({'velocity_native':w.tolist(),'momenta_native':mu.tolist(),'gamma':float(gain[0]),'kangle':float(gain[1]),
            'state_force_inf_native':float(max(abs(force[:nstate]))),'center_force_inf_native':float(max(abs(force[nstate:]))),'Euler_torque_inf_native':float(max(abs(torque))),
            'native_force_and_torque_squared_norm':float(force@force+torque@torque),'normalised_force_norm':float(np.linalg.norm(force)/norm),
            'native_force_residual':force.tolist(),'native_torque_residual':torque.tolist(),'kinetic_energy_native':float(.5*w@mu),
            'optimizer_evaluations':opt.nfev,'optimizer_status':int(opt.status)})
    records.sort(key=lambda r:r['native_force_and_torque_squared_norm'])
    return {'candidates':records,'normalization_base_force_norm':norm,'unrestricted_pressure_gain_rank':int(sum(keep)),
        'unrestricted_linear_normalized_residual':float(np.linalg.norm(linear_res)),'PSD_relaxation_normalized_residual':float(np.linalg.norm(F@z-g)),
        'PSD_relaxation_not_ensemble_or_orbit':True,'PSD_iterations':iteration+1,'minimum_metric_eigenvalue':float(eig[0]),
        'gains_fixed':None if fixed_gains is None else list(map(float,fixed_gains)),'physical_status':'UNKNOWN','physical_region_excluded':False}

def primitive_signature(state,kernel):
    digest=hashlib.sha256(state.x.tobytes()+kernel.hash.encode())
    for name in ['cinetique_preparations_R337.py','rotations_spatiales_R339.py']:
        digest.update((ROOT/'scripts'/name).read_bytes())
    for i,(n,t,p) in enumerate(zip(state.fields,state.tangent,state.parts,strict=True)):
        digest.update(np.asarray(float(state.m.d[('S' if i==0 else 'F')+'_h'])).tobytes())
        for a in [n,t,p]:digest.update(np.ascontiguousarray(a).tobytes())
    for a in [state.Z,state.pull_S]:digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()

def split_primitives(state,name):
    raw2=[];d2=[];raw4=[];d4=[];k0=Kinetic(0.);k1=Kinetic(1.)
    signature=primitive_signature(state,k0)
    for i in range(len(state.fields)):
        path=ROOT/'results'/f'{name}_ATOME_{i}_DECOMPOSITION_R340.npz'
        if path.exists():
            z=np.load(path);assert str(z['signature'])==signature;G2,D2,G4,D4=[z[k] for k in ['G2','D2','G4','D4']]
        else:
            G2,D2=primitive(k0,state,i)
            G,D=primitive(k1,state,i)
            G4,D4=G-G2,D-D2;np.savez_compressed(path,G2=G2,D2=D2,G4=G4,D4=D4,signature=signature)
        raw2.append(G2);d2.append(D2);raw4.append(G4);d4.append(D4)
        print(name,'split kinetic atom',i+1,'/',len(state.fields),flush=True)
    return raw2,d2,raw4,d4

def weighted_checks():
    state=load_state('R338_souple');original=state.m.trial.action;legacy=SpatialQuadratic();oldgram=FastGram();rows=[]
    for i in [0,1]:
        n=state.fields[i];h=float(state.m.d[('S' if i==0 else 'F')+'_h']);v=np.array([.17,-.11,.13,.023,-.031,.017,.08,-.03,.04]);c=np.array([.4,-.3,.7])
        eo,go=original(n,h);in_field,gn=Action(1.)(n,h);assert abs(in_field-eo)<1e-9 and np.max(abs(gn-go))<1e-9
        qo,gqo,gco=legacy(n,h,v,c);q,gq,gc=Kinetic(1.)(n,h,v,c);assert abs(q-qo)<1e-9 and np.max(abs(gq-gqo))<1e-9
        gram_def=float(np.linalg.norm(Gram(1.)(n,h)-oldgram(n,h))/np.linalg.norm(oldgram(n,h)));assert gram_def<3e-11
        k0=Kinetic(0.);q0,g0,gc0=k0(n,h,v,c);q4=q-q0
        for beta in [.17,3.7]:
            kw=Kinetic(beta);qw,gw,gcw=kw(n,h,v,c);linear=abs(qw-(q0+beta*q4));assert linear<3e-9*max(1.,abs(qw))
            direction=np.zeros(len(state.x));sl=slice(0,52) if i==0 else slice(52,54);direction[sl]=np.sin(np.arange(sl.stop-sl.start)*.31+.2);direction/=np.linalg.norm(direction)
            exact=state.pull(i,gw)@direction[sl];checks=[]
            for step in [2e-5,1e-5]:
                fd=(kw(state.field_at(state.x+step*direction,i),h,v,c)[0]-kw(state.field_at(state.x-step*direction,i),h,v,c)[0])/(2*step)
                error=abs(fd-exact);assert error<4e-5*max(1.,abs(exact));checks.append({'step':step,'analytic':float(exact),'finite_difference':float(fd),'absolute_defect':float(error)})
            rows.append({'atom_index':i,'beta':beta,'original_energy_gradient_at_beta1_pass':True,'original_kinetic_at_beta1_pass':True,
                'fast_Gram_at_beta1_relative_defect':gram_def,'linear_kinetic_coefficient_absolute_defect':float(linear),'field_gradient_checks':checks})
    model=install(ReactiveContacts('SF4','current','mobile'),.17,k=1.3,gamma=2.1);native=model.check(state.x)
    dump(ROOT/'resultats/CONTROLES_PARAMETRES_R340.json',{'revision':340,'weighted_native_checks':rows,'current_curvature_whole_potential_checks':native,'physical_validation':False})
    print('R340 weighted energy/kinetic and full reactive-potential gradient controls passed',flush=True)

def scan(names,betas):
    rows=[];started=time.time()
    for name in names:
        state=load_state(name);g2,d2,g4,d4=split_primitives(state,name);model=ReactiveContacts('SF4','current','mobile');frames=np.concatenate([np.eye(3)[None],model.frames])
        for beta in betas:
            install(model,beta);base,C,pot=force_components(model,state.x);raw=[a+beta*b for a,b in zip(g2,g4,strict=True)];der=[a+beta*b for a,b in zip(d2,d4,strict=True)]
            I,D,G,B,E=assemble(state,raw,der,frames);Ic=B.T@I@B;Dc=np.einsum('pi,apq,qj->aij',B,D,B)
            result=fit(Ic,Dc,base,C,model.nstate);best=result['candidates'][0]
            row={'state':name,'beta':float(beta),'alpha':1.,'full_coordinates':state.x.tolist(),'potential':pot,'common_rotation_fit':result,
                'native_shapes_and_geometry_held_during_this_coefficient_fit':True,'calibration_not_independent_validation':True}
            rows.append(row);key=f'{name}_beta_{beta:.9g}'.replace('.','p')
            np.savez_compressed(ROOT/'results'/f'{key}_OPERATEURS_R340.npz',I=Ic,D=Dc,base_gradient=base,coefficient_gradients=C,full_metric=G,frames=frames,coordinates=state.x)
            row['operator_file']=f'{key}_OPERATEURS_R340.npz'
            dump(ROOT/'resultats/BRASSAGE_DEPENDANCES_R340.json',{'revision':340,'rows':rows,'completed':False,'physical_status':'UNKNOWN','physical_regions_excluded':0,'seconds':time.time()-started})
            print(name,'beta',beta,'gamma',round(best['gamma'],6),'k',round(best['kangle'],6),'g',round(best['state_force_inf_native'],6),'gc',round(best['center_force_inf_native'],6),'s',round(time.time()-started,1),flush=True)
    record={'revision':340,'rows':rows,'completed':True,'physical_status':'UNKNOWN','physical_regions_excluded':0,'seconds':time.time()-started,
        'beta_scan_is_algorithmic_not_physical_bounds':True,'same_beta_in_energy_metric_and_reactive_curvature':True,'all_continuous_conditions_exhausted':False}
    dump(ROOT/'resultats/BRASSAGE_DEPENDANCES_R340.json',record);return record

class UpdatedRouth(Routh):
    def __init__(self,row):
        p=row['common_rotation_fit']['candidates'][0];self.beta=row['beta'];self.kangle=p['kangle'];self.gamma=p['gamma'];self.mu0=np.array(p['momenta_native']).reshape(2,3)
        self.magnitudes=np.linalg.norm(self.mu0,axis=1);self.directions=np.divide(self.mu0,self.magnitudes[:,None],out=np.tile([1.,0.,0.],(2,1)),where=self.magnitudes[:,None]>1e-15);self.axes=[]
        for d in self.directions:
            t=np.cross(d,np.eye(3)[np.argmin(abs(d))]);t/=np.linalg.norm(t);self.axes.append(np.stack([t,np.cross(d,t)]))
        self.axes=np.array(self.axes);self.model=install(ReactiveContacts('SF4','current','mobile',stiffness_factor=max(self.gamma,1e-15)),self.beta,self.kangle,self.gamma)
        self.state=self.model.state;self.ns=self.model.nstate;self.nq=len(self.state.x);self.branch='R340';self.protocol='reactif_reajuste'
        self.x0=np.r_[row['full_coordinates'],np.zeros(4)];self.frames=np.concatenate([np.eye(3)[None],self.model.frames]);self.fast=Gram(self.beta);self.kernel=Kinetic(self.beta)
        self.calls=0;self.last=None;self.bounds=np.vstack([self.state.system.bounds,np.tile([-np.pi,np.pi],(4,1))]);self.history=[];self.started=time.time()
    def evaluate(self,x):
        energy,g=super().evaluate(x);self.last.update({'alpha':1.,'beta':self.beta,'existing_kangle':self.kangle,'reactive_curvature_recomputed_each_trial':True});return energy,g

def follow(record,maxiter):
    # Keep exact angle-calibrated candidates separate from the already released
    # R339 point. Select by native residual in fixed alpha=1 units, not by a
    # normalized residual whose denominator changes with beta.
    eligible=[r for r in record['rows'] if r['state'].startswith('R338_')]
    row=min(eligible,key=lambda r:r['common_rotation_fit']['candidates'][0]['native_force_and_torque_squared_norm'])
    m=UpdatedRouth(row);checks=m.controls(m.x0);print('updated Routh controls passed; selected',row['state'],row['beta'],flush=True)
    result=m.relax(maxiter);result.update({'revision':340,'source_row':row,'whole_gradient_initial_controls':checks,
        'no_angle_target_or_new_force_in_objective':True,'beta_gamma_kangle_and_prepared_momentum_modules_fixed_within_this_trial':True,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    dump(ROOT/'resultats/REAJUSTEMENT_CONJOINT_R340.json',result)
    x=np.r_[result['result']['full_coordinates'],result['result']['momentum_chart_coordinates']];endchecks=m.controls(x)
    dump(ROOT/'resultats/CONTROLES_FINAUX_CONJOINTS_R340.json',{'revision':340,'final_point_gradient_checks':endchecks,'replayed_final_point':m.last,'physical_status':'UNKNOWN'})
    return row,result

def transfer(row):
    beta=row['beta'];best=row['common_rotation_fit']['candidates'][0];model=install(ReactiveContacts('SF6','current','mobile'),beta)
    state=model.state;raw=[];der=[];kernel=Kinetic(beta)
    for i in range(len(state.fields)):
        G,D=primitive(kernel,state,i);raw.append(G);der.append(D);print('SF6 same-law metric',i+1,'/',len(state.fields),flush=True)
    frames=np.concatenate([np.eye(3)[None],model.frames]);I,D,G,B,E=assemble(state,raw,der,frames);I=B.T@I@B;D=np.einsum('pi,apq,qj->aij',B,D,B)
    base,C,pot=force_components(model,state.x);gains=np.array([best['gamma'],best['kangle']]);result=fit(I,D,base,C,model.nstate,fixed_gains=gains)
    np.savez_compressed(ROOT/'resultats/SF6_TRANSFERT_OPERATEURS_R340.npz',I=I,D=D,base_gradient=base,coefficient_gradients=C,fixed_gains=gains,coordinates=state.x)
    dump(ROOT/'resultats/SF6_TRANSFERT_MEMES_LOIS_R340.json',{'revision':340,'alpha':1.,'beta':beta,'gamma':float(gains[0]),'kangle':float(gains[1]),
        'only_prepared_velocity_refitted_coefficients_held':True,'native_fields_received_R336_held_not_independently_retuned':True,'potential':pot,'fit':result,
        'scope':'conditional cross-context check, not independent empirical validation or rejection of all SF6 states','physical_status':'UNKNOWN','physical_region_excluded':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checks-only',action='store_true');p.add_argument('--maxiter',type=int,default=30);p.add_argument('--follow-only',action='store_true');a=p.parse_args()
    if not a.follow_only:weighted_checks()
    if a.checks_only:sys.exit(0)
    record=json.loads((ROOT/'resultats/BRASSAGE_DEPENDANCES_R340.json').read_text()) if a.follow_only else scan(['R338_conservee','R338_souple','R339_libere'],[.001,.01,.1,.316227766,1.,3.16227766,10.,100.,1000.])
    row,result=follow(record,a.maxiter);transfer(row)
