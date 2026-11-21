'Full interior tangent-field acceleration correction by a 33x33 Schur solve.\n\nThe retained field coordinates are replaced by independent nodal tangent\naccelerations, rather than counted twice. Same native nodal A,C,G matrices.\nThis closes the instantaneous finite-grid EL balance; no temporal or\nphysical-domain closure is inferred.'
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import hashlib,json,time
import numpy as np
from extend_fields import EnrichedDynamics,ROOT,dump

def constrained_inverse(A,n):
    Ai=np.linalg.inv(A);b=np.einsum('nij,nj->ni',Ai,n);den=np.einsum('ni,ni->n',n,b)
    return Ai-np.einsum('ni,nj->nij',b,b)/den[:,None,None]

def run():
    start=time.time();d=EnrichedDynamics();y=d.initial();dy,H=d.evaluate(y,True);c=d.configuration
    source=ROOT/'resultats/ENRICHI_INITIAL_SECONDES_DIRECTIONS_NODALES_R345.npz';z=np.load(source)
    assert np.max(abs(z['initial_state']-y))<1e-12
    mask=c.state.m.trial.mask;rigid_indices=np.r_[np.arange(6),np.arange(66,93)]
    assert len(rigid_indices)==33
    D=np.zeros((33,33));S=np.zeros_like(D);rhs=np.zeros(33);cache=[];reference_norm=0.
    for i,(n,E) in enumerate(zip(c.state.fields,d.E,strict=True)):
        A,C,G=d.kernel.mass(n,c.model.h(i));Er=E[:9,rigid_indices];Di=Er.T@G@Er
        Ai=constrained_inverse(A[mask],n[mask]);B=C[mask]@Er;res=z[f'residual_{i}'][mask]
        W=Ai@B;ar=np.einsum('nij,nj->ni',Ai,res)
        subtract=np.einsum('nca,ncb->ab',B,W)
        D+=Di;S+=Di-subtract;rhs+=np.einsum('nca,nc->a',B,ar)
        reference_norm+=float(np.sum(res*ar))
        cache.append((A,C,Er,Ai))
    S=.5*(S+S.T);eig,V=np.linalg.eigh(S)
    diagnostics={'revision':345,'full':False,'interior_tangent_field_accelerations':int(mask.sum())*3*5,
        'remaining_rigid_and_reference_rates':33,'Schur_eigenvalues':eig,'Schur_rhs_native':rhs,
        'same_native_action':True,'no_regularization':True,'physical_validation':False}
    dump(ROOT/'resultats/FERMETURE_NODALE_R345.json',diagnostics)
    if eig[0]<1e-10:
        diagnostics['status']='UNQUALIFIED_SCHUR_RANK_NO_MASS_ADDED';diagnostics['full']=True
        dump(ROOT/'resultats/FERMETURE_NODALE_R345.json',diagnostics);print(diagnostics['status'],eig[:5],flush=True);return
    delta=np.linalg.solve(S,rhs);atoms=[];saved={'rigid_acceleration_correction':delta,'rigid_indices':rigid_indices,'Schur':S,'rhs':rhs};group_balance=D@delta
    direct_field_norm=0.;new_norm=0.;crosswork=0.
    for i,(n,old,items) in enumerate(zip(c.state.fields,[z[f'residual_{i}'] for i in range(5)],cache,strict=True)):
        A,C,Er,Ai=items;local=Er@delta;B=C[mask]@Er
        forcing=old[mask]+B@delta
        correction=-np.einsum('nij,nj->ni',Ai,forcing)
        full=np.zeros_like(n);full[mask]=correction
        shift=np.einsum('nij,nj->ni',A,full)+C@local
        new=old+shift;new-=n*np.einsum('ni,ni->n',n,new)[:,None]
        group_balance+=np.einsum('nca,nc->a',B,correction)
        dual_after=np.einsum('nij,nj->ni',Ai,new[mask]);norm_after=float(np.sqrt(max(0.,np.sum(new[mask]*dual_after))));new_norm+=norm_after**2
        tangent=float(np.max(abs(np.sum(n*full,axis=1))))
        atoms.append({'atom_index':i,'interior_residual_inf_native':float(np.max(abs(new[mask]))),
            'interior_mass_dual_residual_norm_native':norm_after,'outside_mask_residual_inf_native':float(np.max(abs(new[~mask]))),
            'field_acceleration_correction_inf_native':float(np.max(np.linalg.norm(correction,axis=1))),
            'tangency_inf_defect':tangent,'outside_mask_is_prescribed_not_declared_free':True})
        saved[f'acceleration_correction_{i}']=full;saved[f'final_residual_{i}']=new
        # Positive native Q for the acceleration direction checks the
        # Schur lift against the original quadrature kernel.
        qi,gn,gt,gv=d.kernel.quadratic(n,c.model.h(i),full,local);direct_field_norm+=qi
        crosswork+=float(np.sum(old[mask]*correction))
    minimum_direction=V[:,0];direct_minimum_Q=0.
    for i,(n,items) in enumerate(zip(c.state.fields,cache,strict=True)):
        A,C,Er,Ai=items;local=Er@minimum_direction
        free=np.zeros_like(n);free[mask]=-np.einsum('nij,nj->ni',Ai,C[mask]@local)
        direct_minimum_Q+=d.kernel.quadratic(n,c.model.h(i),free,local)[0]
    balance=float(np.max(abs(group_balance)));maximum=max(a['interior_residual_inf_native'] for a in atoms)
    assert balance<2e-7 and maximum<2e-8,(balance,maximum)
    assert abs(direct_minimum_Q-eig[0])<3e-8*max(1.,abs(eig[0]))
    assert abs(direct_field_norm+crosswork)<3e-7*max(1.,abs(direct_field_norm)),(direct_field_norm,crosswork)
    diagnostics.update({'full':True,'status':'INSTANTANEOUS_NATIVE_INTERIOR_EL_BALANCE_CLOSED','atoms':atoms,
        'interior_mass_dual_residual_norm_before':float(np.sqrt(reference_norm)),
        'interior_mass_dual_residual_norm_after':float(np.sqrt(new_norm)),
        'all_rigid_EL_correction_equations_inf_defect_native':balance,
        'direct_native_Q_of_correction':direct_field_norm,'negative_old_residual_work_pairing':-crosswork,
        'smallest_Schur_eigenvalue_direct_native_Q':direct_minimum_Q,
        'native_Q_vs_eigenvalue_absolute_defect':abs(direct_minimum_Q-eig[0]),
        'retained_old_field_modes_not_duplicated_as_extra_independent_coordinates':True,
        'all_1111320_interior_tangent_components_solved':int(mask.sum())*3*5==1111320,
        'boundary_reactions_not_a_free_exterior_or_measured_flux':True,
        'pointwise_closure_not_finite_time_stability_or_empirical_validation':True,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seconds':time.time()-start})
    dump(ROOT/'resultats/FERMETURE_NODALE_R345.json',diagnostics)
    np.savez_compressed(ROOT/'resultats/ACCELERATION_NODALE_R345.npz',**saved,initial_state=y)
    print('Full interior',int(mask.sum())*3*5,'components, residual',maximum,'dual',diagnostics['interior_mass_dual_residual_norm_after'],
        'rigid balance',balance,'Schur',eig[[0,-1]],'seconds',round(time.time()-start,1),flush=True)

if __name__=='__main__':run()
