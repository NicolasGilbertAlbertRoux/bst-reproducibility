'Recompute the field response and retain collective rotation in the kinetic pencil.'

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

from sweep_sf4 import read
from fields_and_references import dump
from trials_nodal import digest
from clock_inertia import H_SI,C_SI,MASSES_U
from read_nodes import charge
from modes_mixed import PAIR,SIMPLE,CONTRAST
from kernel_curves import CurveKinetic
from release_collar import face_interior
from relax_nodes import ROOT,NodalSF6,projected
from free_rotations import reduced
from vibrations_angles_moving import Tangent,STATE,BASIS,SIZES
import numpy as np
from scipy.linalg import block_diag,eigh,null_space
from scipy.optimize import linear_sum_assignment


OUTPUT = ROOT/'results'/'R356'
SOURCE = Path(__file__).resolve()
PARENT = ROOT/'results'/'R355'/'VIBRATIONS_ANGLES_MOBILES_R355.json'
N48 = 48
B_REF = .09107
M34_U = 33.967867004
RADIAL = np.column_stack((PAIR@SIMPLE,PAIR@CONTRAST,PAIR*np.tile([1.,-1.],3)[:,None]))
FAMILIES = {'A1g':slice(0,1),'Eg':slice(1,3),'T1u_radial':slice(3,6)}
LABELS = ('radii','bond_directions','port_directions','field_coordinates','global_rotation','translation','bond_roll')


def sym(M):
    return (M+M.T)*.5


def load_state():
    with np.load(STATE) as s:
        n,r,d,O,h = [s[k] for k in ('fields','radii_native','directions','frames','h_native')]
        c = float(s['contact_intercept_native'])
    with np.load(BASIS) as s:
        w = projected_seed(n,s['fields_basis'])
    old = read(SIZES)
    law = old['calibration_of_size_law']
    size = np.array(old['matched_preparation_results']['SF6_N48']['topological_support_radii_native'])
    weights = (size/law['reference_F_support_radius_native'])**law['power']
    return n,r,d,O,h,c,w,weights,old['common_length_scale_native_per_angstrom'],law['common_mass_unit_kg']


def projected_seed(n,raw):
    q = np.array([projected(n,v,face_interior(N48)) for v in raw])
    Q,_ = np.linalg.qr(q.reshape(len(q),-1).T,mode='reduced')
    return Q.T.reshape(q.shape)


def stationary_rates(n,A,C,mask):
    'Tangent minimizer of local kinetic energy at fixed spatial velocity.'
    a = np.linalg.solve(A,np.concatenate((C[:,:,3:9],n[:,:,None]),axis=-1))
    gauge = a[:,:,-1]
    w = a[:,:,:6]-gauge[:,:,None]*(np.einsum('ni,nij->nj',n,a[:,:,:6])
                                  /np.einsum('ni,ni->n',n,gauge)[:,None])[:,None,:]
    w[~mask] = 0.
    tangent = float(np.max(abs(np.einsum('ni,nij->nj',n,w))))
    error = A@w-C[:,:,3:9]
    error -= n[:,:,None]*np.einsum('ni,nij->nj',n,error)[:,None,:]
    if tangent>2e-12 or np.max(abs(error[mask]))>3e-12:
        raise ArithmeticError('The local kinetic minimizer failed its constrained equation')
    return -w,tangent


def construct_basis(n,mass,seed,mode):
    mask = face_interior(N48)
    if mode=='inherited':
        return seed,np.eye(len(seed)),[],None
    sites,ny = len(n),len(seed)
    nlocal = 6*sites
    Y = np.zeros((nlocal+ny,)+n.shape)
    seed_residual = seed.copy()
    local = []
    minima = []
    for i,(ni,(A,C,G)) in enumerate(zip(n,mass,strict=True)):
        w,err = stationary_rates(ni,A,C,mask)
        q,r = np.linalg.qr(w.reshape(-1,6),mode='reduced')
        if min(abs(np.diag(r)))<1e-9:
            raise ArithmeticError('The local frame response lost rank')
        rows = q.T.reshape(6,-1,4)
        Y[i*6:(i+1)*6,i] = rows
        seed_residual[:,i] -= ((seed[:,i].reshape(ny,-1)@q)@q.T).reshape(ny,-1,4)
        local.append({'site':i,'tangent_defect':err,'local_generator_coefficients':r,
                      'unrestricted_kinetic_minimizer_residual_scope':'allowed nodes, three spins and translations'})
        schur = sym(G[3:,3:]+np.einsum('nca,ncb->ab',C[:,:,3:],w))
        minima.append(schur)
    U,_ = np.linalg.qr(seed_residual.reshape(ny,-1).T,mode='reduced')
    Y[nlocal:] = U.T.reshape(seed_residual.shape)
    B = Y.reshape(len(Y),-1)
    S = B@seed.reshape(ny,-1).T
    if np.max(abs(S.T@S-np.eye(ny)))>2e-11:
        raise ArithmeticError('The old response directions were not contained in the enlarged basis')
    return Y,S,local,np.array(minima)


def raw_mass(T,blocks,weights):
    'Per-site contributions before elimination of the cyclic velocities.'
    m,dim = T.m,T.dimension
    total = dim+6+m
    h_fields = slice(5*m,dim)
    sites = []
    for i,((A,C,G),weight) in enumerate(zip(blocks,weights,strict=True)):
        V = np.zeros((9,total))
        O = np.eye(3) if not i else T.O[i-1]
        V[3:6,dim:dim+3] = O.T
        V[6:,dim+3:dim+6] = O.T
        if i:
            j = i-1
            d,r,f = T.d[j],T.r[j],T.P.f0[j]
            db,fb = T.B[j],T.P.B[j]
            V[6:,j] = O.T@d
            V[6:,m+2*j:m+2*j+2] = r*O.T@db.T
            V[6:,dim:dim+3] = O.T@np.cross(np.eye(3),r*d).T
            V[3:6,m+2*j:m+2*j+2] = O.T@np.cross(d,db).T
            V[3:6,3*m+2*j:3*m+2*j+2] = -np.cross(f,fb).T
            V[3:6,dim+6+j] = f
        q = V.T@G@V
        # Most directions are local to a different site. Compute only nonzero columns.
        use = np.flatnonzero(np.any(T.Y[:,i]!=0.,axis=(1,2)))
        J = np.ascontiguousarray(np.moveaxis(T.Y[use,i],0,-1))
        F = J.reshape(-1,len(use))
        h_ids = 5*m+use
        q[np.ix_(h_ids,h_ids)] += F.T@(A@J).reshape(F.shape)
        cross = F.T@C.reshape(-1,9)@V
        q[h_ids,:] += cross
        q[:,h_ids] += cross.T
        sites.append(weight*sym(q))
    return np.array(sites)


def field_hessian(T,old,S,mode):
    D = T.dimension
    P = block_diag(np.eye(30),S)
    if mode=='inherited':
        return old,{'inherited_R355_matrix':True,'old_projected_defect':0.}
    rows = np.zeros((D,D))
    rows[:30,:30] = old[:30,:30]
    x = np.zeros(D)
    start = perf_counter()
    for i in range(30,D):
        x[i] = 1e-4
        a = T.evaluate(x)[1]
        x[i] = -1e-4
        b = T.evaluate(x)[1]
        x[i] = 0.
        rows[:,i] = (a-b)/(2e-4)
        if (i-29)%5==0 or i==D-1:
            print('field Hessian',i-29,'of',D-30,'s',round(perf_counter()-start,1),flush=True)
    asymmetry = float(np.max(abs(rows[30:,30:]-rows[30:,30:].T)))
    rows[30:,:30] = rows[:30,30:].T
    rows = sym(rows)
    errs = []
    for phase in (0.2,0.9):
        v = np.sin(np.arange(D)*.37+phase)
        v /= np.linalg.norm(v)
        pair = []
        for eps in (2e-4,1e-4):
            pair.append((T.evaluate(eps*v)[1]-T.evaluate(-eps*v)[1])/(2.*eps))
        errs.append({'projected_defect_inf':float(np.max(abs(rows@v-pair[1]))),
                     'step_change_inf':float(np.max(abs(pair[0]-pair[1])))})
    old_defect = float(np.max(abs(P.T@rows@P-old)))
    if max(q['projected_defect_inf'] for q in errs)>8e-6 or old_defect>8e-6 or asymmetry>8e-6:
        raise ArithmeticError(f'Hessian projection failed {old_defect}, {asymmetry}, {errs}')
    return rows,{'inherited_R355_geometry_block':True,'field_block_antisymmetry':asymmetry,
                 'old_projected_defect':old_defect,'two_step_checks':errs}


def centroid_control(n,Y,h):
    active = face_interior(N48)
    N = len(Y)
    def local(i):
        z,hi = n[i],h[i]
        c = charge(z,N48,hi)['centroid_native']
        jj = np.flatnonzero(np.any(Y[:,i]!=0.,axis=(1,2)))
        J = np.zeros((3,N))
        for j in jj:
            w = Y[j,i]
            plus,minus = z+1e-4*w,z-1e-4*w
            for v in (plus,minus):
                v /= np.linalg.norm(v,axis=1)[:,None]
                v[~active] = z[~active]
            J[:,j] = (charge(plus,N48,hi)['centroid_native']-charge(minus,N48,hi)['centroid_native'])/(2e-4)
        test = np.sin(np.arange(len(jj))*.4+.3)
        v = np.einsum('i,inc->nc',test,Y[jj,i])
        a,b = z+5e-5*v,z-5e-5*v
        for field in (a,b):
            field /= np.linalg.norm(field,axis=1)[:,None]
            field[~active] = z[~active]
        err = np.linalg.norm((charge(a,N48,hi)['centroid_native']-charge(b,N48,hi)['centroid_native'])/1e-4
                             -J[:,jj]@test)
        return c,J,float(err)
    with ThreadPoolExecutor(max_workers=5) as pool:
        out = list(pool.map(local,range(len(n))))
    C = np.concatenate([q[1] for q in out],axis=0)
    U = null_space(C,rcond=1e-9)
    err = max(q[2] for q in out)
    if len(U.T)!=N-3*len(n) or err>2e-7:
        raise ArithmeticError(f'The fixed-centre linear constraint failed rank or replay: {U.shape}, {err}')
    return np.array([q[0] for q in out]),C,U,err


def constants(I,k):
    ev = np.linalg.eigvalsh(sym(I))
    return k/ev if min(ev)>0. else None


def vibration(K,M,factor):
    s,V = eigh(K,M)
    w = np.sign(s)*np.sqrt(abs(s))*factor
    raw = RADIAL.T@V[:6]
    probes = {}
    for k,sl in FAMILIES.items():
        strength = np.sum(raw[sl]**2,axis=0)
        strength /= sum(strength)
        top = np.argsort(strength)[-6:][::-1]
        probes[k] = {'residues':strength,'top_indices':top,'top_poles_cm_inverse':w[top],
                     'top_fractions':strength[top]}
    defect = float(np.linalg.norm(K@V-(M@V)*s)/max(1.,np.linalg.norm(K@V)))
    if defect>2e-7:
        raise ArithmeticError(f'Generalized vibration eigensolve defect {defect}')
    return {'native_squared_rates':s,'signed_cm_inverse':w,'probes':probes,
            'negative_count_below_1e_minus_8':int(sum(s<-1e-8)),'relative_pencil_defect':defect},V


def rotation(K,MR,factor,rotconst):
    D = len(K)
    M,C,I = MR[:D,:D],MR[:D,D:],MR[D:,D:]
    s,V = eigh(K,M)
    Ctilde = V.T@C
    free = sym(I-C.T@np.linalg.solve(M,C))
    if min(s)<=0.:
        return {'positive_internal_stiffness_required':True,'min_native_rate':float(min(s)),
                'co_transported_B_cm_inverse':constants(I,rotconst),'high_frequency_B_cm_inverse':constants(free,rotconst)}
    def at(f):
        q = (f/factor)**2
        mat = I+(Ctilde.T*(q/(s-q)))@Ctilde
        return mat,constants(mat,rotconst)
    fgrid = np.geomspace(.008,1100.,400)
    samples = []
    for f in fgrid:
        m,b = at(f)
        samples.append([f,*([None]*3 if b is None else b.tolist())])
    obs = {}
    for f in (0.,B_REF,2.*B_REF,4.*B_REF,1.,5.,50.,642.,774.):
        eff,b = at(f)
        q = (f/factor)**2
        exact = I+q*C.T@np.linalg.solve(K-q*M,C)
        defect = float(np.max(abs(exact-eff)))
        if defect>3e-6*max(1.,float(np.max(abs(exact)))):
            raise ArithmeticError('Modal rotation response did not reproduce the resolvent')
        obs[str(f)] = {'B_cm_inverse':b,'inertia_eigenvalues_native':np.linalg.eigvalsh(sym(eff)),
                       'nearest_pole_gap_cm_inverse':float(min(abs(np.sqrt(s)*factor-f))),
                       'rotational_response_defect_inf':defect}
    strength = np.sum(Ctilde**2,axis=1)
    return {'co_transported_B_cm_inverse':constants(I,rotconst),'high_frequency_B_cm_inverse':constants(free,rotconst),
            'internal_poles_cm_inverse_at_fixed_global_pose':np.sqrt(s)*factor,
            'normalized_rotation_residue':strength/strength.sum(),'probe_frequencies':obs,
            'scan_cm_inverse':samples,'quantum_transition_frequency_used_only_as_a_probe_scale':True}


def modes_and_rotation(K,Msites,sub,factor,rotconst):
    D = len(K)
    P = block_diag(sub,np.eye(Msites.shape[1]-D))
    site = np.einsum('ai,nab,bj->nij',P,Msites,P)
    K = sym(sub.T@K@sub)
    N = len(K)
    M = site.sum(axis=0)
    d = N+3
    MR = reduced(M,d)
    MV = reduced(MR,N)
    if min(np.linalg.eigvalsh(MV))<=0. or min(np.linalg.eigvalsh(MR))<=0.:
        raise ArithmeticError('The selected branch has a nonpositive metric')
    v,V = vibration(K,MV,factor)
    sr = M+((M34_U/MASSES_U[0])-1.)*site[0]
    Mi = reduced(reduced(sr,d),N)
    iso,W = vibration(K,Mi,factor)
    pair = abs(V.T@MV@W)
    row,col = linear_sum_assignment(-pair)
    candidates = np.array(v['probes']['T1u_radial']['top_indices'])[:3]
    choose = col[candidates]
    cross = v['signed_cm_inverse'][candidates]
    more = iso['signed_cm_inverse'][choose]
    return {'mechanical_coordinate_count':N,'vibration':v,'rotation_response':rotation(K,MR,factor,rotconst),
            'kinetic_min_eig_native':float(min(np.linalg.eigvalsh(MV))),
            'sulfur_mass_only_isotope_probe':{'32_poles_cm_inverse':cross,'34_poles_cm_inverse':more,
                '34_minus_32_cm_inverse':more-cross,'matched_mass_metric_overlaps':pair[candidates,choose],
                'measured_nu3_32_34_cm_inverse':[947.9,930.5],
                'three_strongest_odd_radial_residues_only_no_band_assignment':True},
            'rigid_angular_momenta_constrained_zero_for_vibrations':True},(K,MR,MV)


def main(mode):
    n,r,d,O,h,c,seed,weights,scale,unit = load_state()
    old = read(PARENT)
    base = old['cases']['N48_size_power_R355']
    anchor = base['spectra']['angular_and_field_mobile']
    factor = 774./np.sqrt(anchor['native_squared_rates'][anchor['symmetric_anchor_index']])
    rotconst = H_SI/(8.*np.pi**2*100.*C_SI*unit*(1e-10/scale)**2)
    model = NodalSF6(workers=6)
    try:
        model.mask = face_interior(N48)
        model.model.geometry.d['r0'] = c
        kern = CurveKinetic(model.model.common['beta'],[])
        mass = [kern.mass(a,b) for a,b in zip(n,h,strict=True)]
        Y,S,locals_,minima = construct_basis(n,mass,seed,mode)
        T = Tangent(model,n,r,d,O,Y)
        Msites = raw_mass(T,mass,weights)
        del mass
        D = T.dimension
        Kold = np.array(old['native_Hessian'])
        Mproj = block_diag(np.eye(30),S,np.eye(12))
        native = reduced(Mproj.T@Msites.sum(axis=0)@Mproj,33)
        merr = float(np.max(abs(native-np.array(base['reduced_metric']))))
        if merr>2e-9:
            raise ArithmeticError(f'Kinetic R355 replay changed by {merr}')
        base_branch,_ = modes_and_rotation(Kold,np.einsum('ai,nab,bj->nij',Mproj,Msites,Mproj),
                                            np.eye(33),factor,rotconst)
        print('R355 B co-transport, 2B probe, inertia-only',
              base_branch['rotation_response']['co_transported_B_cm_inverse'],
              base_branch['rotation_response']['probe_frequencies'][str(2.*B_REF)]['B_cm_inverse'],
              base_branch['rotation_response']['high_frequency_B_cm_inverse'],flush=True)
        files = (SOURCE,STATE,BASIS,SIZES,PARENT,ROOT/'scripts'/'vibrations_angles_mobiles_R355.py',
                 ROOT/'scripts'/'lire_noeuds_R354.py',ROOT/'scripts'/'noyau_courbes_R352.py')
        info = {'revision':356,'full':True,'physical_validation':False,'grid':N48,
                'frozen_R355_scale_cm_inverse_per_native_rate':factor,'NIST_B_reference_cm_inverse':B_REF,
                'R355_kinetic_matrix_defect_inf':merr,'input_sha256':{str(f.relative_to(ROOT)):digest(f) for f in files}}
        if mode=='inherited':
            dump(OUTPUT/'REPONSE_REPERES_R355_REJOUEE_R356.json',{**info,'inherited_33_coordinates':base_branch})
            return
        K,checks = field_hessian(T,Kold,S,mode)
        unrestricted,_ = modes_and_rotation(K,Msites,np.eye(D),factor,rotconst)
        print('unrestricted',unrestricted['vibration']['probes']['Eg']['top_poles_cm_inverse'],flush=True)
        positions,C,U,ce = centroid_control(n,Y,h)
        sub = block_diag(np.eye(30),U)
        centred,(Kc,Mc,Mv) = modes_and_rotation(K,Msites,sub,factor,rotconst)
        print('centres held',centred['vibration']['probes']['Eg']['top_poles_cm_inverse'],flush=True)
        matrix = OUTPUT/'MATRICES_REPERES_R356.npz'
        np.savez_compressed(matrix,raw_mass_sites=Msites,projected_K=K,inherited_field_coordinates=S,
                           centroid_linear_map=C,centroid_kernel=U,centred_K=Kc,centred_M_rotation=Mc,
                           centred_M_vibration=Mv,rotconst=rotconst,frequency_factor=factor)
        dump(OUTPUT/'REPONSE_REPERES_ET_CHAMPS_R356.json',{**info,'basis_size':len(Y),
              'basis_order':LABELS,'local_minimizers':locals_,'site_minimized_spatial_metrics':minima,
              'kinetic_basis_contains_the_unconstrained_site_spin_and_translation_minimizers':True,
              'centroid_coordinates_at_rest_native':positions,'centroid_constraint_rank':int(np.linalg.matrix_rank(C)),
              'centroid_finite_difference_replay_error_max':ce,'hessian_checks':checks,
              'inherited_33_coordinates':base_branch,'all_field_directions_free':unrestricted,
              'local_signed_topological_centroids_held':centred,'matrix_archive':matrix.name,'matrix_sha256':digest(matrix)})
    finally:
        model.close()


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('basis',choices=('inherited','enriched'))
    main(p.parse_args().basis)
