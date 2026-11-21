'Distinguish positive density reweighting from a dilation-invariant density argument.'

import itertools
from pathlib import Path

from sweep_sf4 import read
from collection_curves import matrices
from collection_native_curves import native_fields
from calibrate_curves import M4,M64,M_RATIO,X,isotropic
from fields_and_references import dump
from curves_of_distribution import M6_FRACTION,laws,shape,solve
from dynamics_nodal_scripts import NodalDynamics
from trials_nodal import digest
from reader_transported import FixedReader
from read_equilibria import NATIVE,read_states
from relax_nodes import OUTPUT,ROOT
from connect_curves_kernel import F4,F6,exact_blocks,fit_native,load_bank,read_rotor
from sizes_substrate import nodes,stencils
import numpy as np
from scipy.optimize import linprog


SOURCE = Path(__file__).resolve()
BANK6 = OUTPUT/'BANQUE_POLI_N48_R354.npz'
STATE6 = OUTPUT/'ETAT_POLI_N48_R354.npz'
CORRECTED = OUTPUT/'ETAT_RECALE_N48_R354.npz'


def size(n,h):
    m = round(len(n)**(1/3))
    grid = n.reshape(m,m,m,4)
    M,R,S = 0.,np.zeros(3),0.
    for signs,(d,x) in zip(itertools.product((1,-1),repeat=3),stencils(n,h,nodes(m,h)),strict=True):
        axes = tuple(i for i,s in enumerate(signs) if s<0)
        cube = np.flip(grid,axis=axes) if axes else grid
        a = cube[:-1,:-1,:-1].reshape(-1,4)
        rho = np.abs(np.linalg.det(np.concatenate((a[:,None],d),axis=1)))
        M += float(rho.sum())
        R += rho@x
        S += float(rho@np.sum(x*x,axis=1))
    result = np.sqrt(S/M-(R/M)@(R/M))
    if result<=0.:
        raise ArithmeticError('Nonpositive topological support radius')
    return float(result)


def bounds(bins,constant):
    raw = matrices(bins)
    lower,upper = np.zeros(raw.shape[:2]),np.zeros(raw.shape[:2])
    roundoff = 96.*np.finfo(float).eps
    for j,(s,rows) in enumerate(zip(constant,raw,strict=True)):
        U = np.vstack((-np.linalg.solve(s[:3,:3],s[:3,3:]),np.eye(3)))
        for k,A in enumerate(rows):
            norm = float(np.max(abs(A)))
            pad = roundoff*max(norm,1e-300)
            upper[j,k] = np.trace(U.T@A@U)/3.+pad*(1.+np.linalg.norm(U)**2)
            eigen = np.linalg.eigvalsh(A[:3,:3])[0]-pad
            if eigen<=0.:
                continue
            P = -np.linalg.solve(A[:3,:3],A[:3,3:])
            V = np.vstack((P,np.eye(3)))
            residual = A[:3,:3]@P+A[:3,3:]
            correction = np.sum(residual**2)/(3.*eigen)
            lower[j,k] = max(0.,np.trace(V.T@A@V)/3.-correction-pad*(1.+np.linalg.norm(V)**2))
    return lower,upper


def certificate(b4,u4,b6,u6):
    l4,h4 = bounds(b4,u4)
    l6,h6 = bounds(b6,u6)
    lo,hi = np.vstack((l4,l6)),np.vstack((h4,h6))
    R = np.zeros((3,12))
    R[0,0],R[0,1] = 1.,-M_RATIO
    R[1,5:] = -M6_FRACTION
    R[1,5] += 1.
    R[2,:5],R[2,5:] = -M64,1.
    C = np.vstack((np.maximum(R,0.)@lo+np.minimum(R,0.)@hi,
                  -np.maximum(R,0.)@hi-np.minimum(R,0.)@lo))
    # Any satisfying profile has a nonnegative combination with C @ w <= 0.
    s = np.max(np.maximum(abs(lo),abs(hi)),axis=0)
    active = s>1e-24*max(s)
    A = C[:,active]/s[active]
    obj = np.r_[np.zeros(6),-1.]
    opt = linprog(obj,A_ub=np.column_stack((-A.T,np.ones(A.shape[1]))),
                  b_ub=np.zeros(A.shape[1]),A_eq=np.array([[1.,1.,1.,1.,1.,1.,0.]]),b_eq=[1.],
                  bounds=[(0.,1.)]*6+[(None,None)],method='highs')
    y = opt.x[:6] if opt.success else None
    tau = float(np.min(y@A)) if y is not None else None
    return {'binwise_mass_lower_and_upper_enclosure':True,
        'tested_density_bins_above_relative_support_1e_minus_24':int(active.sum()),
        'neglected_bin_total_unweighted_bound':float(sum(s[~active])),
        'dual_weights_on_three_lower_then_three_upper_inequalities':y,
        'minimum_signed_dual_margin_after_bin_scaling':tau,
        'positive_reweighting_refuted_on_retained_discrete_bins':bool(tau is not None and tau>1e-7),
        'scope':'fixed two preparations, one positive weight per retained density bin, stated local mass fractions'}


def template(b,sizes,catalog):
    x = X[None,:]*np.asarray(sizes)[:,None]**6
    f = np.array([shape(x,np.ones_like(x),c,w,0) for c,w,_ in catalog])
    return matrices(np.einsum('axt,kax->kat',b,f))


def exact_sized(n,h,beta,sizes,terms):
    return np.array([exact_blocks(f[None],{'h_native':[spacing]},beta,
                                  [{**t,'center':t['center']/R**6} for t in terms])[0]
                     for f,spacing,R in zip(n,h,sizes,strict=True)])


def main():
    model = NodalDynamics(48)
    n4,g4,beta = native_fields(model,F4)
    h4 = np.array(g4['h_native'])
    with np.load(STATE6) as z:
        n6,h6,r6,d6,o6 = [z[k] for k in ('fields','h_native','radii_native','directions','frames')]
    b4,u4,_,bank4 = load_bank(F4)
    with np.load(BANK6) as z:
        b6,u6 = z['bins'],z['constant']
    proof = certificate(b4,u4,b6,u6)
    print('Pure density cone',proof,flush=True)
    s4,s6 = [[size(n,h) for n,h in zip(fields,hs,strict=True)] for fields,hs in ((n4,h4),(n6,h6))]
    catalog = [(c,w,0) for c in np.geomspace(1e-20,1e6,135) for w in (.30,.55,.9,1.5,2.4)]
    p4,p6 = template(b4,s4,catalog),template(b6,s6,catalog)
    amp,check = solve(u4,u6,p4,p6,np.array([1.+.22/w for _,w,_ in catalog]))
    terms = laws(catalog,amp) if amp is not None else []
    record = {'revision':354,'full':True,'physical_validation':False,'pure_density_bin_certificate':proof,
        'size_definition':'RMS support of the absolute topological density about its own centroid',
        'shared_window_argument':'rho_squared * support_radius ** 6',
        'topological_rms_radius_native_by_site':{'received_SF4':s4,'relaxed_SF6':s6},
        'size_normalized_binned_catalog_count':len(catalog),'size_normalized_binned_fit':check,
        'candidate':None}
    files = [SOURCE,bank4,BANK6,STATE6,NATIVE,ROOT/'scripts'/'recaler_inertie_R354.py',
             ROOT/'scripts'/'courbes_de_repartition_R352.py',ROOT/'scripts'/'tailles_substrat_R351.py']
    print('Dilation-invariant density argument',s4,s6,check,terms,flush=True)
    if terms:
        constants,extra = {},{}
        for tag,n,h,R in ((F4,n4,h4,s4),(F6,n6,h6,s6)):
            C = exact_sized(n,h,beta,R,[])
            extra[tag] = np.array([exact_sized(n,h,beta,R,[{**t,'amplitude':1.}])-C for t in terms])
            constants[tag] = C
        v,audit = fit_native(constants,extra,np.array([t['amplitude'] for t in terms]))
        t = [{**f,'amplitude':float(a)} for f,a in zip(terms,v,strict=True) if a>1e-10]
        common = M4/isotropic(constants[F4]+np.tensordot(v,extra[F4],axes=1)).sum()
        candidate = {'exact_integral_mass_constraints':audit,'terms':t,'one_mass_unit_kg':common,'readings':{}}
        record['candidate'] = candidate
        k = read(NATIVE)
        scale = k['native_scale_per_angstrom']
        rows = [('SF4_retained',n4,h4,g4,None,None,None),('SF6_relaxed',n6,h6,None,r6,d6,o6)]
        if CORRECTED.exists():
            with np.load(CORRECTED) as z:
                rows.append(('SF6_recalibrated',z['fields'],z['h_native'],None,z['radii_native'],z['directions'],z['frames']))
            files.append(CORRECTED)
        for name,n,h,g,r,d,o in rows:
            for count in (48,64):
                fields = n if count==48 else FixedReader(n,64,3,None).anchor
                H = h if count==48 else h*47/63
                R = [size(f,hh) for f,hh in zip(fields,H,strict=True)]
                G = exact_sized(fields,H,beta,R,t)
                if g is None:
                    result = read_states(G,np.concatenate((np.eye(3)[None],o)),d,d,r,scale,common)
                else:
                    result = read_rotor(G,g,scale,common,False,np.array(k['native_SF4_torsions_rad']),None)
                candidate['readings'][f'{name}_N{count}'] = result
                print('Size-normalized candidate',name,count,result if g is not None else result['reads']['relaxed_geometry'],flush=True)
    record['input_sha256'] = {str(p.relative_to(ROOT)):digest(p) for p in files}
    dump(OUTPUT/'LOI_TAILLE_ET_CONE_R354.json',record)


if __name__=='__main__':
    main()
