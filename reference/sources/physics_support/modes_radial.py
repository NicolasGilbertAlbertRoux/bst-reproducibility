'Compare the reserved even radial mode with the frozen and relaxed field Hessians.'

import argparse
from multiprocessing import get_context
from pathlib import Path

from sweep_sf4 import read
from fields_and_references import dump
from trials_nodal import digest
from inertia_and_entrainment import free_local
from reader_transported import FixedReader,tensor_map
from read_equilibria import DIRECT,NATIVE,read_states
from prepare_contact_prestressed import bare_curvature
from match_contact import FREQUENCIES,TransferredContact
from relax_sf6 import OUTPUT,ROOT
from kernel_curves import CurveKinetic
from free_rotations import reduced
import relax_sf6 as solver
import numpy as np
from scipy.linalg import eigvalsh


SOURCE = Path(__file__).resolve()
CORRECTED = OUTPUT/'CONTACT_RECALIBRE_R353.json'


def field_hessian(model,z,workers):
    # The finite-difference prediction uses current rather than fixed contact curvature.
    solver.FD_MODEL = model
    _,g = model.profile(z)
    eps = 4e-5
    tasks = [(z,j,eps,g,'current') for j in range(model.n)]
    with get_context('fork').Pool(workers) as pool:
        cols = np.asarray(pool.map(solver.hessian_column,tasks,chunksize=1))
    raw = cols[:,:model.n].T
    H = .5*(raw+raw.T)
    eig,Q = np.linalg.eigh(H)
    if eig[0]<=0.:
        raise ArithmeticError('The field Hessian is not positive in the explored chart')
    v = np.r_[Q[:,0],np.zeros(len(z)-model.n)]
    secants = []
    for step in (2e-4,1e-4):
        _,gp = model.profile(z+step*v)
        _,gm = model.profile(z-step*v)
        fd = (gp[:model.n]-gm[:model.n])/(2*step)
        secants.append({'step':step,'curvature':float(v[:model.n]@fd),
                        'Hessian_direction_residual_inf':float(max(abs(fd-H@v[:model.n])))})
    return H,{'one_sided_step':eps,'antisymmetric_part_before_symmetrization_inf':float(max(abs((raw-raw.T).ravel()))),
              'profiled_field_hessian_eigenvalues':eig,'softest_direction_centred_differences':secants}


def radial_mass(grams,dirs,frames):
    local = np.array([free_local(G) for G in grams])
    V = np.zeros((7,3,9))
    V[:,:,6:] = frames.transpose(0,2,1)
    for j,D in enumerate(dirs,1):
        V[j,:,j-1] = frames[j].T@D
    mass = np.einsum('nai,nab,nbj->ij',V,local,V)
    return reduced(mass,6)



def normalized_rates(state,Jr,fields,norms):
    rates = []
    for i,(f,tangent,parts,norm) in enumerate(zip(fields,state.tangent,state.parts,norms,strict=True)):
        loc = slice(50,52) if i==0 else slice(50+2*i,52+2*i)
        d = tangent[:,:,None]*(parts@Jr[loc])[:,None,:]
        if i==0:
            d += state.pull_S[:,:,None]*(state.Z@Jr[:50])[:,None,:]
        rates.append((d-f[:,:,None]*np.einsum('nc,nca->na',f,d)[:,None,:])/norm[:,None,None])
    return rates


def interpolate_rates(reader,rates):
    output = []
    for n,h,weight in zip(reader.anchor,rates,reader.weight,strict=True):
        d = np.stack([tensor_map(h[:,:,j],reader.P,48) for j in range(h.shape[2])],axis=2)
        output.append((d-n[:,:,None]*np.einsum('nc,nca->na',n,d)[:,None,:])/weight[:,:,None])
    return output


def effective_masses(grid,spacing,rates,dirs,frames,beta,terms):
    kernel = CurveKinetic(beta,terms)
    rigid,delta = np.zeros((30,30)),np.zeros((30,30))
    local_grams = []
    for i,(field,h,J,O) in enumerate(zip(grid,spacing,rates,frames,strict=True)):
        A,C,G = kernel.mass(field,h)
        local_grams.append(G[3:,3:])
        V = np.zeros((9,30))
        V[3:6,9+3*i:12+3*i] = np.eye(3)
        V[6:,6:9] = O.T
        if i:
            V[6:,i-1] = O.T@dirs[i-1]
        rigid += V.T@G@V
        flat = J.reshape(-1,6)
        delta[:6,:6] += flat.T@(A@J).reshape(-1,6)
        mixed = (flat.T@C.reshape(-1,9))@V
        delta[:6] += mixed
        delta[:,:6] += mixed.T
    grams = np.array(local_grams)
    fast,slow = reduced(rigid,6),reduced(rigid+delta,6)
    if np.max(abs(fast-radial_mass(grams,dirs,frames)))>2e-8:
        raise ArithmeticError('The zero-shape-velocity Schur reduction changed')
    return grams,fast,slow

def measure(K,M):
    pair = np.repeat(np.eye(3),2,axis=0)/np.sqrt(2.)
    C,W = pair.T@K@pair,pair.T@M@pair
    a = np.ones(3)/np.sqrt(3.)
    E = np.column_stack((np.array([1.,-1.,0.])/np.sqrt(2.),
                         np.array([1.,1.,-2.])/np.sqrt(6.)))
    w2 = (a@C@a)/(a@W@a)
    ratios = np.sqrt(eigvalsh(E.T@C@E,E.T@W@E)/w2)
    held = FREQUENCIES['Eg']/FREQUENCIES['A1g']
    return {'sym_stretch_native_squared_rate':float(w2),'Eg_over_A1g_ratios':ratios,
            'Eg_cm_inverse_when_A1g_is_anchored':FREQUENCIES['A1g']*ratios,
            'Eg_relative_errors_to_reference':ratios/held-1.,
            'pair_even_radial_stiffness':C,'pair_even_radial_mass':W,
            'offdiagonal_stiffness_to_diagonal_scale':float(np.max(abs(C-np.diag(np.diag(C))))/(np.trace(C)/3.)),
            'A1g_Eg_stiffness_mixing_relative':float(np.linalg.norm(a@C@E)/(a@C@a))}


def main(workers):
    model = TransferredContact()
    rec = read(CORRECTED)
    final = rec['final']
    x = np.array(final['coordinates'])
    z = x[model.active]
    H,checks = field_hessian(model,z,workers)
    model.m.evaluate(x)
    r = np.asarray(final['radii_native'])
    d = np.asarray(final['directions'])
    frames = np.concatenate((np.eye(3)[None],model.m.frames))
    a0 = model.ideal_directions
    raw = np.array(model.m.state.fields)
    norms = np.linalg.norm(raw,axis=2)
    fields = raw/norms[:,:,None]
    hs = np.array([model.m.h(i) for i in range(7)])
    c = model.m.reduction['contacts']
    U = model.m.last['contacts']
    Kd = np.array([p['kappa']+bare_curvature(model.well,u) for p,u in zip(c,r,strict=True)])
    cross = np.array([v['delta']*p['kappa_state_gradient']-p['kappa']*p['port_state_gradient']
                      for p,v in zip(c,U,strict=True)])
    K0 = np.diag(Kd)
    Hqq = H+cross.T@(cross/Kd[:,None])
    Jr = -np.linalg.solve(Hqq,cross.T)
    K1 = K0+cross@Jr
    rates = normalized_rates(model.m.state,Jr,fields,norms)
    # Direct differential check for the full-space radial-to-field block.
    kdir = np.sin(.2+np.arange(model.n)*.31)
    kdir /= np.linalg.norm(kdir)
    h = 4e-5
    perturb = np.r_[kdir,np.zeros(18)]
    model.m.evaluate(x+h*perturb)
    plus = np.array([u['radial_force'] for u in model.m.last['contacts']])
    model.m.evaluate(x-h*perturb)
    minus = np.array([u['radial_force'] for u in model.m.last['contacts']])
    cross_defect = float(max(abs((plus-minus)/(2*h)-cross@kdir)))
    if cross_defect>2e-4:
        raise ArithmeticError(f'The radial-to-field Hessian replay failed: {cross_defect}')
    mass_laws = read(NATIVE),read(DIRECT)
    laws = {'native_R352':(mass_laws[0]['native_terms'],mass_laws[0]['native_common_mass_unit_kg']),
            'direct_R352':(mass_laws[1]['fitted_curve'],mass_laws[1]['calibration_and_holdouts']['common_mass_unit_kg'])}
    scale = model.common['scale_native_per_angstrom']
    readings,modes = {},{}
    fine = FixedReader(fields,64,3,None)
    fine_rates = interpolate_rates(fine,rates)
    for count,grid,spacing,J in ((48,fields,hs,rates),(64,fine.anchor,hs*47/63,fine_rates)):
        for name,(terms,unit) in laws.items():
            G,M0,M1 = effective_masses(grid,spacing,J,d,frames,model.common['beta'],terms)
            key = f'N{count}_{name}'
            readings[key] = read_states(G,frames,d,a0,r,scale,unit)
            modes[key] = {'fixed_internal_fields':measure(K0,M0),'quasistatic_internal_fields':measure(K1,M1),
                          'quasistatic_induced_kinetic_relative_correction':
                              float(np.linalg.norm(M1-M0)/np.linalg.norm(M0))}
            print(key,'reserved Eg ratio',modes[key]['quasistatic_internal_fields']['Eg_over_A1g_ratios'],
                  'ABC',readings[key]['reads']['relaxed_geometry']['rotation_constants_cm_inverse'],flush=True)
    q = (FREQUENCIES['Eg']/FREQUENCIES['A1g'])**2
    reference = {'bands_cm_inverse':FREQUENCIES,'Eg_over_A1g':float(np.sqrt(q)),
        'equal_mass_cubic_pair_stiffness_model_required_offdiagonal_to_diagonal_ratio':(1.-q)/(1.+2.*q)}
    files = (SOURCE,CORRECTED,NATIVE,DIRECT,ROOT/'scripts'/'relaxer_SF6_R353.py',
             ROOT/'scripts'/'lire_equilibres_R353.py',ROOT/'scripts'/'raccorder_contact_R353.py')
    dump(OUTPUT/'MODES_RADIAUX_ET_ROTATION_R353.json',{'revision':353,'full':True,'physical_validation':False,
        'scope':'even radial tangent, held directions, fast and quasi-static internal limits; not a full vibration spectrum',
        'reserved_frequency_comparison':reference,'field_hessian_controls':checks,
        'radial_to_field_directional_difference_inf':cross_defect,'fixed_field_radial_Hessian':K0,
        'adiabatic_field_radial_Hessian':K1,'quasistatic_chart_velocity_per_radius':Jr,
        'readings':readings,'radial_modes':modes,
        'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in files}})


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=6)
    main(p.parse_args().workers)
