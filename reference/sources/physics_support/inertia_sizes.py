'Fit mass on the SF4 anchor, then read SF6 and both sets of rotations.'

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sweep_sf4 import read
from collection_native_curves import native_density
from collection_curves import matrices
from calibrate_curves import M4,M6,REF,SF6_REF,isotropic
from fields_and_references import dump
from curves_of_distribution import laws,solve
from trials_nodal import digest
from clock_inertia import MASSES_U,U_SI
from inertia_and_entrainment import Rotor
from reader_transported import FixedReader
from read_nodes import mass_laws
from law_size import certificate,exact_sized,size,template
from refit_inertia import bases,exact_template
from release_sf4 import OUTPUT,R354,ROOT
from connect_curves_kernel import F4,F6,exact_blocks,fit_native,read_rotor
from free_rotations import CALIBRATION
from rotations_spatial import skew
import numpy as np


SOURCE = Path(__file__).resolve()
SF4 = OUTPUT/'ETAT_SF4_OPPOSITION_R355.npz'
SF6 = R354/'ETAT_FACES_FIXES_N48_R354.npz'


def load(path):
    with np.load(path) as z:
        n,h,r,d,O = [z[k] for k in ('fields','h_native','radii_native','directions','frames')]
    g = {'h_native':h,'centers':np.vstack((np.zeros(3),r[:,None]*d)),
         'frames':np.concatenate((np.eye(3)[None],O))}
    return n,g


def report(G,g,scale,massunit,is_six):
    n = len(G)-1
    radii = np.linalg.norm(g['centers'][1:],axis=1)
    free = read_rotor(G,g,scale,massunit,is_six,np.zeros(n),radii)
    limits = {}
    embeddings = np.array([np.block([[O.T,np.zeros((3,3))],[-O.T@skew(x),O.T]])
                           for x,O in zip(g['centers'],g['frames'],strict=True)])
    m = np.einsum('nai,nab,nbj->ij',embeddings,G,embeddings)
    w = np.linalg.eigvalsh(m[:3,:3]-m[:3,3:]@np.linalg.solve(m[3:,3:],m[3:,:3]))
    v = np.linalg.eigvalsh(Rotor(g,G,scale,np.full(3,SF6_REF) if is_six else REF).matrix(np.zeros(n)))
    if min(w)<=0. or min(v)<=0.:
        raise ArithmeticError('A rotation inertia limit is not positive')
    rf = np.asarray(free['rotation_constants_cm_inverse'])
    I = Rotor(g,G,scale,np.full(3,SF6_REF) if is_six else REF).read(np.zeros(n))['rotation_moments_native']
    limits['locked_frames_rotation_constants_cm_inverse'] = rf*np.asarray(I)/w
    limits['free_frames_rotation_constants_cm_inverse'] = rf
    if np.any(limits['locked_frames_rotation_constants_cm_inverse']>rf+1e-10):
        raise ArithmeticError('Free-frame minimization should not raise a principal inertia')
    iso = isotropic(G)*massunit/U_SI
    return {**free,**limits,'translation_mass_site_u':iso,
            'translation_mass_site_errors_relative':iso/np.r_[MASSES_U[0],np.full(n,MASSES_U[1])]-1.}


def main():
    inputs = {F4:load(SF4),F6:load(SF6)}
    c = read(CALIBRATION)['candidate']
    beta,scale = c['beta'],c['scale_native_per_angstrom']
    law_bank = {k:terms for k,(terms,_) in mass_laws().items()}
    laws_received = {k:v[1] for k,v in mass_laws().items()}
    law_bank['unweighted'] = []
    grids = {}
    for name,(n,g) in inputs.items():
        h = g['h_native']
        fine = FixedReader(n,64,3,None).anchor
        for count,fields,hs in ((48,n,h),(64,fine,h*47/63)):
            gg = {**g,'h_native':hs}
            R = np.array([size(f,s) for f,s in zip(fields,hs,strict=True)])
            G = {k:exact_blocks(fields,gg,beta,terms) for k,terms in law_bank.items()}
            grids[name,count] = fields,gg,R,G
    low = grids[F4,48]
    m = isotropic(low[3]['unweighted'])
    R = low[2]
    ratio = np.log(R[0]/R[1])
    if abs(ratio)<1e-6:
        raise ArithmeticError('The two mass anchors cannot identify a size exponent')
    nu = float(np.log((MASSES_U[0]/MASSES_U[1])/(m[0]/m[1]))/ratio)
    u = float(MASSES_U[1]*U_SI/m[1])
    one_case_units = {k:float(M4/isotropic(g).sum()) for k,g in low[3].items()}
    outcomes = {}
    for (name,count),(n,g,sizes,blocks) in grids.items():
        out = {k:report(G,g,scale,one_case_units[k],name==F6) for k,G in blocks.items()}
        out['frozen_R352'] = {k:report(blocks[k],g,scale,v,name==F6) for k,v in laws_received.items()}
        scaled = blocks['unweighted']*(sizes/R[1])[:,None,None]**nu
        out['common_size_power'] = report(scaled,g,scale,u,name==F6)
        out['topological_support_radii_native'] = sizes
        outcomes[f'{name[:3]}_N{count}'] = out
        print(name,count,'R',sizes,'size power',nu,'mass',out['common_size_power']['total_mass_error_with_common_unit'],
              'rot',out['common_size_power']['rotation_errors_relative'],flush=True)
    all_banks = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name in (F4,F6):
            n,g,_,_ = grids[name,48]
            all_banks[name] = np.array(list(pool.map(lambda args:native_density(*args,beta),zip(n,g['h_native'],strict=True))))
    constant = {k:matrices(v.sum(axis=1)) for k,v in all_banks.items()}
    by_catalog = {}
    centers = np.geomspace(1e-19,100.,103)
    widths = (.35,.60,1.0,1.8,2.8)
    for method in ('raw_density','density_times_support_radius_power_six'):
        catalog = [(c,w,0) for c in centers for w in widths]
        def integrate(B,name):
            return bases(B,catalog) if method=='raw_density' else template(B,grids[name,48][2],catalog)
        g4,g6 = constant[F4],constant[F6]
        windows = [integrate(all_banks[k],k) for k in (F4,F6)]
        coeff,diagnostic = solve(g4,g6,*windows,np.array([1.+.22/w for _,w,_ in catalog]))
        out = {'catalog_size':len(catalog),'fit_uses_SF4_and_SF6_masses':True,
               'binned_feasible':coeff is not None,'binned_diagnostic':diagnostic}
        if coeff is not None:
            params = laws(catalog,coeff)
            exact = {}
            for name in (F4,F6):
                n,g,sizes,_ = grids[name,48]
                if method=='raw_density':
                    _,exact[name] = exact_template(n,g['h_native'],beta,params)
                else:
                    exact[name] = np.array([exact_sized(n,g['h_native'],beta,sizes,[{**p,'amplitude':1.}])-constant[name]
                                            for p in params])
            amp,check = fit_native(constant,exact,np.array([p['amplitude'] for p in params]))
            out['exact_diagnostic'] = check
            out['exact_terms'] = [{**p,'amplitude':float(v)} for p,v in zip(params,amp,strict=True)]
            if check['mass_constraints_met']:
                g4 = constant[F4]+np.tensordot(amp,exact[F4],axes=1)
                unit = float(M4/isotropic(g4).sum())
                out['exact_case_readings'] = {}
                for (name,count),(n,g,sizes,_) in grids.items():
                    terms = out['exact_terms']
                    G = exact_blocks(n,g,beta,terms) if method=='raw_density' else exact_sized(n,g['h_native'],beta,sizes,terms)
                    out['exact_case_readings'][f'{name[:3]}_N{count}'] = report(G,g,scale,unit,name==F6)
        by_catalog[method] = out
        print(method,'windows',out['catalog_size'],'feasible',out['binned_feasible'],
              'exact',out.get('exact_diagnostic'),flush=True)
    dual = certificate(all_banks[F4],constant[F4],all_banks[F6],constant[F6])
    files = (SOURCE,SF4,SF6,CALIBRATION,ROOT/'scripts'/'lire_noeuds_R354.py',
             ROOT/'scripts'/'loi_taille_R354.py',ROOT/'scripts'/'courbes_de_repartition_R352.py')
    dump(OUTPUT/'INERTIE_COMMUNE_R355.json',{'revision':355,'full':True,'physical_validation':False,
         'calibration_of_size_law':{'two_anchors':['S32 in SF4 N48','first axial F19 in SF4 N48'],
             'power':nu,'common_mass_unit_kg':u,'reference_F_support_radius_native':R[1],
             'log_size_leverage':float(ratio),'rest_cases_and_rotations_held_out':True,
             'operator_scope':'static whole-site kinetic multiplier, derivative in a time action still open'},
         'common_length_scale_native_per_angstrom':scale,'N64_is_only_kinetic_quadrature':True,
         'one_SF4_mass_anchored_units_for_controls':one_case_units,'matched_preparation_results':outcomes,
         'positive_mass_fits':by_catalog,'density_bin_dual':dual,
         'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in files}})


if __name__=='__main__':
    main()
