'Directly integrate the selected nonlinear windows, then freeze their amplitudes.'

from pathlib import Path

from sweep_sf4 import read
from sweep_beta import base_fields
from collection_curves import CASE,OUTPUT,PAIRS,integrand,matrices,shape_inputs
from calibrate_curves import Bank,MAIN4,MAIN6,M4,REF,isotropic
from fields_and_references import dump
from curves_of_distribution import predict,shape,solve
from dynamics_nodal_scripts import NodalDynamics
from trials_nodal import digest
from inertia_and_entrainment import Rotor,rotor_abc,search
from quadrature_topological import samples
from free_rotations import OUTPUT as R349,ROOT
import numpy as np
from scipy.stats import qmc


FITS = OUTPUT/'FENETRES_DE_REPARTITION_R352.json'
SELECTED = 'only_density_windows'


def matrices_for(field,h,beta,scheme,terms):
    G = np.zeros((len(terms)+1,len(PAIRS)))
    for x,y,v,b,tr,w,_ in integrand(field,h,scheme):
        vector = np.array([w,*[w*shape(x,y,k['center'],k['width_log'],k['chi_power']) for k in terms]])
        row = np.column_stack([(1.+beta*tr)*np.einsum('ni,ni->n',v[:,i],v[:,j])-
                               beta*np.einsum('ni,ni->n',b[:,i],b[:,j]) for i,j in PAIRS])
        G += vector@row
    return matrices(G)


def quadrature(fields,spacing,beta,scheme,terms):
    G = np.array([matrices_for(f,h,beta,scheme,terms) for f,h in zip(fields,spacing,strict=True)])
    return G[:,0],G[:,1:].swapaxes(0,1)


def native_inputs():
    data = read(FITS)['families']
    seed = data[SELECTED]
    terms = seed['terms']
    a0 = np.array([r['amplitude'] for r in terms])
    if len(a0)!=3 or any(t['chi_power']!=0 for t in terms):
        raise ValueError('The selected three density windows were expected')
    banks = {tag:Bank(tag) for tag in CASE}
    model = NodalDynamics(48)
    return seed,terms,a0,banks,model


def slow_control(model,terms,a,scale,munit):
    rpath = R349/'REPONSE_CONTACTS_R349.json'
    response = read(rpath)
    candidate = response['candidate']
    initial = model.unpack(model.project(model.initial())[0])[0]
    n = base_fields(model,candidate,initial)
    g = response['rows'][0]
    held = []
    for gauss in (2,3):
        u,p = quadrature(n,model.h,candidate['beta'],samples(gauss,3),terms)
        rotor = Rotor(g,u+np.tensordot(a,p,axes=1),scale,REF,J=g['contact_direction_jacobian'])
        if gauss==2:
            pts = (2*qmc.Sobol(4,scramble=True,seed=2352).random_base2(10)-1)*np.pi
            theta,_ = search(rotor,np.zeros(4),pts)
        result = rotor.read(theta)
        I = np.asarray(result['rotation_moments_native'])*munit*(1e-10/scale)**2
        H,C = 6.62607015e-34,299792458.
        abc = H/(8*np.pi**2*(100*C)*I)
        held.append({'gauss':gauss,'ABC_with_common_units_cm_inverse':abc,
                     'relative_errors':abc/REF-1.,'isotropic_translation_component_relative_to_free_formula_mass':
                         np.trace(result['mass_translation_native'])*munit/(3.*M4)-1.})
    return held


def main():
    seed,terms,a0,banks,model = native_inputs()
    entries = {}
    for name in CASE:
        tag,gauss,order = CASE[name]
        n,g,beta = shape_inputs(model,tag)
        const,windows = quadrature(n,g['h_native'],beta,samples(gauss,order),terms)
        baseline_error = float(np.max(abs(const-banks[name].const)))
        if baseline_error>2e-8:
            raise ArithmeticError(f'Exact quadrature lost the native reference {name}, {baseline_error}')
        entries[name] = {'baseline_gram':const,'windows':windows,'native_baseline_replay_defect':baseline_error}
        print('direct integration',name,'reference defect',baseline_error,flush=True)
    a4,a6 = entries[MAIN4],entries[MAIN6]
    a,test = solve(a4['baseline_gram'],a6['baseline_gram'],a4['windows'],a6['windows'],np.ones(len(terms)))
    if a is None:
        raise ArithmeticError(f'Direct integration changed feasibility of the three windows: {test}')
    fitted = [{**term,'amplitude':float(z)} for term,z in zip(terms,a,strict=True)]
    grams = {name:r['baseline_gram']+np.tensordot(a,r['windows'],axes=1) for name,r in entries.items()}
    status = predict(banks,grams)
    h,m = status['common_length_unit_m'],status['common_mass_unit_kg']
    slow = slow_control(model,fitted,a,1e-10/h,m)
    record = {'revision':352,'full':True,'physical_validation':False,'fitted_curve':fitted,
              'initial_binned_amplitudes':a0,'three_mass_calibration_equations':test,
              'direct_matrix_bases':entries,'calibration_and_holdouts':status,
              'slow_SF4_contact_null_at_same_units':slow,'SF6_rotation_was_reserved':True,
              'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (
                  FITS,Path(__file__).resolve(),ROOT/'scripts'/'banque_courbes_R352.py',
                  R349/'REPONSE_CONTACTS_R349.json',*(b.path for b in banks.values()))}}
    dump(OUTPUT/'CALAGE_DIRECT_R352.json',record)
    print('direct terms',fitted,'relative equations',test,flush=True)
    for name,v in status['cases'].items():
        print(name,'ABC relative errors',v['rotation_relative_errors_same_units'],
              'm error',v['mass_relative_error_with_same_unit'],'anisotropy',v['translation_anisotropy'],flush=True)
    print('slow control',slow,flush=True)


if __name__=='__main__':
    main()
