'Keep the relaxed profile response as independent dynamical coordinates.'

from pathlib import Path

from sweep_sf4 import read
from fields_and_references import dump
from trials_nodal import digest
from reader_transported import FixedReader
from read_equilibria import DIRECT,NATIVE
from modes_radial import CORRECTED,interpolate_rates,normalized_rates
from kernel_curves import CurveKinetic
from match_contact import FREQUENCIES,TransferredContact
from relax_sf6 import OUTPUT,ROOT
from free_rotations import reduced
import numpy as np
from scipy.linalg import block_diag,eigh


RESULT = OUTPUT/'MODES_RADIAUX_ET_ROTATION_R353.json'
PAIR = np.repeat(np.eye(3),2,axis=0)/np.sqrt(2.)
SIMPLE = np.ones((3,1))/np.sqrt(3.)
CONTRAST = np.column_stack((np.array([1.,-1.,0.])/np.sqrt(2.),np.array([1.,1.,-2.])/np.sqrt(6.)))


def metric(grid,hs,rates,d,O,beta,terms):
    K = CurveKinetic(beta,terms)
    full = np.zeros((30,30))
    for i,(n,h,J,F) in enumerate(zip(grid,hs,rates,O,strict=True)):
        A,C,G = K.mass(n,h)
        V = np.zeros((9,30))
        V[3:6,9+3*i:12+3*i] = np.eye(3)
        V[6:,6:9] = F.T
        if i:
            V[6:,:3] = np.outer(F.T@d[i-1],PAIR[i-1])
        full += V.T@G@V
        q = J.reshape(-1,3)
        full[3:6,3:6] += q.T@(A@J).reshape(-1,3)
        cross = (q.T@C.reshape(-1,9))@V
        full[3:6] += cross
        full[:,3:6] += cross.T
    return reduced(full,6)


def modes(K,M,B):
    P = block_diag(B,B)
    s,X = eigh(P.T@K@P,P.T@M@P)
    if min(s)<=0.:
        raise ValueError('The projected oscillations are not positive')
    strength = np.sum(X[:B.shape[1],:]**2,axis=0)
    strength /= strength.sum()
    return {'squared_native_rates':s,'relative_radial_force_residues':strength}


def run():
    model = TransferredContact()
    prior = read(RESULT)
    final = read(CORRECTED)['final']
    model.m.evaluate(np.array(final['coordinates']))
    n = np.array(model.m.state.fields)
    norms = np.linalg.norm(n,axis=2)
    fields = n/norms[:,:,None]
    Jr = np.array(prior['quasistatic_chart_velocity_per_radius'])
    rates = normalized_rates(model.m.state,Jr@PAIR,fields,norms)
    fine = FixedReader(fields,64,3,None)
    rates64 = interpolate_rates(fine,rates)
    d = np.array(final['directions'])
    O = np.concatenate((np.eye(3)[None],model.m.frames))
    h = np.array([model.m.h(i) for i in range(7)])
    K0 = PAIR.T@np.array(prior['fixed_field_radial_Hessian'])@PAIR
    K1 = PAIR.T@np.array(prior['adiabatic_field_radial_Hessian'])@PAIR
    D = K0-K1
    K = np.block([[K0,-D],[-D,D]])
    F = np.vstack((np.eye(3),np.eye(3)))
    laws = {'direct_R352':read(DIRECT)['fitted_curve'],'native_R352':read(NATIVE)['native_terms']}
    result = {}
    for count,grid,hs,J in ((48,fields,h,rates),(64,fine.anchor,h*47/63,rates64)):
        for law,terms in laws.items():
            key = f'N{count}_{law}'
            M = metric(grid,hs,J,d,O,model.common['beta'],terms)
            a = prior['radial_modes'][key]
            fast_defect = float(np.max(abs(M[:3,:3]-a['fixed_internal_fields']['pair_even_radial_mass'])))
            slow_defect = float(np.max(abs(F.T@M@F-a['quasistatic_internal_fields']['pair_even_radial_mass'])))
            if max(fast_defect,slow_defect)>2e-7:
                raise ArithmeticError(f'Mixed metric lost a limit: {fast_defect}, {slow_defect}')
            a1g,eg = modes(K,M,SIMPLE),modes(K,M,CONTRAST)
            ia = int(np.argmax(a1g['relative_radial_force_residues']))
            ie = np.argsort(eg['relative_radial_force_residues'])[-2:]
            ae = np.sqrt(a1g['squared_native_rates'][ia])
            all_eg = np.sqrt(eg['squared_native_rates'])/ae
            omega = FREQUENCIES['A1g']*all_eg
            result[key] = {'metric':M,'fast_mass_replay_defect':fast_defect,'slow_mass_replay_defect':slow_defect,
                'A1g_modes':a1g,'Eg_modes':eg,'A1g_anchor_index_by_largest_radial_force_residue':ia,
                'Eg_two_highest_radial_force_residue_indices':ie,
                'Eg_poles_cm_inverse_on_same_A1g_anchor':omega,
                'Eg_strongest_radial_pair_cm_inverse':np.sort(omega[ie])}
            print(key,'Eg radial poles',omega,'weights',eg['relative_radial_force_residues'],flush=True)
    files = (Path(__file__).resolve(),RESULT,CORRECTED,DIRECT,NATIVE,ROOT/'scripts'/'modes_radiaux_R353.py')
    dump(OUTPUT/'MODES_MIXTES_R353.json',{'revision':353,'full':True,'physical_validation':False,
        'scope':'Galerkin dynamics of the even radial displacements and their six-dimensional static field response before projection',
        'no_new_frequency_fit_except_the_symmetric_display_anchor':True,
        'radial_force_residues_are_not_experimental_spectral_intensities':True,
        'full_projected_stiffness':K,'cases':result,'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in files}})


if __name__=='__main__':
    run()
