'Calibrate one density-squared coefficient using the S/F mass ratio only.'

from pathlib import Path

from sweep_sf4 import read
from sweep_beta import base_fields
from fields_and_references import dump
from dynamics_nodal_scripts import NodalDynamics
from trials_nodal import digest
from clock_inertia import OUTPUT as R350, REF
from inertia_and_entrainment import (BANK, M_RATIO, SF6_REF, Rotor, blocks, search, spin_free_mass)
from reader_transported import FixedReader
from kernel_topological import SOURCE as C_SOURCE, TopologicalKinetic
from weights_local import weighted_grams
from refit_dependencies import State
from free_rotations import CALIBRATION, OUTPUT as R349, ROOT
from sizes_substrate import OUTPUT, SHAPES
import numpy as np
from scipy.optimize import brentq
from scipy.stats import qmc


def checks(field,h,beta,lam):
    kernel = TopologicalKinetic(beta,lam)
    v = np.sin(np.arange(9)*.37+.27)*.025
    n = np.ascontiguousarray(field)
    t = np.zeros_like(n)
    t[:,1:] = np.cross(n[:,1:],np.array([.009,-.005,.003]))
    value,gn,gt,gv = kernel.quadratic(n,h,t,v)
    A,C,G = kernel.mass(n,h)
    mass_Q = np.sum(np.einsum('ni,nij,nj->n',t,A,t))+2*np.einsum('ni,nia,a->',t,C,v)+v@G@v
    defect = abs(value-mass_Q)/max(1.,abs(value))
    if defect>2e-11:
        raise ArithmeticError('Weighted mass and quadratic kernel disagree')
    small = np.sin(.028*np.arange(n.size)).reshape(n.shape)
    small -= n*np.sum(n*small,axis=1)[:,None]
    small /= np.linalg.norm(small)
    velocity = np.cos(.17*np.arange(n.size)).reshape(n.shape)
    velocity /= np.linalg.norm(velocity)
    dv = np.cos(np.arange(9)*.2)
    analytic = np.sum(gn*small)+np.sum(gt*velocity)+gv@dv
    rows = []
    for step in (4e-6,2e-6):
        qp = kernel.quadratic(n+step*small,h,t+step*velocity,v+step*dv)[0]
        qm = kernel.quadratic(n-step*small,h,t-step*velocity,v-step*dv)[0]
        fd = (qp-qm)/(2*step)
        rows.append({'step':step,'analytic':float(analytic),'numeric':fd,'absolute_defect':abs(fd-analytic)})
    if max(r['absolute_defect'] for r in rows)>4e-7:
        raise ArithmeticError(f'Weighted kinetic derivative failed: {rows}')
    return {'mass_quadratic_relative_defect':defect,'finite_differences':rows}


def weighted_blocks(fields,h,beta):
    native,weighted = TopologicalKinetic(beta,0.),TopologicalKinetic(beta,1.)
    g0 = np.array([native.mass(a,d)[2][3:,3:] for a,d in zip(fields,h,strict=True)])
    g1 = np.array([weighted.mass(a,d)[2][3:,3:] for a,d in zip(fields,h,strict=True)])
    return g0,g1-g0


def choose_weight(g0,extra):
    def test(lam):
        S,F = g0[:2]+lam*extra[:2]
        return spin_free_mass(S)/spin_free_mass(F)-M_RATIO
    return float(brentq(test,0.,32.,xtol=1e-12))


def compare(rotors,points,initial):
    angles,base = search(rotors[0],initial,points)
    return {'twists_rad_fitted_only_N48':angles,'n_sobol':len(points),'sampled_best_N48':base,
            'readings':[r.read(angles) for r in rotors]}


def main():
    data = read(BANK)['rows']
    candidate = read(CALIBRATION)['candidate']
    contacts = read(R349/'REPONSE_CONTACTS_R349.json')
    pr = read(R350/'CONTACT_PRECONTRAINT_R350.json')['prestressed_seed']['candidate']
    model = NodalDynamics(48)
    initial = model.unpack(model.project(model.initial())[0])[0]
    beta = candidate['beta']
    assigned = {}
    control = {}
    for label,c in zip(SHAPES,(candidate,contacts['candidate'],pr),strict=True):
        fields = base_fields(model,c,initial)
        rows = sorted((r for r in data if r['preparation']==label),key=lambda x:x['count'])
        builds = []
        for row in rows:
            h = row['h_native']
            n = fields if row['count']==48 else FixedReader(fields,row['count'],3,model).anchor
            g0,extra = weighted_blocks(n,h,beta)
            if np.max(abs(g0-sum(blocks(row))))>2e-8:
                raise ArithmeticError('Zero-density-weight action failed its baseline check')
            if label==SHAPES[0] and row['count']==48:
                raw = weighted_grams(n[0],h[0],beta)
                mismatch = float(np.max(abs(raw['topological_density_square']-extra[0])))
                if mismatch>2e-8:
                    raise ArithmeticError('Scalar and vectorized topology weights differ')
                control['native_numpy_weighted_block_inf_defect'] = mismatch
                lam = choose_weight(g0,extra)
                control['kinetic_variation_checks'] = [checks(a,d,beta,lam) for a,d in zip(n[:2],h[:2],strict=True)]
            builds.append((row,g0,extra))
            S,F = g0[:2]+lam*extra[:2]
            print(label,row['count'],'lambda',lam,'predicted mass ratio',spin_free_mass(S)/spin_free_mass(F),flush=True)
        assigned[label] = builds
    points = (2*qmc.Sobol(4,scramble=True,seed=1351).random_base2(11)-1)*np.pi
    systems = {}
    for label in SHAPES:
        ratios = []
        rotors = []
        for i,(row,g0,extra) in enumerate(assigned[label]):
            G = g0+lam*extra
            J = contacts['rows'][i]['contact_direction_jacobian'] if label==SHAPES[1] else None
            rotors.append(Rotor(row,G,row['length_scale_native_per_angstrom'],REF,J=J))
            ratios.append(spin_free_mass(G[0])/spin_free_mass(G[1]))
        result = compare(rotors,points,np.zeros(4))
        result['grid_counts'] = [r['count'] for r,_,_ in assigned[label]]
        result['predicted_local_S_over_F1_mass_ratios'] = ratios
        result['equivalent_lambda_if_isotope_ratio_refitted_by_grid'] = [choose_weight(g,extra) for _,g,extra in assigned[label]]
        systems[label] = result
        print(label,'ABC error', [float(max(abs(r['relative_errors_if_isotropic_translation']))) for r in result['readings']],flush=True)
    row6 = next(r for r in data if r['species']=='SF6')
    state6 = State('SF6')
    g0,extra = weighted_blocks(state6.fields,row6['h_native'],beta)
    r6 = np.linalg.norm(np.array(row6['centers'])[1:],axis=1)
    if np.ptp(r6)>1e-9:
        raise ArithmeticError('Reserved SF6 is not at its octahedral radius')
    ref6 = np.full(3,SF6_REF)
    h6 = r6[0]/1.5607
    reserved = {'SF6_at_measured_geometry_without_any_kinetic_refit':Rotor(row6,g0+lam*extra,h6,ref6).read(np.zeros(6)),
                'SF6_with_same_length_unit_as_SF4':Rotor(row6,g0+lam*extra,candidate['scale_native_per_angstrom'],ref6).read(np.zeros(6))}
    parents = (BANK,CALIBRATION,R349/'REPONSE_CONTACTS_R349.json',R350/'CONTACT_PRECONTRAINT_R350.json',
                Path(__file__).resolve(),C_SOURCE,ROOT/'scripts'/'noyau_topologique_R351.py')
    dump(OUTPUT/'DENSITE_TOPOLOGIQUE_ET_ROTATIONS_R351.json',{'revision':351,'full':True,
         'kinetic_multiplier':'1+lambda*det(n,D1 n,D2 n,D3 n)^2','lambda_native_length_to_the_sixth':lam,
         'mass_anchor_count':48,'mass_anchor_preparation':SHAPES[0],'isotope_mass_ratio':M_RATIO,
         'static_energy_received':True,'results':systems,'kernel_controls':control,'SF6_reserved':reserved,
         'physical_validation':False,'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in parents}})
    print('one kinetic topological coefficient',lam,flush=True)


if __name__=='__main__':
    main()
