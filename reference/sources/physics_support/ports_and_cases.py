'Test whether one monotone curve of the received radial port spans both cases.'

from pathlib import Path

from sweep_sf4 import LENGTH_TARGETS,LENGTH_UNC,read
from collection_curves import OUTPUT,ROOT
from calibrate_curves import GIVEN_R6,M64
from fields_and_references import dump
from contacts_reactive import ReactiveContacts
from test_fit import AdjustedFrames
from trials_nodal import digest
from clock_inertia import OUTPUT as R350
from precision_sum import AccurateSumAction
from prepare_contact_prestressed import segments
from refit_dependencies import install
from free_rotations import CALIBRATION
from sizes_substrate import OUTPUT as R351
import numpy as np
from scipy.optimize import brentq


def radial_interval(well,kappa,lo,hi):
    s = well.radius_sum
    all_curves = []
    for a,b,U in segments(well):
        l,r = max(a,lo/s),min(b,hi/s)
        if l>=r:
            continue
        # U is in the local fraction offset. The inverse target b=r+W'/k.
        bfun = U.deriv()/(s*kappa)
        bfun.coef[0] += s*a
        bfun.coef[1] += s
        stationary = [float(x.real) for x in bfun.deriv().roots() if abs(x.imag)<1e-11 and l-a<x.real<r-a]
        values = bfun([l-a,r-a,*stationary])
        all_curves.extend(values)
    if not all_curves:
        raise ArithmeticError('The radius window left the received well')
    return np.array([min(all_curves),max(all_curves)])


def contacts(molecule,raw,targets,well,common_scale,r0):
    rows = []
    for i,(contact,target) in enumerate(zip(raw,targets,strict=True)):
        kappa = contact['kappa']
        shift = contact['sulfur_port']+contact['F_port']
        radius = target*common_scale
        dr = LENGTH_UNC*common_scale
        expected = radial_interval(well,kappa,radius-dr,radius+dr)
        rows.append({'molecule':molecule,'site':i+1,'received_port_sum_native':shift,
                     'kappa_native':kappa,'target_radius_window_native':[radius-dr,radius+dr],
                     'allowed_contact_reference_curve_value':expected,
                     'target_central_radius':radius,
                     'force_at_target_native':well.separation_energy_derivative(radius)[1]+kappa*(radius-r0-shift)})
    return rows


def central_mass():
    file1 = R351/'ENTRAINEMENT_ET_ROTATIONS_R351.json'
    file2 = R351/'TRANSFERT_ET_ECHELLES_R351.json'
    load = read(file1)['results_SF4']['free']['entrainment_center']['readings_after_torsion_fit'][0]
    unweighted = read(file2)['SF6_same_kinetic_parameters_no_torsion_fit']['original']
    m4,m6 = [np.trace(np.array(r['mass_translation_native']))/3. for r in (load,unweighted)]
    return {'R351_calibrated_SF4_native_mass':m4,'original_SF6_native_mass':m6,'target_SF6_with_same_unit':M64*m4,
        'central_correction_needed_if_other_blocks_are_fixed':M64*m4-m6,
        'most_favorable_required_correction_with_directional_masses':
            M64*max(load['mass_translation_eigenvalues_native'])-min(unweighted['mass_translation_eigenvalues_native']),
        'scope':'R351 shapes, chosen frames, and positive central-only inertial loading'}


def main():
    candidate = read(CALIBRATION)['candidate']
    model4 = AdjustedFrames(48,candidate)
    try:
        n,_,x,_ = model4.unpack(model4.adjusted_seed)
        model4.potential(n,x)
        m = model4.received.c.model
        r0 = float(m.m.d['r0'])
        scale = candidate['scale_native_per_angstrom']
        well4 = m.m.trial.well
        rows = contacts('SF4',m.last['contacts'],LENGTH_TARGETS,well4,scale,r0)
    finally:
        model4.close()
    model6 = ReactiveContacts('SF6','current','mobile')
    install(model6,candidate['beta'],candidate['kangle'],candidate['gamma'])
    model6.action.action = AccurateSumAction(candidate['beta'])
    model6.evaluate(model6.state.x)
    if float(model6.m.d['r0'])!=r0:
        raise ValueError('The received contact reference differs between cases')
    well6 = model6.m.trial.well
    rows += contacts('SF6',model6.last['contacts'],np.full(6,GIVEN_R6),well6,scale,r0)
    ordered = sorted(rows,key=lambda r:r['received_port_sum_native'])
    violations = {'increasing':[],'decreasing':[]}
    for i,a in enumerate(ordered):
        for b in ordered[i+1:]:
            pa,pb = [r['received_port_sum_native'] for r in (a,b)]
            if pb-pa<=1e-10:
                continue
            the,ha = a['allowed_contact_reference_curve_value']
            lb,hb = b['allowed_contact_reference_curve_value']
            labels = [(r['molecule'],r['site']) for r in (a,b)]
            if the>hb:
                violations['increasing'].append({'sites':labels,'native_gap':the-hb,'x_difference':pb-pa})
            if lb>ha:
                violations['decreasing'].append({'sites':labels,'native_gap':lb-ha,'x_difference':pb-pa})
    shifts = {}
    for name,well in (('SF4',well4),('SF6',well6)):
        group = [r for r in rows if r['molecule']==name]
        each = [r['allowed_contact_reference_curve_value']-r['received_port_sum_native']-r0 for r in group]
        low,high = max(x[0] for x in each),min(x[1] for x in each)
        shifts[name] = {'common_additive_offset_interval':[low,high],'has_intersection':bool(low<=high)}
        if low<=high:
            offset = 0. if low<=0.<=high else .5*(low+high)
            solved = []
            for r in group:
                b = r0+r['received_port_sum_native']+offset
                k = r['kappa_native']
                def residual(rad):
                    return well.separation_energy_derivative(rad)[1]+k*(rad-b)
                native = brentq(residual,*r['target_radius_window_native'],xtol=1e-13)
                eps = 1e-5
                solved.append({'site':r['site'],'radius_native':native,'radius_angstrom':native/scale,
                    'force_defect':abs(residual(native)),
                    'conditional_radial_curvature':(residual(native+eps)-residual(native-eps))/(2*eps)})
            shifts[name].update({'applied_offset_native':offset,'radial_replay':solved})
    files = (CALIBRATION,R351/'ENTRAINEMENT_ET_ROTATIONS_R351.json',R351/'TRANSFERT_ET_ECHELLES_R351.json',
             ROOT/'sources'/'R321'/'PUITS_SF_R319.npz',Path(__file__).resolve())
    record = {'revision':352,'full':True,'physical_validation':False,
              'central_only_mass_test':central_mass(),'ports':rows,'violations':violations,
              'relative_offset_after_conditioning_on_molecule':shifts,
              'two_contexts_only_for_offset_curve':True,
              'common_scale_native_per_angstrom':scale,'radius_window_angstrom_plus_minus':LENGTH_UNC,
              'SF6_window_note':'diagnostic tolerance set equal to the SF4 comparison window; no claim of measured SF6 uncertainty',
              'scope':'held fields, angular directions, contact curvature law and radial well. The constraints are radial only.',
              'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in files}}
    dump(OUTPUT/'PORTS_MULTICAS_R352.json',record)
    print('monotone port ordering violations',[len(x) for x in violations.values()],
          'offsets',shifts,'central mass',record['central_only_mass_test'],flush=True)


if __name__=='__main__':
    main()
