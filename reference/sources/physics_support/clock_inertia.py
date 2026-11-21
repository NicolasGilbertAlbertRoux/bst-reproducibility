'Transport the historical clock and test absolute inertia with a mass anchor.'

from fractions import Fraction
from pathlib import Path

from sweep_sf4 import LENGTH_TARGETS, LENGTH_UNC, read
from fields_and_references import dump
from trials_nodal import digest
from free_rotations import CALIBRATION, OUTPUT as R349, REF, ROOT, kinetic21, reduced, twisted_frames
import numpy as np
from scipy.linalg import block_diag, null_space


OUTPUT = ROOT/'results'/'R350'
C_SI = 299792458.
H_SI = 6.62607015e-34
U_SI = 1.66053906892e-27
MASSES_U = (31.9720711744,18.99840316273)
V53_LENGTH_M = 2.3541395497804916e-11
V53_TIME_S = 7.293065819633371e-16
SOURCES = {
    'rotational_ABC': 'https://cccbdb.nist.gov/expgeom2x.asp?casno=7783600',
    'rotational_paper': 'https://doi.org/10.1080/0026897031000068497',
    'isotope_masses': 'https://physics.nist.gov/cgi-bin/MolSpec/mole.pl?molecule=SF&prefix=di&type=fancy',
    'reserved_nu6': 'https://doi.org/10.1016/S0022-2852(02)00054-1',
    'V53': 'BST_MASTER_PROGRESS_LIST_POST_R9_2026-09-21.md, R243, ETALONNAGES_HISTORIQUES.json de R174 cité',
}


def effective_matrix(row, twists, slow):
    X,O,G = (np.array(row[key]) for key in ('centers','frames','native_blocks'))
    frames = twisted_frames(X,O,np.asarray(twists))
    D = kinetic21(X,frames,G)
    if slow:
        J = np.array(row['contact_direction_jacobian'])
        for j,(old,new) in enumerate(zip(O[1:],frames[1:],strict=True)):
            sl = slice(3*j+3,3*j+6)
            J[j,sl] = J[j,sl]@old@new.T
        lift = block_diag(np.eye(6),null_space(J))
        D = lift.T@D@lift
    return reduced(D,6)


def absolute_rotor(D, reference_I, length_m, mass_si):
    I = np.linalg.eigvalsh(reduced(D,3))
    permutation = [3,4,5,0,1,2]
    mass_rotating_free = np.linalg.eigvalsh(reduced(D[np.ix_(permutation,permutation)],3))
    mass_rotating_held = np.linalg.eigvalsh(D[3:,3:])
    low,high = mass_rotating_free[0],mass_rotating_held[-1]
    # An optimistic interval allows any translation direction and any amount of free overall rotation.
    inertia_SI = mass_si*length_m**2*I/np.array([high,low])[:,None]
    constants = H_SI/(8*np.pi**2*(100*C_SI)*inertia_SI[::-1])
    required_mass = mass_si*length_m**2*I/reference_I
    discrepancy = np.where(REF<constants[0],constants[0]/REF-1,
                           np.where(REF>constants[1],1-constants[1]/REF,0.))
    return {'native_principal_moments':I,'translation_eigenvalues_omega_free':mass_rotating_free,
            'translation_eigenvalues_omega_held':mass_rotating_held,
            'native_mass_envelope':[low,high],
            'absolute_rotational_constants_envelope_cm_inverse':constants,
            'native_mass_required_by_ABC':required_mass,
            'minimum_relative_error_each_constant':discrepancy,
            'contains_all_measured_constants':bool(np.all(discrepancy==0)),
            'relative_span_translation_eigenvalues_with_omega_free':mass_rotating_free[-1]/mass_rotating_free[0]-1}


def negative_pivots(matrix):
    'Exact inertia certificate for a small rational symmetric matrix.'
    a = [[Fraction(x) for x in row] for row in matrix]
    negative = 0
    for j in range(len(a)):
        pivot = a[j][j]
        if not pivot:
            raise ArithmeticError('Zero pivot in an exact inertia certificate')
        negative += pivot < 0
        for i in range(j+1,len(a)):
            for k in range(i,len(a)):
                a[k][i] -= a[k][j]*a[i][j]/pivot
    return negative


def all_orientations_bound(row, minimum_length, mass_si, upper_blocks=None):
    'Use Loewner bounds and exact certificates on the saved anchor matrices.'
    X = np.array(row['centers'])
    blocks = np.array(row['native_blocks'])
    if upper_blocks is None:
        upper_blocks = blocks
    order = [3,4,5,0,1,2]
    minimum,maximum = [],[]
    for G,U in zip(blocks,upper_blocks,strict=True):
        L = G[np.ix_(order,order)]
        w = int(np.floor(100*np.linalg.eigvalsh(reduced(L,3))[0]))
        u = int(np.ceil(100*np.linalg.eigvalsh(U[3:,3:])[-1]))
        low = [[Fraction(float(G[i,j]))-(Fraction(w,100) if i==j and i>=3 else 0)
                for j in range(6)] for i in range(6)]
        high = [[(Fraction(u,100) if i==j else 0)-Fraction(float(U[i+3,j+3]))
                 for j in range(3)] for i in range(3)]
        if negative_pivots(low) or negative_pivots(high):
            raise ArithmeticError('Translation block bound failed the exact certificate')
        minimum.append(Fraction(w,100))
        maximum.append(Fraction(u,100))
    # The inertia of the optimistic scalar masses uses their own centre of mass.
    xx = [[Fraction(float(a)) for a in x] for x in X]
    c = [sum(m*x[j] for m,x in zip(minimum,xx,strict=True))/sum(minimum) for j in range(3)]
    rr = [[x[j]-c[j] for j in range(3)] for x in xx]
    I = [[sum(m*((sum(t*t for t in x) if i==j else 0)-x[i]*x[j])
              for m,x in zip(minimum,rr,strict=True)) for j in range(3)] for i in range(3)]
    floor = np.floor(100*np.linalg.eigvalsh(np.array(I,dtype=float))).astype(int)
    for n,m in enumerate(floor):
        shifted = [[I[i][j]-(Fraction(int(m),100) if i==j else 0)
                    for j in range(3)] for i in range(3)]
        if negative_pivots(shifted)!=n:
            raise ArithmeticError('Inertia eigenvalue lower bound failed the exact certificate')
    moment_lower = (floor/100)*mass_si*minimum_length**2/float(sum(maximum))
    abc_upper = H_SI/(8*np.pi**2*100*C_SI*moment_lower)
    record = {'count':row['count'],'local_translation_scalar_floors':list(map(float,minimum)),
              'local_locked_translation_scalar_caps':list(map(float,maximum)),
              'total_translation_scalar_cap':float(sum(maximum)),
              'I_native_eigenvalue_floors':floor/100,
              'absolute_ABC_upper_cm_inverse':abc_upper,
              'absolute_minimum_relative_shortfall':1.-abc_upper/REF,
              'all_three_constants_excluded':bool(np.all(abc_upper<REF)),
              'exact_rational_certificates_on_archived_float_matrices':True}
    if 'contact_direction_jacobian' in row:
        J = np.array(row['contact_direction_jacobian'])*np.sqrt(row['contact_stiffnesses'])[:,None]
        cap = max(np.linalg.eigvalsh(G[:3,:3])[-1] for G in blocks)
        record['lowest_positive_pole_lower_bound_for_any_bond_twists'] = float(np.sqrt(np.linalg.eigvalsh(J@J.T)[0]/cap))
    return record


def main():
    free = read(CALIBRATION)
    slow = read(R349/'BALAYAGE_LENT_R349.json')
    contact = read(R349/'REPONSE_CONTACTS_R349.json')
    axes = read(R349/'CALAGE_AXES_R349.json')['branches']
    bank = [r for r in read(R349/'BANQUE_ROTATION_R349.json')['rows'] if r['stage']=='anchor']
    per_angstrom = free['candidate']['scale_native_per_angstrom']
    length = 1e-10/per_angstrom
    radii = np.linalg.norm(np.array(bank[0]['centers'])[1:],axis=1)
    interval_m = np.array([max((LENGTH_TARGETS-LENGTH_UNC)/radii),min((LENGTH_TARGETS+LENGTH_UNC)/radii)])*1e-10
    if not interval_m[0]<=length<=interval_m[1]:
        raise ArithmeticError('R348 length calibration outside its own intervals')
    history = read(ROOT/'results'/'R347'/'EXTERIEUR_VIDE_R347.json')
    if history['native_small_wave_speed_long_wavelength']!=1.:
        raise ValueError('Expected the R347 native vacuum speed')
    mass_si = (MASSES_U[0]+4*MASSES_U[1])*U_SI
    I_meas = H_SI/(8*np.pi**2*(100*C_SI)*REF)
    clocks = {'R348_length_with_R347_vacuum_equal_c':length/C_SI,'V53_untransported':V53_TIME_S}
    J1 = np.array([REF[1]+REF[2],REF[0]+REF[2],REF[0]+REF[1]])
    reserved_nu6 = 729.7
    clock_rows = []
    for name,time in clocks.items():
        omega = 2*np.pi*100*C_SI*time
        row = {'candidate':name,'seconds_per_native_time':time,
               'window_R349_seconds':.04*time,
               'J1_gaps_cm_inverse_from_calibration':J1,'J1_gaps_native_angular_frequency_if_directly_identified':J1*omega,
               'reserved_nu6_cm_inverse':reserved_nu6,'reserved_nu6_native_angular_frequency':reserved_nu6*omega,
               'orientation_modes':[]}
        for data in contact['rows']:
            poles = np.sort(data['positive_angular_frequencies_native'])
            row['orientation_modes'].append({'count':data['count'],
                'poles_native_angular':poles,'poles_in_cm_inverse_under_this_clock':poles/omega,
                'max_J1_over_smallest_pole':max(J1*omega)/poles[0],
                'nu6_over_smallest_pole':reserved_nu6*omega/poles[0]})
        clock_rows.append(row)
    branches = [
        ('free_ratios',bank,free['twists_rad'],[r['direct'] for r in free['anchors']],False),
        ('free_ratios_and_axis',bank,np.deg2rad(axes['free_frames']['twists_deg']),
         [r['reading'] for r in axes['free_frames']['rows']],False),
        ('slow_ratios',contact['rows'],np.deg2rad(slow['best_relative_twists_deg']),slow['readings'],True),
        ('slow_ratios_and_axis',contact['rows'],np.deg2rad(axes['slow_contact']['twists_deg']),
         [r['reading'] for r in axes['slow_contact']['rows']],True)
    ]
    rows = []
    for name,inputs,angles,readings,constrained in branches:
        for data,prior in zip(inputs,readings,strict=True):
            D = effective_matrix(data,angles,constrained)
            r = absolute_rotor(D,I_meas,length,mass_si)
            # Taking the extrema of the geometry-compatible unit can only widen this exclusion test.
            band = r['absolute_rotational_constants_envelope_cm_inverse']
            geom_envelope = band*(length/interval_m[::-1,None])**2
            err = np.where(REF<geom_envelope[0],geom_envelope[0]/REF-1,
                           np.where(REF>geom_envelope[1],1-geom_envelope[1]/REF,0.))
            r.update({'branch':name,'count':data['count'],
                      'geometry_widened_constants_envelope_cm_inverse':geom_envelope,
                      'geometry_widened_minimum_relative_errors':err,
                      'prior_inertia_replay_inf':float(max(abs(r['native_principal_moments']-prior['eigenvalues_native'])))})
            if r['prior_inertia_replay_inf']>5e-9:
                raise ArithmeticError(f'R349 absolute inertia mismatch in {name}')
            rows.append(r)
    exclusion = []
    for prep,entries in [('R349_free_anchor_fields',bank),('R349_zero_extension_anchor_fields',contact['rows'])]:
        for row in entries:
            bound = all_orientations_bound(row,interval_m[0]*(1.-1e-10),mass_si)
            bound['anchor_fields'] = prep
            if 'lowest_positive_pole_lower_bound_for_any_bond_twists' in bound:
                pole = bound['lowest_positive_pole_lower_bound_for_any_bond_twists']
                bound['max_J1_over_pole_lower_bound'] = {
                    name:float(max(J1)*2*np.pi*100*C_SI*t/pole) for name,t in clocks.items()}
            exclusion.append(bound)
    record = {
        'full':True,'revision':350,'references':SOURCES,'NIST_calibration_ABC_cm_inverse':REF,
        'measured_I_kg_m2_as_rigid_rotor':I_meas,'formula_isotopic_mass_kg':mass_si,
        'nominal_formula_mass_u':mass_si/U_SI,'S_and_F_isotopic_mass_u':MASSES_U,
        'mass_binding_correction_omitted':True,'R348_length_m':length,'R348_geometry_length_interval_m':interval_m,
        'V53':{'historical_length_m':V53_LENGTH_M,'historical_time_s':V53_TIME_S,
               'implied_speed_m_s_if_reused_together':V53_LENGTH_M/V53_TIME_S,
               'native_length_ratio_R348_over_V53':length/V53_LENGTH_M,
               'R348_scale_with_V53_clock_m_s':length/V53_TIME_S,
               'vacuum_clock_ratio_V53_over_length_divided_by_c':V53_TIME_S/clocks['R348_length_with_R347_vacuum_equal_c']},
        'clock_candidates':clock_rows,'absolute_rotation_tests':rows,'all_orientations_bound':exclusion,
        'all_orientation_scope':'the two R349 anchor field families and their centres, any spatial orientations for the mass and inertia inequality; the pole inequality uses their specific S–F bond twists',
        'physical_validation':False,
        'mass_anchor_scope':'one scalar native mass unit from known isotopic formula mass and the translation response, no rotational constant in the calibration of mass or length',
        'interpretation_scope':'conditional test with rigid-rotor quantization; a contact pole is not assigned to a rovibrational band or identified with an observed transition',
        'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [CALIBRATION,*[R349/f'{name}.json' for name in (
            'BANQUE_ROTATION_R349','SWEEP_LENT_R349','REPONSE_CONTACTS_R349','CALAGE_AXES_R349')],
            ROOT/'results'/'R347'/'EXTERIEUR_VIDE_R347.json',Path(__file__).resolve()]}
    }
    dump(OUTPUT/'HORLOGE_ET_INERTIE_ABSOLUE_R350.json',record)
    print('R348 L interval',interval_m,'T if vacuum=c',clocks,'historical speed',V53_LENGTH_M/V53_TIME_S,flush=True)
    for row in rows:
        print(row['branch'],row['count'],'mass envelope',row['native_mass_envelope'],
              'ABC envelope',row['geometry_widened_constants_envelope_cm_inverse'],
              'min relative errors',row['geometry_widened_minimum_relative_errors'],flush=True)
    for row in exclusion:
        print('all orientations',row['anchor_fields'],row['count'],
              'ABC caps',row['absolute_ABC_upper_cm_inverse'],
              'gaps to targets',row['absolute_minimum_relative_shortfall'],flush=True)


if __name__=='__main__':
    main()
