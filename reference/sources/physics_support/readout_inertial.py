'Compare the geometric centres and the positions read from the kinetic action.'

import argparse
import json
from pathlib import Path

from fields_and_references import dump
from dynamics_nodal_scripts import NodalDynamics, ROOT
from trials_nodal import digest
from refit_dependencies import skew
from rotation_prepared import OUTPUT
import numpy as np


REFERENCE = {
    'geometry':{'paper':'Tolles and Gwinn (1962)', 'doi':'10.1063/1.1732702',
                'nist':'https://cccbdb.nist.gov/exp2x.asp?casno=7783600',
                'inherited_R336_geometry':True,
                'r_ax_angstrom':1.646,'r_eq_angstrom':1.545,'listed_radius_uncertainty_angstrom':.003,
                'angles_deg':{'ax_ax':173.07,'eq_eq':101.55,'ax_eq':87.81},
                'listed_angle_uncertainty_deg':.5},
    'rotation':{'paper':'Raffael, Smith and Newnham (2003)', 'doi':'10.1080/0026897031000068497',
                'nist':'https://cccbdb.nist.gov/expgeom2x.asp?casno=7783600',
                'isotopologue':'32SF4', 'technique':'FTIR rovibrational spectroscopy',
                'abc_cm_inverse':[.22308,.13632,.10741],
                'uncertainties_in_NIST_table':None,
                'prior_use':'R279 compared B/A for the R262 rotor. Separate measurement from the 1962 geometry.'}}


def kinetic_offset(K, M):
    normalized = np.linalg.solve(M, K.T).T
    antisymmetric = (normalized-normalized.T)/2
    c = antisymmetric[[2,0,1],[1,2,0]]
    return c, normalized-antisymmetric


def lengths_and_angles(centers):
    d = centers[1:]-centers[0]
    lengths = np.linalg.norm(d, axis=1)
    dirs = d/lengths[:,None]
    angles = np.degrees(np.arccos(np.clip(dirs@dirs.T,-1,1)))
    geom = REFERENCE['geometry']
    target = np.array([geom['r_ax_angstrom']]*2+[geom['r_eq_angstrom']]*2)
    radius_uncertainty = geom['listed_radius_uncertainty_angstrom']
    low, high = (target-radius_uncertainty)/lengths, (target+radius_uncertainty)/lengths
    ratios = lengths[:2,None]/lengths[None,2:]
    # Diagnostic bound for every common conversion factor. No conversion is adopted.
    normalized = lengths/target
    umin, umax = min(normalized), max(normalized)
    i, j = np.triu_indices(4,1)
    a = geom['angles_deg']
    expected = np.array([a['ax_ax'], a['ax_eq'], a['ax_eq'], a['ax_eq'], a['ax_eq'], a['eq_eq']])
    return {'coordinates_native':centers, 'lengths_native':lengths,
            'ordered_radius_labels':['Ax1','Ax2','Eq1','Eq2'], 'four_axial_equatorial_ratios':ratios,
            'reference_axial_equatorial_ratio_interval':[(target[0]-radius_uncertainty)/(target[2]+radius_uncertainty),
                                                         (target[0]+radius_uncertainty)/(target[2]-radius_uncertainty)],
            'admissible_common_scale_interval_per_site_angstrom_per_native':np.column_stack((low,high)),
            'common_scale_interval_overlap':bool(max(low)<=min(high)),
            'common_scale_gap_angstrom_per_native':float(max(low)-min(high)),
            'minimum_possible_max_relative_length_error_over_common_scale':float((umax-umin)/(umax+umin)),
            'six_pairs':[f'{u+1}-{v+1}' for u,v in zip(i,j,strict=True)],
            'six_angles_deg':angles[i,j], 'six_reference_angles_deg':expected,
            'six_angle_errors_deg':angles[i,j]-expected}


def read(model, state, time, source):
    n, _, x, _ = model.unpack(state)
    centers, _, _, frames, _, embeddings = model.geometry(x)
    offsets, controls = [], []
    global_mass = np.zeros((6,6))
    shift = np.array([.13,-.23,.17])
    shift_map = np.eye(6)
    shift_map[3:,:3] = skew(shift)
    for field, h, E in zip(n,model.h,embeddings,strict=True):
        _, _, G = model.kernel.mass(field,h)
        M, K = G[6:,6:], G[3:6,6:]
        center, irreducible = kinetic_offset(K,M)
        recharted = shift_map.T @ G[3:,3:] @ shift_map
        c_recharted, s_recharted = kinetic_offset(recharted[:3,3:],recharted[3:,3:])
        covariance_defect = max(np.max(abs(center-c_recharted-shift)), np.max(abs(irreducible-s_recharted)))
        if covariance_defect > 1e-11:
            raise ArithmeticError('The inertial position failed the change of origin test')
        offsets.append(center)
        controls.append({'offset_in_local_frame_native':center,
                         'irreducible_normalized_rotation_translation_coupling_native':irreducible,
                         'irreducible_coupling_norm_native':float(np.linalg.norm(irreducible)),
                         'origin_covariance_inf':float(covariance_defect)})
        rigid = E[:,:6]
        global_mass += rigid.T @ G @ rigid
    world_offsets = np.einsum('nij,nj->ni',frames,np.array(offsets))
    I = global_mass[:3,:3]-global_mass[:3,3:]@np.linalg.solve(global_mass[3:,3:],global_mass[3:,:3])
    inertias = np.linalg.eigvalsh((I+I.T)/2)
    if inertias[0] <= 0:
        raise ArithmeticError('The frozen-shape rotor inertia is not positive')
    rotation_ratios = inertias[0]/inertias[1:]
    abc = np.array(REFERENCE['rotation']['abc_cm_inverse'])
    return {'count':model.count, 'time_native':time,
            'source':str(source.relative_to(ROOT)), 'source_sha256':digest(source),
            'direct_centers':lengths_and_angles(centers),
            'kinetic_centers':lengths_and_angles(centers+world_offsets), 'kinetic_origin_controls':controls,
            'frozen_shapes_native_rotor':{'principal_inertias':inertias, 'B_over_A_C_over_A':rotation_ratios,
                'reference_ratios':abc[1:]/abc[0], 'relative_errors':rotation_ratios/(abc[1:]/abc[0])-1}}


def run(stage):
    rows = []
    for count in (48,64):
        model = NodalDynamics(count)
        if stage=='anchor':
            seed = ROOT/'results'/'INITIAL_NODAL_R345.npz'
            state, _ = model.project(model.initial())
            time = 0.
        else:
            seed = OUTPUT/f'N{count}_DT005.npz'
            with np.load(seed) as archive:
                meta = json.loads(str(archive['metadata']))
                if not meta['full']:
                    raise ValueError('Read the final accepted state only after its run completes')
                state = archive['state']
            time = meta['history'][-1]['time_native']
        row = read(model,state,time,seed)
        rows.append(row)
        print('observables',count,time,'direct lower bound',
              row['direct_centers']['minimum_possible_max_relative_length_error_over_common_scale'],
              'kinetic lower bound',row['kinetic_centers']['minimum_possible_max_relative_length_error_over_common_scale'],
              'rotor relative errors',row['frozen_shapes_native_rotor']['relative_errors'],flush=True)
    dump(OUTPUT/f'LECTURES_{stage.upper()}_R347.json',{'reference':REFERENCE,'rows':rows,
         'reader_choices_frozen_at_anchor':True,'position_kinetic_definition':'antisymmetric part of K M^-1',
         'frozen_rotor':'simultaneous spatial rotation, remove global translation at P=0, hold shapes and relative poses',
         'conditional_physical_comparison':True,'physical_validation':False,
         'source_sha256':digest(Path(__file__).resolve())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('anchor','terminal'))
    options = parser.parse_args()
    run(options.stage)
