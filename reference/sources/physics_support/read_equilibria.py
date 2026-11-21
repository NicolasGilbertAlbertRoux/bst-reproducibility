'Read the relaxed fields with the frozen R352 inertia and diagnose contact intervals.'

import argparse
from pathlib import Path

from sweep_sf4 import read
from calibrate_curves import GIVEN_R6,isotropic
from fields_and_references import dump
from trials_nodal import digest
from integrate_curves import quadrature
from reader_transported import FixedReader
from ports_and_cases import radial_interval
from quadrature_topological import samples
from relax_sf6 import OUTPUT,R352,RADIAL,ROOT,SF6
from connect_curves_kernel import exact_blocks,read_rotor
import numpy as np


SOURCE = Path(__file__).resolve()
INPUTS = {'fields':'FIELDS_FIXES_ANGLES','all':'FIELDS_ET_ANGLES_LIBRES'}
DIRECT = R352/'CALAGE_DIRECT_R352.json'
NATIVE = R352/'NOYAU_ET_TRANSFERT_R352.json'


def slope_interval(rows):
    low,high = -np.inf,np.inf
    witnesses = [None,None]
    for i,(x,lo,hi) in enumerate(rows):
        for j,(y,l,h) in enumerate(rows):
            d = y-x
            if d<=1e-12:
                continue
            a,b = (l-hi)/d,(h-lo)/d
            if a>low:
                low,witnesses[0] = a,(i+1,j+1)
            if b<high:
                high,witnesses[1] = b,(i+1,j+1)
    return {'slope_interval':[None if not np.isfinite(v) else float(v) for v in (low,high)],
            'witness_pairs':witnesses,'feasible':bool(low<=high)}


def interval_test(model,record):
    target = read(RADIAL)
    scale = model.common['scale_native_per_angstrom']
    radius = GIVEN_R6*scale
    dr = target['radius_window_angstrom_plus_minus']*scale
    p6 = np.array(record['reference_b_native'])-model.geometry.d['r0']
    k = record['kappa_native']
    b6 = [radial_interval(model.well,stiff,radius-dr,radius+dr) for stiff in k]
    offsets = [val-p-model.geometry.d['r0']+model.offset for val,p in zip(b6,p6,strict=True)]
    x = max(r[0] for r in offsets)
    y = min(r[1] for r in offsets)
    rows4 = [(r['received_port_sum_native'],*r['allowed_contact_reference_curve_value'])
             for r in target['ports'] if r['molecule']=='SF4']
    rows6 = [(p,*r) for p,r in zip(p6,b6,strict=True)]
    a,b = slope_interval(rows4),slope_interval(rows6)
    if not all(isinstance(v,float) for r in (a,b) for v in r['slope_interval']):
        raise ArithmeticError('Degenerate port interval')
    lo,hi = max(a['slope_interval'][0],b['slope_interval'][0]),min(a['slope_interval'][1],b['slope_interval'][1])
    return {'reference_radius_angstrom':GIVEN_R6,'diagnostic_half_window_angstrom':dr/scale,
            'site_offsets_absolute':offsets,'one_SF6_additive_offset_intersection':[x,y],
            'one_SF6_additive_offset_exists':bool(x<=y),'conditioned_port_rows_SF6':rows6,
            'SF4_retained_preparation_slope_interval':a,'SF6_relaxed_preparation_slope_interval':b,
            'same_affine_slope_separate_intercepts_interval':[lo,hi],
            'same_affine_slope_separate_intercepts_feasible':bool(lo<=hi),
            'scope':'actual port and curvature at frozen geometries and fields; SF6 length window is a diagnostic, not an uncertainty'}


def read_states(G,frames,dirs,anchor_dirs,r,scale,unit):
    rows = {}
    for name,D,radii in (('relaxed_geometry',dirs,r),
                          ('reference_length_only',dirs,np.full(6,GIVEN_R6*scale)),
                          ('reference_length_and_anchor_directions',anchor_dirs,np.full(6,GIVEN_R6*scale))):
        row = {'centers':np.vstack((np.zeros(3),D*radii[:,None])),'frames':frames}
        z = read_rotor(G,row,scale,unit,True,np.zeros(6),radii)
        z.pop('SF6_radial_replay_geometry_applied')
        z['uses_reference_lengths'] = name!='relaxed_geometry'
        rows[name] = z
    local = isotropic(G)
    return {'reads':rows,'local_isotropic_components_native':local,
            'sulfur_share_of_isotropic_total':float(local[0]/local.sum()),
            'F_site_mass_spread_relative':float(np.ptp(local[1:])/np.mean(local[1:]))}


def main(case,cubic):
    file = OUTPUT/f'{INPUTS[case]}_R353.json'
    fit = read(file)['final']
    model = SF6()
    coordinates = np.array(fit['coordinates'])
    model.m.evaluate(coordinates)
    raw = np.array(model.m.state.fields)
    norms = np.linalg.norm(raw,axis=2)
    fields = raw/norms[:,:,None]
    h = np.array([model.m.h(i) for i in range(len(fields))])
    frames = np.concatenate((np.eye(3)[None],model.m.frames))
    dirs = np.array(fit['directions'])
    _,_ = model.m.evaluate(model.anchor)
    anchor_dirs = model.m.last['D'].copy()
    r = np.array(fit['radii_native'])
    a,b = read(DIRECT),read(NATIVE)
    scale = model.common['scale_native_per_angstrom']
    if abs(b['native_scale_per_angstrom']-scale)>1e-12:
        raise ValueError('The common unit changed')
    laws = {'direct_R352':(a['fitted_curve'],a['calibration_and_holdouts']['common_mass_unit_kg']),
            'native_R352':(b['native_terms'],b['native_common_mass_unit_kg'])}
    rows = {}
    for count,grid,hs in ((48,fields,h),(64,FixedReader(fields,64,3,None).anchor,h*(47/63))):
        geometry = {'h_native':hs}
        for name,(terms,unit) in laws.items():
            G = exact_blocks(grid,geometry,model.common['beta'],terms)
            key = f'N{count}_{name}'
            rows[key] = read_states(G,frames,dirs,anchor_dirs,r,scale,unit)
            result = rows[key]['reads']['relaxed_geometry']
            print(case,key,'mass error',result['total_mass_error_with_common_unit'],
                  'actual ABC',result['rotation_constants_cm_inverse'],flush=True)
    if cubic:
        terms,unit = laws['direct_R352']
        base,extra = quadrature(fields,h,model.common['beta'],samples(2,3),terms)
        G = base+np.tensordot([k['amplitude'] for k in terms],extra,axes=1)
        rows['cubic_Gauss2_direct_R352'] = read_states(G,frames,dirs,anchor_dirs,r,scale,unit)
    target = model.geometry.target
    theta = np.degrees(np.arctan2(np.linalg.norm(np.cross(dirs[:,None],dirs[None,:]),axis=2),dirs@dirs.T))
    aerr = theta-np.degrees(np.arccos(np.clip(target,-1.,1.)))
    result = {'revision':353,'full':True,'physical_validation':False,'case':case,'fields_from_relaxation':True,
              'no_inertia_parameter_refit':True,'scale_native_per_angstrom':scale,'radii_angstrom':r/scale,
              'angle_max_error_to_inherited_gram_deg':float(np.max(abs(aerr))),
              'angle_errors_deg':aerr,'field_normalization_inf_defect':float(np.max(abs(norms-1.))),
              'readings':rows,'contact_inverse_diagnostic':interval_test(model,fit),
              'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (SOURCE,file,DIRECT,NATIVE,RADIAL)}}
    dump(OUTPUT/f'LECTURE_{case.upper()}_R353.json',result)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('case',choices=INPUTS)
    p.add_argument('--cubic',action='store_true')
    a = p.parse_args()
    main(a.case,a.cubic)
