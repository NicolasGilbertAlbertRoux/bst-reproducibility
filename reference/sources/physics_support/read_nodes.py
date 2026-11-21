'Use the unchanged R352 inertia and measure the centres of the relaxed fields.'

import argparse
import itertools
from pathlib import Path

from sweep_sf4 import read
from fields_and_references import dump
from trials_nodal import digest
from reader_transported import FixedReader
from read_equilibria import DIRECT,NATIVE,read_states
from relax_nodes import ANCHOR,OUTPUT,ROOT,NodalSF6
from connect_curves_kernel import exact_blocks
from sizes_substrate import nodes,stencils
import numpy as np


def mass_laws():
    a,b = read(DIRECT),read(NATIVE)
    return {'direct_R352':(a['fitted_curve'],a['calibration_and_holdouts']['common_mass_unit_kg']),
            'native_R352':(b['native_terms'],b['native_common_mass_unit_kg'])}


def charge(field,n,h):
    positions = nodes(n,h)
    f = field.reshape(n,n,n,4)
    total = 0.
    absolute = 0.
    moment = np.zeros(3)
    signs = itertools.product((1,-1),repeat=3)
    for sign,(d,x) in zip(signs,stencils(field,h,positions),strict=True):
        axes = tuple(j for j,s in enumerate(sign) if s<0)
        z = np.flip(f,axis=axes) if axes else f
        n0 = z[:-1,:-1,:-1].reshape(-1,4)
        mat = np.concatenate((n0[:,None],d),axis=1)
        rho = np.linalg.det(mat)*(h**3/(16.*np.pi**2))
        total += float(np.sum(rho))
        absolute += float(np.sum(abs(rho)))
        moment += rho@x
    if abs(total)<.1:
        raise ArithmeticError('The signed topological centroid is ill-conditioned')
    return {'signed_charge_quadrature':total,'absolute_charge_quadrature':absolute,
            'centroid_native':moment/total}


def node_geometry(n,raw,hs,r,d,frames,scale):
    m = round(n.shape[1]**(1/3))
    after = [charge(z,m,h) for z,h in zip(n,hs,strict=True)]
    before = [charge(z,m,h) for z,h in zip(raw,hs,strict=True)]
    a,b = [np.array([row['centroid_native'] for row in rows]) for rows in (before,after)]
    origin = d*r[:,None]
    corrected = origin+np.einsum('nij,nj->ni',frames,b[1:])-b[0]
    return {'before_R353':before,'after':after,'centroid_changes_local_native':b-a,
            'max_centroid_shift_angstrom':float(np.max(np.linalg.norm(b-a,axis=1))/scale),
            'nominal_contact_lengths_angstrom':r/scale,
            'topological_centroid_lengths_angstrom':np.linalg.norm(corrected,axis=1)/scale,
            'scope':'signed gnomonic topological quadrature on the original N48 grid; auxiliary geometry reader'}


def main(label):
    f = OUTPUT/f'ETAT_{label}_R354.npz'
    with np.load(f) as x:
        n,r,d,frames,hs = [x[key] for key in ('fields','radii_native','directions','frames','h_native')]
    model = NodalSF6(workers=5)
    try:
        beta = model.model.common['beta']
        scale = model.model.common['scale_native_per_angstrom']
        centroids = node_geometry(n,model.n0,hs,r,d,frames,scale)
        O = np.concatenate((np.eye(3)[None],frames))
        fine = FixedReader(n,64,3,None)
        rows = {}
        for count,grid,spacing in ((48,n,hs),(64,fine.anchor,hs*(47/63))):
            for key,(terms,unit) in mass_laws().items():
                G = exact_blocks(grid,{'h_native':spacing},beta,terms)
                value = read_states(G,O,d,model.model.ideal_directions,r,scale,unit)
                rows[f'N{count}_{key}'] = value
                print(label,count,key,value['reads']['relaxed_geometry']['rotation_errors_relative'],
                      value['reads']['relaxed_geometry']['total_mass_error_with_common_unit'],flush=True)
        files = (Path(__file__).resolve(),f,ANCHOR,DIRECT,NATIVE,ROOT/'scripts'/'tailles_substrat_R351.py',
                 ROOT/'scripts'/'relaxer_noeuds_R354.py')
        dump(OUTPUT/f'LECTURE_{label}_R354.json',{'revision':354,'full':True,'physical_validation':False,
            'no_kinetic_refit':True,'N64_is_normalized_cubic_transfer_of_N48':True,'geometry':centroids,
            'frozen_kinetic_curves':rows,'input_sha256':{str(q.relative_to(ROOT)):digest(q) for q in files}})
        print('centroid shift A',centroids['max_centroid_shift_angstrom'],'effective r A',
              centroids['topological_centroid_lengths_angstrom'],flush=True)
    finally:
        model.close()


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('label')
    main(p.parse_args().label)
