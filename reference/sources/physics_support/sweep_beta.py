'Cover the received coefficient window with a one-sided bound, without new fits.'

from pathlib import Path

from sweep_sf4 import field_F, read
from fields_and_references import dump
from dynamics_full import NodalKinetic
from dynamics_nodal_scripts import NodalDynamics
from trials_nodal import digest
from clock_inertia import OUTPUT, R349, ROOT, all_orientations_bound
from reader_transported import FixedReader
from free_rotations import CALIBRATION
import numpy as np


def base_fields(model,candidate,initial_fields):
    n = initial_fields.copy()
    for i,q in enumerate(candidate['F_preparations'],1):
        n[i] = field_F(model.c.model,i,q)
    return n/np.linalg.norm(n,axis=2)[...,None]


def local_G(fields,spacing,beta):
    kernel = NodalKinetic(beta)
    return np.array([kernel.mass(a,h)[2][3:,3:].copy() for a,h in zip(fields,spacing,strict=True)])


def qualification(model,fields,expected,beta,count):
    h = np.array(model.h)*47/(count-1)
    n = fields if count==48 else FixedReader(fields,count,3,model).anchor
    lo,hi,mid = [local_G(n,h,beta*k) for k in (0.,4.,1.)]
    reconstructed = lo+(hi-lo)/4.
    affine_defect = float(np.max(abs(reconstructed-mid)))
    archived_defect = float(np.max(abs(mid-np.array(expected))))
    if max(affine_defect,archived_defect)>2e-8:
        raise ArithmeticError('The local kinetic blocks do not replay R349 or affine beta')
    return lo,(hi-lo)/4.,affine_defect,archived_defect


def main():
    source = read(CALIBRATION)['candidate']
    responses = read(R349/'REPONSE_CONTACTS_R349.json')
    bank = [r for r in read(R349/'BANQUE_ROTATION_R349.json')['rows'] if r['stage']=='anchor']
    clock_path = OUTPUT/'HORLOGE_ET_INERTIE_ABSOLUE_R350.json'
    clock = read(clock_path)
    L0 = clock['R348_geometry_length_interval_m'][0]*(1-1e-10)
    mass = clock['formula_isotopic_mass_kg']
    model = NodalDynamics(48)
    initial_fields = model.unpack(model.project(model.initial())[0])[0]
    grid = np.geomspace(.25,4,65)
    records = []
    for label,candidate,rows in [('free_anchor',source,bank),
                                 ('contact_null_anchor',responses['candidate'],responses['rows'])]:
        fields = base_fields(model,candidate,initial_fields)
        for row in rows:
            count = row['count']
            base,slope,affine,replayed = qualification(model,fields,row['native_blocks'],source['beta'],count)
            bounds = []
            center = {'count':count,'centers':row['centers']}
            safety = 5e-8*np.eye(6)[None,:,:]
            for lo,hi in zip(grid[:-1],grid[1:],strict=True):
                left,right = base+lo*slope-safety,base+hi*slope+safety
                result = all_orientations_bound({**center,'native_blocks':left},L0,mass,upper_blocks=right)
                result['beta_factor_interval'] = [float(lo),float(hi)]
                bounds.append(result)
            worst = np.min([r['absolute_minimum_relative_shortfall'] for r in bounds],axis=0)
            records.append({'fields':label,'count':count,'affine_defect':affine,
                'archived_G_replay_defect':replayed,'certified_relative_shortfalls':worst,'intervals':bounds,
                'all_intervals_excluded':all(r['all_three_constants_excluded'] for r in bounds)})
            print(label,count,'checked',len(bounds),'intervals, lower shortfalls',worst,flush=True)
    record = {'revision':350,'full':True,'coefficient':'beta with alpha held at 1',
              'native_beta_reference':source['beta'],'beta_factor_range':[.25,4.],
              'arithmetic_margin_per_kinetic_block':5e-8,'rows':records,
              'all_intervals_excluded':all(r['all_intervals_excluded'] for r in records),
              'scope':'kinetic test only; two fixed shapes and geometries; changing beta may require a new force and contact preparation',
              'physical_validation':False,'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (
                  CALIBRATION,R349/'REPONSE_CONTACTS_R349.json',R349/'BANQUE_ROTATION_R349.json',
                  clock_path,Path(__file__).resolve(),ROOT/'sources'/'cinetique_complete_R343.c')}}
    dump(OUTPUT/'BALAYAGE_BETA_R350.json',record)


if __name__=='__main__':
    main()
