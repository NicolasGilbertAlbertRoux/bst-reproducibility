'Recompute native nodal dynamics with the selected ligand preparations.'

import argparse
import json
from pathlib import Path
from time import perf_counter

from sweep_sf4 import (ANGLE_UNC, LENGTH_TARGETS, LENGTH_UNC, OUTPUT, ROOT, SITES,
                             field_F, read)
from fields_and_references import dump
from dynamics_nodal_scripts import contract_covector
from trials_nodal import checkpoint, digest
from solve_acceleration_nodal import constrained_inverse
from readout_inertial import read as read_inertia
from reader_transported import FixedReader
from precision_sum import AccurateSumAction
from rotation_prepared import GATES, RotatingFrames, initial_world_frame, observed, step
import numpy as np


SOURCE = Path(__file__).resolve()
CANDIDATES = OUTPUT/'CANDIDATS_R348.json'
R345_CONTROL = ROOT/'results'/'CONTROLES_ENRICHIS_R345.json'
CASES = {f'N{n}_DT{ms:03d}':{'count':n,'dt':ms/1000,'end':.04}
         for n in (48,64) for ms in (10,5)}


def get_candidate(label):
    return next(row for row in read(CANDIDATES)['candidates'] if row['label']==label)


class AdjustedFrames(RotatingFrames):
    def __init__(self, count, candidate):
        super().__init__(count)
        self.candidate = candidate
        native = self.received.unpack(self.received_seed)
        self.target_covectors = self.received.kinetic(*native)['p']
        self.coarse_anchor = native[0].copy()
        for i, q in enumerate(candidate['F_preparations'],1):
            self.coarse_anchor[i] = field_F(self.received.c.model,i,q)
        self.coarse_anchor /= np.linalg.norm(self.coarse_anchor,axis=2)[...,None]
        for d in (self,self.received):
            if d.c.beta!=candidate['beta']:
                d.c.model.action.action = AccurateSumAction(candidate['beta'])
            d.c.beta = d.kernel.beta = candidate['beta']
            d.c.gamma = d.c.model.factor = candidate['gamma']
            d.c.kangle = d.c.model.m.weight = candidate['kangle']
        self.reader = None if count==48 else FixedReader(self.coarse_anchor,count,3,self.received)
        self.anchor = self.coarse_anchor.copy() if self.reader is None else self.reader.anchor.copy()
        self.pose = np.array(candidate['center_chart'])
        rate = np.zeros_like(self.anchor)
        angle_rate = read(R345_CONTROL)['initial']['configuration_velocity'][84:87]
        rate[:,:,1:] = np.cross(angle_rate,self.anchor[:,:,1:])
        mass = np.zeros((30,30))
        self.frame_columns = []
        for n,w,h,E in zip(self.anchor,rate,self.h,self.geometry(self.pose)[-1],strict=True):
            A,C,G = self.kernel.mass(n,h)
            W = np.ascontiguousarray(C[:,:,3:])
            self.frame_columns.append(W)
            Z = constrained_inverse(A,n)@W
            gram = W.reshape(-1,6).T@Z.reshape(-1,6)
            w -= Z@np.linalg.solve(gram,contract_covector(W,w))
            mass += E.T@G@E
        L = np.linalg.cholesky((mass+mass.T)/2)
        v = np.linalg.solve(L.T,np.linalg.solve(L,self.target_covectors))
        self.adjusted_seed, _ = self.project(self.pack(self.anchor,rate,self.pose,v))
        match = self.kinetic(*self.unpack(self.adjusted_seed))
        self.matching = {'target_R345':self.target_covectors,'replayed':match['p'],
                         'relative_inf':float(np.max(abs(match['p']-self.target_covectors)))/np.max(abs(self.target_covectors)),
                         'kinetic_native':match['T'],'minimum_collective_mass_eigenvalue':float(np.linalg.eigvalsh((mass+mass.T)/2)[0])}
        if self.matching['relative_inf']>GATES['matched_collective_covector_relative_inf']:
            raise ArithmeticError('The R345 momenta are not matched by the adjusted preparation')


def snapshot(model,state,frame,t):
    row = observed(model,state,frame,t)
    reference = model.candidate['reading']
    g = row['geometry']
    length = np.array(g['radii_native'])/model.candidate['scale_native_per_angstrom']
    i,j = np.triu_indices(4,1)
    angles = np.array(g['all_pair_angles_deg'])[i,j]
    dl = length-LENGTH_TARGETS
    da = angles-reference['six_reference_angles_deg']
    row['geometry_calibration'] = {'sites':SITES,'lengths_angstrom':length,'length_errors_angstrom':dl,
                                   'six_angles_deg':angles,'six_angle_errors_deg':da,
                                   'four_lengths_inside_published_intervals':bool(max(abs(dl))<=LENGTH_UNC+2e-12),
                                   'six_angles_inside_published_intervals':bool(max(abs(da))<=ANGLE_UNC+2e-12)}
    return row


def native_control(model,state):
    derivative = model.evaluate(state,diagnostics=True)
    _,acc,_,a = model.unpack(derivative)
    pose_gradient = np.array(model.received.c.last['gradient'][60:84])
    contacts = model.received.c.model.last['contacts']
    radial = np.array([r['radial_force'] for r in contacts])
    radii = model.received.c.model.last['radii']
    well = model.received.c.model.m.trial.well
    increments = (1e-4,5e-5)
    stiffnesses = [[r['kappa']+(well.separation_energy_derivative(l+h)[1]-
                    well.separation_energy_derivative(l-h)[1])/(2*h)
                    for r,l in zip(contacts,radii,strict=True)] for h in increments]
    readback = {'balance':model.last,'raw_radial_potential_derivatives':radial,
                'all_pose_potential_gradients':pose_gradient,
                'kappas':np.array([r['kappa'] for r in contacts]),
                'held_field_radial_stiffnesses_fd_native':stiffnesses,'radial_stiffness_fd_steps_native':increments,
                'maximum_field_acceleration_native':float(np.max(np.linalg.norm(acc,axis=2))),
                'rigid_acceleration_inf_native':float(np.max(abs(a)))}
    return readback


def audit_candidates():
    rows = []
    for candidate in read(CANDIDATES)['candidates']:
        model = AdjustedFrames(48,candidate)
        try:
            state = model.adjusted_seed
            geometry = snapshot(model,state,initial_world_frame(),0.)
            control = native_control(model,state)
            residual = np.max(abs(control['raw_radial_potential_derivatives']-candidate['radial_residuals_at_rest']))
            kappa = np.max(abs(control['kappas']-candidate['kappa_native']))
            if residual>2e-7 or kappa>2e-6:
                raise ArithmeticError('Reduced radial solver disagrees with the native full potential')
            read_inertial = read_inertia(model,state,0.,CANDIDATES)
            row = {'label':candidate['label'],'candidate_source_sha256':digest(CANDIDATES),
                   'control':control,'moment_match':model.matching,'observation':geometry,
                   'radial_formula_native_agreement_inf':residual,'kappa_native_agreement_inf':kappa,
                   'inertial_observations':read_inertial}
            rows.append(row)
            print(candidate['label'],'native radial agreement',residual,'acceleration',control['rigid_acceleration_inf_native'],
                  'rotor relative errors',read_inertial['frozen_shapes_native_rotor']['relative_errors'],flush=True)
        finally:
            model.close()
    dump(OUTPUT/'REJEU_NATIF_R348.json',{'rows':rows,'source_sha256':digest(SOURCE),'physical_validation':False})


def integrate(name):
    case = CASES[name]
    case_seed = get_candidate('lois_recues')
    model = AdjustedFrames(case['count'],case_seed)
    target = OUTPUT/f'{name}.npz'
    proof = {str(p.relative_to(ROOT)):digest(p) for p in (SOURCE,CANDIDATES,R345_CONTROL,ROOT/'MANIFESTE_R347.json')}
    try:
        if target.exists():
            with np.load(target) as saved:
                state,record = saved['state'],json.loads(str(saved['metadata']))
            if record['case']!=case or record['input_sha256']!=proof:
                raise ValueError('Checkpoint belongs to a different R348 recipe')
            if record['full']:
                dump(target.with_suffix('.json'),record)
                return
            frame = np.array(record['history'][-1]['world_frame'])
        else:
            state,frame = model.adjusted_seed,initial_world_frame()
            row = snapshot(model,state,frame,0.)
            row['native_control'] = native_control(model,state)
            record = {'revision':348,'case':case,'name':name,'candidate':case_seed['label'],'input_sha256':proof,
                      'moment_match':model.matching,'history':[row],'accepted_steps':0,'full':False,'physical_validation':False}
            checkpoint(target,state,record)
            dump(target.with_suffix('.json'),record)
        total = round(case['end']/case['dt'])
        first = record['history'][0]
        start = perf_counter()
        for k in range(record['accepted_steps'],total):
            state,frame,projection = step(model,state,frame,case['dt'])
            time = round((k+1)*case['dt'],12)
            row = snapshot(model,state,frame,time)
            dH = abs(row['H_native']-first['H_native'])/abs(first['H_native'])
            dP = float(np.linalg.norm(row['spatial_P']-first['spatial_P']))
            dJ = float(np.linalg.norm(row['spatial_J']-first['spatial_J']))/np.linalg.norm(first['spatial_J'])
            row.update({'energy_relative_drift':dH,'spatial_P_absolute_drift':dP,'spatial_J_relative_drift':dJ,
                        'projection_inf':projection})
            if (dH>GATES['H_relative'] or dP>GATES['spatial_P_absolute'] or dJ>GATES['spatial_J_relative'] or
                projection>GATES['state_projection_inf']):
                raise ArithmeticError(f'{name}: native numerical gate at {time}, checkpoint retained')
            if k+1==total:
                row['native_control'] = native_control(model,state)
            record['history'].append(row)
            record['accepted_steps'],record['full'] = k+1,k+1==total
            checkpoint(target,state,record)
            dump(target.with_suffix('.json'),record)
            values = row['geometry_calibration']
            print(name,'t',time,'dH/H',dH,'dJ/J',dJ,'max length error',max(abs(values['length_errors_angstrom'])),
                  'max angle error',max(abs(values['six_angle_errors_deg'])),'s',round(perf_counter()-start,1),flush=True)
    finally:
        model.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('auditer',*CASES))
    args = parser.parse_args()
    if args.action=='auditer':
        audit_candidates()
    else:
        integrate(args.action)
