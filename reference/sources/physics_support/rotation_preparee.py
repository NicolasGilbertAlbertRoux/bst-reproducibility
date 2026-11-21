'R345 rotating covectors in the R346 field-defined frames, with a fixed contact reader.'

import argparse
import json
from pathlib import Path
from time import perf_counter

from frames_defined import DefinedFrames
from fields_and_references import dump
from dynamics_nodal_scripts import NodalDynamics, ROOT, contract_covector
from trials_nodal import checkpoint, digest, geometry_snapshot
from solve_acceleration_nodal import constrained_inverse
from reader_transported import FixedReader
from parallel_ports import ParallelNodalDynamics
from refit_dependencies import skew
import numpy as np


OUTPUT = ROOT / 'results' / 'R347'
SOURCE = Path(__file__).resolve()
SEED = ROOT / 'results' / 'INITIAL_NODAL_R345.npz'
CASES = {f'N{count}_DT{millis:03d}': {'count':count, 'dt':millis/1000, 'end':.04}
         for count in (48, 64) for millis in (10, 5)}
GATES = {'matched_collective_covector_relative_inf':2e-12, 'projection_metric_identity_relative':1e-9,
         'shape_covector_relative_inf':2e-12, 'H_relative':2e-7,
         'spatial_P_absolute':2e-6, 'spatial_J_relative':2e-6,
         'state_projection_inf':1e-6, 'temporal_state_scaled_inf':3e-5,
         'late_return_state_scaled_inf':3e-5, 'instantaneous_reversal_rhs_inf':2e-7}


def protocol():
    declaration = {'revision':347, 'cases':CASES, 'gates':GATES, 'reader_count':48,
                   'source_rates':'R345 preparation, transported to the same cube by the R346 cubic rule',
                   'matched_covectors':'all 30 collective covectors from the original 48-node rotating preparation',
                   'shape_rates':'local tangent kinetic projection, preserving covectors on allowed variations',
                   'common_observation_times_native':[.0, .02, .04],
                   'space_transport_uncertainty':'report spatial differences separately from temporal differences',
                   'physical_validation':False}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    file = OUTPUT/'PROTOCOLE_R347.json'
    if file.exists():
        if json.loads(file.read_text()) != declaration:
            raise ValueError('R347 protocol changed')
    else:
        dump(file, declaration)


class RotatingFrames(DefinedFrames):
    def __init__(self, count, parallel=True):
        super().__init__(count)
        self.received = ParallelNodalDynamics(48) if parallel else NodalDynamics(48)
        self.received_seed, _ = self.received.project(self.received.initial())
        source_fields = self.received.unpack(self.received_seed)[0]
        self.reader = None if count == 48 else FixedReader(source_fields, count, 3, self.received)
        anchor = source_fields if count == 48 else self.reader.anchor
        if np.max(abs(anchor-self.anchor)) > 2e-12:
            raise ArithmeticError('Field-frame anchor differs from the received contact anchor')

    def potential(self, fields, pose):
        if self.reader is None:
            return self.received.potential(fields, pose)
        contact, gradients, pose_gradient, _, _ = self.reader.contact_potential(fields, pose)
        own = [self.c.model.action.action(field, h) for field, h in zip(fields, self.h, strict=True)]
        field_gradient = [gradient+row[1] for gradient, row in zip(gradients, own, strict=True)]
        return float(contact+sum(row[0] for row in own)), field_gradient, pose_gradient, self.received.c.last['geometry']

    def close(self):
        if hasattr(self.received, 'executor'):
            self.received.executor.shutdown()


def match_preparation(model):
    'Fix p and preserve the intrinsic covectors on the allowed tangent.'
    coarse = model.received.kinetic(*model.received.unpack(model.received_seed))
    if model.count == 48:
        source = model.received_seed
    else:
        free = NodalDynamics(model.count)
        source, _ = free.project(free.initial())
        del free
    n, t, x, v = model.unpack(source)
    if np.max(abs(n-model.anchor)) > 2e-12:
        raise ArithmeticError('The unconstrained preparation and the constraint anchor differ')
    raw = model.kinetic(n, t, x, v)
    mass = np.zeros((30,30))
    corrected = []
    initial_overlap = []
    for field, rate, W, h, E in zip(n, t, model.frame_columns, model.h, raw['E'], strict=True):
        A, _, G = model.kernel.mass(field, h)
        Z = constrained_inverse(A, field) @ W
        w = W.reshape(-1,6)
        gram = w.T @ Z.reshape(-1,6)
        overlap = contract_covector(W, rate)
        corrected.append(rate - Z @ np.linalg.solve(gram, overlap))
        initial_overlap.append(overlap)
        mass += E.T @ G @ E
    mass = (mass+mass.T)/2
    chol = np.linalg.cholesky(mass)
    solve = lambda rhs: np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
    own_v, common_v = solve(raw['p']), solve(coarse['p'])
    own, _ = model.project(model.pack(n, corrected, x, own_v))
    prepared, _ = model.project(model.pack(n, corrected, x, common_v))
    local = model.kinetic(*model.unpack(own))
    result = model.kinetic(*model.unpack(prepared))
    displacement = model.kinetic(n, t-model.unpack(own)[1], x, v-own_v)
    delivered = model.unpack(prepared)[1]
    distance = model.kinetic(n, t-delivered, x, v-common_v)
    relative_loss = (raw['T']-local['T'])/raw['T']
    orthogonality = abs(raw['T']-local['T']-displacement['T'])/raw['T']
    pscale = max(1., float(np.max(abs(coarse['p']))))
    p_error = float(np.max(abs(result['p']-coarse['p'])))/pscale
    own_error = float(np.max(abs(local['p']-raw['p'])))/pscale
    shape_error = 0.
    for field, W, before, after in zip(n, model.frame_columns, raw['pi'], result['pi'], strict=True):
        normal = W.reshape(-1,6)
        delta = after-before
        delta -= field*np.sum(field*delta, axis=1)[:,None]
        residual = delta.reshape(-1)-normal@np.linalg.solve(normal.T@normal, normal.T@delta.reshape(-1))
        shape_error = max(shape_error, float(np.max(abs(residual)))/max(1., float(np.max(abs(before)))))
    if (max(p_error,own_error) > GATES['matched_collective_covector_relative_inf'] or
        relative_loss < -GATES['projection_metric_identity_relative'] or
        orthogonality > GATES['projection_metric_identity_relative'] or
        shape_error > GATES['shape_covector_relative_inf']):
        raise ArithmeticError('Kinetic projection or collective momentum transport failed')
    result = {'coarse_target_covectors':coarse['p'], 'native_before_covectors':raw['p'],
              'native_prepared_covectors':result['p'], 'matched_covectors_relative_inf':p_error,
              'local_projection_covectors_relative_inf':own_error,
              'allowed_shape_covectors_relative_inf':shape_error, 'source_frame_velocity_overlaps':initial_overlap,
              'kinetic_R345_native':coarse['T'], 'kinetic_raw_this_grid_native':raw['T'],
              'kinetic_after_local_projection_native':local['T'],
              'kinetic_after_common_momentum_match_native':result['T'],
              'local_projection_kinetic_fraction_removed':relative_loss,
              'local_projection_Pythagorean_relative_error':orthogonality,
              'kinetic_distance_to_common_momentum_fraction':distance['T']/raw['T'],
              'maximum_field_rate_change_native':float(np.max(np.linalg.norm(t-delivered,axis=2))),
              'maximum_collective_rate_change_native':float(np.max(abs(common_v-v))),
              'minimum_collective_mass_eigenvalue':float(np.linalg.eigvalsh(mass)[0])}
    return prepared, result


def world_rate(state, frame):
    return np.column_stack((frame[:,:3] @ skew(state[-30:-27]), frame[:,:3] @ state[-27:-24]))


def step(model, state, frame, dt):
    k1, f1 = model.evaluate(state), world_rate(state, frame)
    s2, a2 = state+.5*dt*k1, frame+.5*dt*f1
    k2, f2 = model.evaluate(s2), world_rate(s2, a2)
    s3, a3 = state+.5*dt*k2, frame+.5*dt*f2
    k3, f3 = model.evaluate(s3), world_rate(s3, a3)
    s4, a4 = state+dt*k3, frame+dt*f3
    k4, f4 = model.evaluate(s4), world_rate(s4, a4)
    accepted, defect = model.project(state+(dt/6)*(k1+2*k2+2*k3+k4))
    frame = frame+(dt/6)*(f1+2*f2+2*f3+f4)
    u, _, vt = np.linalg.svd(frame[:,:3])
    if np.linalg.det(u@vt) <= 0:
        raise ArithmeticError('Global frame projection left SO(3)')
    frame[:,:3] = u@vt
    return accepted, frame, defect


def observed(model, state, frame, time):
    row = geometry_snapshot(model, state, time)
    rotation, position = frame[:,:3], frame[:,3]
    body = row['body_momentum']
    linear = rotation@body[3:6]
    row.update({'world_frame':frame, 'spatial_P':linear, 'spatial_J':rotation@body[:3]+np.cross(position,linear)})
    return row


def inputs(name):
    files = [SOURCE, SEED, OUTPUT/'PROTOCOLE_R347.json', ROOT/'MANIFESTE_R346.json',
             ROOT/'results'/'RECETTE_CHAMPS_LIBRES_R345.json']
    return {'case':CASES[name], 'sha256':{str(path.relative_to(ROOT)):digest(path) for path in files},
            'finite_difference_step_native':1e-5, 'parallel_contact_reader':True}


def initial_world_frame():
    return np.column_stack((np.eye(3), np.zeros(3)))


def run(name):
    recipe = inputs(name)
    case = recipe['case']
    target = OUTPUT/f'{name}.npz'
    log = target.with_suffix('.json')
    model = RotatingFrames(case['count'])
    try:
        if target.exists():
            with np.load(target) as saved:
                state, record = saved['state'], json.loads(str(saved['metadata']))
            if record['inputs'] != recipe:
                raise ValueError('Checkpoint belongs to a different recipe')
            if record['full']:
                dump(log, record)
                return
            frame = np.array(record['history'][-1]['world_frame'])
        else:
            state, matching = match_preparation(model)
            frame = initial_world_frame()
            row = observed(model, state, frame, 0.)
            model.evaluate(state, diagnostics=True)
            row['native_balance'] = model.last
            record = {'revision':347, 'name':name, 'inputs':recipe, 'momentum_match':matching,
                      'accepted_steps':0, 'history':[row], 'full':False, 'physical_validation':False}
            checkpoint(target,state,record)
            dump(log,record)
            print(name, 'covectors', matching['matched_covectors_relative_inf'], 'local kinetic loss',
                  matching['local_projection_kinetic_fraction_removed'], 'initial eigenmin',
                  model.last['Schur_minimum'], flush=True)
        start = perf_counter()
        dt = case['dt']
        total = round(case['end']/dt)
        initial = record['history'][0]
        for k in range(record['accepted_steps'], total):
            state, frame, projection = step(model, state, frame, dt)
            at = round((k+1)*dt,12)
            row = observed(model, state, frame, at)
            H = abs(row['H_native']-initial['H_native'])/abs(initial['H_native'])
            P = float(np.linalg.norm(row['spatial_P']-initial['spatial_P']))
            J = float(np.linalg.norm(row['spatial_J']-initial['spatial_J']))/np.linalg.norm(initial['spatial_J'])
            row.update({'energy_relative_drift':H, 'spatial_P_absolute_drift':P,
                        'spatial_J_relative_drift':J, 'projection_inf':projection})
            if (H > GATES['H_relative'] or P > GATES['spatial_P_absolute'] or J > GATES['spatial_J_relative']
                or projection > GATES['state_projection_inf']):
                raise ArithmeticError(f'{name} numerical gate at {at}: {(H,P,J,projection)}')
            if k+1 == total:
                model.evaluate(state, diagnostics=True)
                row['native_balance'] = model.last
            record['history'].append(row)
            record['accepted_steps'] = k+1
            record['full'] = k+1 == total
            checkpoint(target,state,record)
            dump(log,record)
            print(name, 't', at, 'dH/H', H, 'dJ/J', J, 'dP', P,
                  'seconds', round(perf_counter()-start,1), flush=True)
    finally:
        model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', choices=CASES)
    options = parser.parse_args()
    protocol()
    run(options.name)
