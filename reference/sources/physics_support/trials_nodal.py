'Source-bound controls, instantaneous mesh series and continued RK4 runs.'

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import scipy

from dynamics_nodal import FullNodalDynamics, dump
from fields_and_references import plain
from dynamics_nodal_scripts import NodalDynamics, ROOT


OUTPUT = ROOT / 'results' / 'R346'
CASES = {
    'N48_DT010': {'count': 48, 'padding': 0, 'dt': .01, 'end': .20, 'prefix': 'N2'},
    'N48_DT005': {'count': 48, 'padding': 0, 'dt': .005, 'end': .20, 'prefix': 'N4'},
    'N40_DT005': {'count': 40, 'padding': 0, 'dt': .005, 'end': .12, 'prefix': None},
    'N64_DT005': {'count': 64, 'padding': 0, 'dt': .005, 'end': .12, 'prefix': None},
    'BORD56_DT005': {'count': 48, 'padding': 4, 'dt': .005, 'end': .12, 'prefix': None},
}
GATES = {
    'temporal_state_scaled_inf': 3e-5,
    'energy_relative': 2e-7,
    'constraint_projection': 1e-6,
    'translation_momentum_norm_absolute_drift': 2e-6,
    'angular_momentum_square_relative_drift': 2e-6,
    'factored_rhs_scaled_inf': 3e-6,
    'maximum_snapshot_interval_native': .02,
}


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def install_protocol():
    protocol = {'revision': 346, 'cases': CASES, 'numerical_gates': GATES,
                'mesh_snapshot_counts': [40, 48, 64, 80, 96],
                'transport': 'component spline, S3 projection, tangent rates, fixed cube, P=0',
                'vacuum_extension': 'four added nodes per face, same spacing and received interior',
                'same_action_coefficients': True, 'physical_validation': False}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / 'PROTOCOLE_R346.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('The declared protocol differs from the existing one')
    if not path.exists():
        dump(path, protocol)


def bound_inputs(case):
    paths = [ROOT / 'results' / 'RECETTE_CHAMPS_LIBRES_R345.json',
             ROOT / 'results' / 'INITIAL_NODAL_R345.npz',
             ROOT / 'scripts' / 'dynamique_nodale_R346.py', Path(__file__).resolve(),
             OUTPUT / 'PROTOCOLE_R346.json']
    if case['prefix']:
        paths.append(ROOT / 'results' / f"{case['prefix']}_CHAMPS_LIBRES_R345.npz")
    if case['padding']:
        paths.append(ROOT / 'scripts' / 'flux_nodal_R346.py')
    return {'case': case, 'sha256': {str(p.relative_to(ROOT)): digest(p) for p in paths},
            'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
            'platform': platform.platform(), 'longdouble_epsilon': float(np.finfo(np.longdouble).eps)}


def controls():
    original = FullNodalDynamics(derivative_step=1e-5)
    factored = NodalDynamics()
    y0 = factored.initial()
    with np.load(ROOT / 'results' / 'N4_CHAMPS_LIBRES_R345.npz') as saved:
        y1 = saved['last_state']
    comparisons = []
    for name, y in [('initial', y0), ('t=0.02', y1)]:
        reference = original.evaluate(y, diagnostics=True)
        candidate = factored.evaluate(y, diagnostics=True)
        difference = float(np.max(abs(candidate - reference) / np.maximum(1, abs(reference))))
        row = {'state': name, 'scaled_rhs_inf': difference,
               'energy_absolute_difference': abs(factored.last['H_native'] - original.last['H_native']),
               'minimum_Schur_difference': abs(factored.last['Schur_eigenvalues_native'][0]
                                               - original.last['Schur_eigenvalue_range_native'][0]),
               'R345_rhs_seconds': original.last['RHS_seconds'],
               'R346_rhs_seconds': factored.last['RHS_seconds']}
        if difference >= GATES['factored_rhs_scaled_inf'] or row['energy_absolute_difference'] >= 1e-9:
            raise ArithmeticError(f'R345 comparison failed: {row}')
        comparisons.append(row)
        print(row, flush=True)
    dump(OUTPUT / 'CONTROLE_FACTORISATION_R346.json', {'full': True, 'comparisons': comparisons,
                                                       'numerical_limit': GATES['factored_rhs_scaled_inf']})


def mesh(count, order):
    model = NodalDynamics(count, order=order)
    y = model.initial()
    model.evaluate(y, diagnostics=True)
    result = {'full': True, 'count': count, 'spline_order': order,
              'h': model.h, 'box_side': np.array(model.h) * (count - 1), **model.last}
    dump(OUTPUT / f'POINT_N{count}_ORDRE{order}.json', result)
    print('mesh', count, 'order', order, 'trace', result['Schur_trace'],
          'min', result['Schur_eigenvalues_native'][0], 'max', result['Schur_eigenvalues_native'][-1],
          'forcing', np.linalg.norm(result['Schur_rhs_native']),
          'acc', np.linalg.norm(result['rigid_acceleration']), flush=True)


def checkpoint(path, state, metadata):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as file:
        np.savez_compressed(file, state=state, metadata=np.array(json.dumps(plain(metadata), allow_nan=False)))
    temporary.replace(path)


def geometry_snapshot(model, state, at):
    value = model.value(state)
    _, _, x, v = model.unpack(state)
    value.update({'time_native': at, 'rigid_position': x, 'rigid_velocity': v})
    return value


def integrate(name):
    control = OUTPUT / 'CONTROLE_FACTORISATION_R346.json'
    if not control.exists() or not json.loads(control.read_text())['full']:
        raise ValueError('Qualify the factored solver before running trajectories')
    case = CASES[name]
    dt = case['dt']
    proof = bound_inputs(case)
    model = NodalDynamics(case['count'], case['padding'])
    start_time = .02 if case['prefix'] else 0.0
    total = round((case['end'] - start_time) / dt)
    stride = round(GATES['maximum_snapshot_interval_native'] / dt)
    file = OUTPUT / f'{name}.npz'
    log = OUTPUT / f'{name}.json'
    flux = None
    if case['padding']:
        from flux_nodal import ExteriorEnergy
        flux = ExteriorEnergy(model)
    if file.exists():
        with np.load(file) as saved:
            state, record = saved['state'], json.loads(str(saved['metadata']))
        if record['inputs'] != proof:
            raise ValueError(f'{name}: the checkpoint belongs to another recipe')
        if record['full']:
            dump(log, record)
            return
    else:
        if case['prefix']:
            with np.load(ROOT / 'results' / f"{case['prefix']}_CHAMPS_LIBRES_R345.npz") as prefix:
                state = prefix['last_state']
                received_initial = prefix['initial_state']
            initial = geometry_snapshot(model, received_initial, 0.0)
        else:
            state = model.initial()
            initial = geometry_snapshot(model, state, 0.0)
        first = geometry_snapshot(model, state, start_time)
        if flux is not None:
            first['cut'] = flux.snapshot(state)
        record = {'revision': 346, 'name': name, 'inputs': proof, 'initial': initial,
                  'first': first, 'full': False, 'accepted_steps': 0, 'planned_steps': total,
                  'max_projection': 0.0, 'history': [], 'physical_validation': False}
    started = perf_counter()
    for k in range(record['accepted_steps'], total):
        k1 = model.evaluate(state)
        k2 = model.evaluate(state + .5 * dt * k1)
        k3 = model.evaluate(state + .5 * dt * k2)
        k4 = model.evaluate(state + dt * k3)
        state, defect = model.project(state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        del k1, k2, k3, k4
        record['max_projection'] = max(record['max_projection'], defect)
        if (k + 1) % stride and k + 1 != total:
            continue
        at = round(start_time + (k + 1) * dt, 12)
        entry = geometry_snapshot(model, state, at)
        if flux is not None:
            entry['cut'] = flux.snapshot(state)
        record['history'].append(entry)
        record['accepted_steps'] = k + 1
        record['full'] = k + 1 == total
        checkpoint(file, state, record)
        dump(log, record)
        drift = (entry['H_native'] - record['initial']['H_native']) / record['initial']['H_native']
        print(name, 't', at, 'dH/H', drift, 'projection', defect,
              'radii', entry['geometry']['radii_native'], 'seconds', round(perf_counter()-started, 1), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('controls')
    grid = commands.add_parser('mesh')
    grid.add_argument('count', type=int, nargs='+')
    grid.add_argument('--order', type=int, default=3, choices=(1, 3))
    trajectory = commands.add_parser('run')
    trajectory.add_argument('name', choices=CASES)
    args = parser.parse_args()
    install_protocol()
    if args.action == 'controls':
        controls()
    elif args.action == 'mesh':
        for size in args.count:
            mesh(size, args.order)
    else:
        integrate(args.name)
