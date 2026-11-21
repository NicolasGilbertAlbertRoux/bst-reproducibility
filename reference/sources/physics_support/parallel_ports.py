'Evaluate independent R338 virtual curvatures concurrently in their original order.'

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path

from dynamics_nodal_scripts import NodalDynamics, ROOT
from angles_internal import move
import trials_nodal as runs
from fields_and_references import dump
import numpy as np


SOURCE = Path(__file__).resolve()
CASE = 'N64_DT005'


class ParallelNodalDynamics(NodalDynamics):
    def __init__(self, count=48, padding=0, order=3):
        super().__init__(count, padding, order)
        self.executor = ThreadPoolExecutor(max_workers=4)
        model = self.c.model
        if model.law != 'current' or model.F_reader != 'mobile':
            raise ValueError('This evaluation adapter is for the declared R338 current/mobile case')
        evaluate = model.evaluate
        original_refresh = model.state.refresh
        curvature = model.curvature

        def native_action(item):
            field, index = item
            return model.action.action(field, model.h(index))

        def parallel_evaluate(x):
            if model.action.record:
                raise RuntimeError('Independent curvatures require a non-recording action')
            x = np.asarray(x, float)
            fields = list(self.current_fields)
            own = list(self.executor.map(native_action, zip(fields, range(len(fields)), strict=True)))
            m = model.m
            positions = x[model.nstate:].reshape(m.N, 3)
            directions = move(m.D0, np.einsum('ia,iac->ic', positions[:,1:], m.tangents))
            requests = []
            for j, direction in enumerate(directions):
                opposite = -(model.frames[j].T @ direction)
                for atom, axis in ((0,direction),(j+1,opposite)):
                    reader = model.port(fields[atom], atom, axis)
                    requests.append((fields[atom],atom,reader['p'],own[atom]))
            values = list(self.executor.map(lambda request: curvature(*request), requests))
            cached = iter(zip(requests, values, strict=True))

            def refresh_cached(pose):
                if not np.array_equal(x, pose):
                    raise ValueError('Prepared action rows belong to another pose')
                model.state.x = x.copy()
                model.state.fields = fields
                model.state.m.fields = fields
                model.action.rows = own
                model.state.m.self_energies = [row[0] for row in own]

            def curvature_cached(field, atom, weights, baseline):
                request, value = next(cached)
                if (field is not request[0] or atom != request[1] or not np.array_equal(weights, request[2])
                        or baseline is not request[3]):
                    raise ValueError('Curvature request differs from its native ordering')
                return value

            model.state.refresh = refresh_cached
            model.curvature = curvature_cached
            try:
                answer = evaluate(x)
                if next(cached, None) is not None:
                    raise RuntimeError('The native evaluator skipped a curvature term')
            finally:
                model.state.refresh = original_refresh
                model.curvature = curvature
            return answer

        model.evaluate = parallel_evaluate


def qualify():
    source = runs.OUTPUT / f'{CASE}.npz'
    raw = source.read_bytes()
    with np.load(BytesIO(raw)) as saved:
        state = saved['state']
        record = json.loads(str(saved['metadata']))
    serial = NodalDynamics(64)
    expected = serial.evaluate(state)
    parallel = ParallelNodalDynamics(64)
    try:
        returned = parallel.evaluate(state)
        difference = float(np.max(abs(expected-returned)/np.maximum(1,abs(expected))))
        energy = abs(serial.last['H_native']-parallel.last['H_native'])
        if difference > 1e-10 or energy > 1e-10:
            raise ArithmeticError('The execution adapter changed the native solution')
        result = {'full':True, 'proof_time_native':record['history'][-1]['time_native'],
                  'proof_state_sha256':hashlib.sha256(raw).hexdigest(), 'scaled_rhs_inf':difference,
                  'energy_absolute_difference':energy, 'serial_seconds':serial.last['RHS_seconds'],
                  'parallel_seconds':parallel.last['RHS_seconds'], 'worker_count':4,
                  'source_sha256':runs.digest(SOURCE), 'physical_validation':False}
        dump(runs.OUTPUT/'CONTROLE_PARALLELE_PORTS_R346.json', result)
        print(result, flush=True)
    finally:
        parallel.executor.shutdown()


def continue_run():
    check = runs.OUTPUT/'CONTROLE_PARALLELE_PORTS_R346.json'
    proof = json.loads(check.read_text())
    if not proof['full'] or proof['source_sha256'] != runs.digest(SOURCE):
        raise ValueError('First qualify the current execution adapter')
    with np.load(runs.OUTPUT/f'{CASE}.npz') as saved:
        record = json.loads(str(saved['metadata']))
    if record['full']:
        return
    segment = {'time_native_start':record['history'][-1]['time_native'],
               'source':'scripts/parallele_ports_R346.py', 'source_sha256':runs.digest(SOURCE),
               'qualification':check.name, 'worker_count':4}
    native_checkpoint = runs.checkpoint

    def checkpoint_with_segment(path, state, metadata):
        entries = metadata.setdefault('execution_adapters', [])
        if segment not in entries:
            entries.append(segment)
        native_checkpoint(path,state,metadata)

    runs.NodalDynamics = ParallelNodalDynamics
    runs.checkpoint = checkpoint_with_segment
    runs.integrate(CASE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('qualify','continue'))
    options = parser.parse_args()
    if options.action == 'qualify':
        qualify()
    else:
        continue_run()
