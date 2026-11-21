'Test a holonomic local definition of the five frames with the same action.'

import argparse
import json
from pathlib import Path

from dynamics_nodal_scripts import NodalDynamics, apply_nodal, contract_covector
from trials_nodal import OUTPUT, checkpoint, digest, geometry_snapshot, install_protocol
from fields_and_references import dump
from solve_acceleration_nodal import constrained_inverse
from probe_boundary import momentum_drift
import numpy as np


def tangent_columns(field, columns):
    normal = np.einsum('ni,nij->nj', field, columns)
    return columns - field[:, :, None] * normal[:, None, :]


class DefinedFrames(NodalDynamics):
    'Each chart satisfies six fixed overlaps C0.T @ (n - n0) = 0.'

    def __init__(self, count):
        self.frame_columns = None
        super().__init__(count)
        seed, _ = super().project(super().initial())
        anchor, rates, x, v = self.unpack(seed)
        self.anchor = anchor.copy()
        self.at_rest = self.pack(self.anchor, np.zeros_like(rates), x, np.zeros_like(v))
        self.frame_columns = []
        for n, h in zip(self.anchor, self.h, strict=True):
            _, coupling, _ = self.kernel.mass(n, h)
            self.frame_columns.append(np.ascontiguousarray(coupling[:, :, 3:]))

    def project(self, y):
        y, defect = super().project(y)
        if self.frame_columns is None:
            return y, defect
        n, t, _, _ = self.unpack(y)
        for field, rate, anchor, columns in zip(n, t, self.anchor, self.frame_columns, strict=True):
            matrix = columns.reshape(-1, 6)
            original = field.copy()
            for _ in range(4):
                residual = contract_covector(columns, field - anchor)
                projected = tangent_columns(field, columns)
                gram = matrix.T @ projected.reshape(-1, 6)
                if np.max(abs(residual)) < 2e-11:
                    break
                field -= projected @ np.linalg.solve(gram, residual)
                field /= np.linalg.norm(field, axis=1)[:, None]
            else:
                raise ArithmeticError('Projection on the six frame overlaps failed')
            rate -= field * np.sum(field * rate, axis=1)[:, None]
            rate -= projected @ np.linalg.solve(gram, contract_covector(columns, rate))
            defect = max(defect, float(np.max(abs(original - field))))
        return y, defect

    def evaluate(self, y, diagnostics=False):
        y, _ = self.project(y)
        fields, rates, x, v = self.unpack(y)
        speed_squared = np.sum(rates * rates, axis=2)[..., None]
        _, _, _, _, _, embeddings = self.geometry(x)
        _, potential, gx, _ = self.potential(fields, x)
        drift_rigid, drift_fields = momentum_drift(self, fields, rates, x, v)
        rigid_residual = drift_rigid + np.r_[np.zeros(6), gx]
        schur = np.zeros((30, 30))
        mass = np.zeros_like(schur)
        forcing = -rigid_residual.copy()
        blocks = []
        for i, (field, embedding, h, W) in enumerate(zip(
                fields, embeddings, self.h, self.frame_columns, strict=True)):
            A, cross, G = self.kernel.mass(field, h)
            C = np.ascontiguousarray(cross[:, :, 3:])
            del cross
            E = embedding[3:]
            H = constrained_inverse(A, field)
            HW = H @ W
            HC = H @ C
            w = W.reshape(-1, 6)
            c = C.reshape(-1, 6)
            gram = w.T @ HW.reshape(-1, 6)
            multiplier_matrix = np.linalg.solve(gram, w.T @ HC.reshape(-1, 6))
            constrained_coupling = HC - HW @ multiplier_matrix
            residual = potential[i] + drift_fields[i]
            drive = apply_nodal(H, residual)
            radial = field * speed_squared[i]
            multiplier = np.linalg.solve(gram, contract_covector(W, drive + radial))
            field_drive = -drive + HW @ multiplier
            mass += E.T @ G[3:, 3:] @ E
            schur += E.T @ (G[3:, 3:] - c.T @ constrained_coupling.reshape(-1, 6)) @ E
            forcing -= E.T @ contract_covector(C, field_drive)
            blocks.append((A, C, E, W, residual, constrained_coupling, field_drive,
                           multiplier, multiplier_matrix, radial))
            del H, HW, HC

        schur = (schur + schur.T) / 2
        lower = np.linalg.cholesky(schur)
        acceleration = np.linalg.solve(lower.T, np.linalg.solve(lower, forcing))
        nodal = []
        rigid_balance = rigid_residual + mass @ acceleration
        defect = {'constraint': 0.0, 'velocity_constraint': 0.0,
                  'acceleration_constraint': 0.0, 'field_EL': 0.0, 'reaction_power': 0.0}
        for i, (A, C, E, W, residual, constrained_coupling, field_drive,
                multiplier, multiplier_matrix, radial) in enumerate(blocks):
            local = E @ acceleration
            tangent = field_drive - constrained_coupling @ local
            ambient = tangent - radial
            reaction = multiplier + multiplier_matrix @ local
            balance = apply_nodal(A, tangent) + C @ local + residual - W @ reaction
            balance -= fields[i] * np.sum(fields[i] * balance, axis=1)[:, None]
            velocity_constraint = contract_covector(W, rates[i])
            defect['constraint'] = max(defect['constraint'], float(np.max(abs(
                contract_covector(W, fields[i] - self.anchor[i])))))
            defect['velocity_constraint'] = max(defect['velocity_constraint'], float(np.max(abs(velocity_constraint))))
            defect['acceleration_constraint'] = max(defect['acceleration_constraint'], float(np.max(abs(
                contract_covector(W, ambient)))))
            defect['field_EL'] = max(defect['field_EL'], float(np.max(abs(balance))))
            defect['reaction_power'] += abs(float(reaction @ velocity_constraint))
            rigid_balance += E.T @ contract_covector(C, tangent)
            nodal.append(ambient)
        defect['rigid_EL'] = float(np.max(abs(rigid_balance)))
        if max(defect.values()) > 1e-7:
            raise ArithmeticError(f'Constrained field/rigid balance failed: {defect}')
        self.last = {'defects': defect, 'rigid_acceleration': acceleration}
        if diagnostics:
            self.last['Schur_minimum'] = float(np.linalg.eigvalsh(schur)[0])
        return self.pack(rates, nodal, v[6:], acceleration)


def rk4_step(model, state, dt):
    k1 = model.evaluate(state)
    k2 = model.evaluate(state + .5 * dt * k1)
    k3 = model.evaluate(state + .5 * dt * k2)
    k4 = model.evaluate(state + dt * k3)
    return model.project(state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))


def run(count, steps):
    model = DefinedFrames(count)
    state = model.at_rest
    dt = .02 / steps
    model.evaluate(state, diagnostics=True)
    initial_control = model.last
    anchor = json.loads((OUTPUT / f'LIMITE_N{count}_ORDRE3.json').read_text())
    acceleration_defect = float(np.max(abs(initial_control['rigid_acceleration'] - np.array(
        anchor['identical_rest_anchor']['constrained_rigid_acceleration']))))
    if acceleration_defect > 1e-8:
        raise ArithmeticError('The dynamic constraint disagrees with the independently solved anchor')
    initial = geometry_snapshot(model, state, 0)
    history = []
    for step in range(steps):
        state, projection = rk4_step(model, state, dt)
        row = geometry_snapshot(model, state, (step+1)*dt)
        row['projection'] = projection
        history.append(row)
        print('CADRES', count, 'dt', dt, 't', row['time_native'], 'dH',
              row['H_native'] - initial['H_native'], flush=True)
    model.evaluate(state, diagnostics=True)
    final_control = model.last
    energy_drift = max(abs(row['H_native'] - initial['H_native']) / initial['H_native'] for row in history)
    if energy_drift > 2e-7:
        raise ArithmeticError('Constrained trajectory failed the native energy gate')
    record = {'revision': 346, 'full': True, 'count': count, 'steps': steps,
              'duration_native': .02, 'initial': initial, 'history': history,
              'anchor_acceleration_absolute_defect': acceleration_defect,
              'initial_control': initial_control, 'final_control': final_control,
              'maximum_relative_energy_drift': energy_drift,
              'initial_rates': 'all zero, separate preparation from R345',
              'frame_definition': 'six overlaps per field, fixed at the rest anchor',
              'source_sha256': digest(Path(__file__).resolve()),
              'no_added_mass_or_changed_energy': True, 'physical_validation': False}
    target = OUTPUT / f'CADRES_N{count}_{steps}PAS'
    checkpoint(target.with_suffix('.npz'), state, record)
    dump(target.with_suffix('.json'), record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('count', type=int, choices=(48, 64))
    parser.add_argument('--steps', type=int, choices=(2, 4), default=4)
    arguments = parser.parse_args()
    install_protocol()
    run(arguments.count, arguments.steps)
