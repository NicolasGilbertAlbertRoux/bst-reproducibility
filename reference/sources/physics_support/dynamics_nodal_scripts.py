'R345 full nodal action with local Schur contractions and grid transport.'

import os
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from scipy.ndimage import map_coordinates

from dynamics_nodal import FullNodalDynamics, ROOT
from solve_acceleration_nodal import constrained_inverse


PARENT_COUNT = 48
VACUUM = np.array([1.0, 0.0, 0.0, 0.0])


def grid_coordinates(count):
    'Positions expressed in parent lattice indices on the same cube.'
    coordinates = np.indices((count,) * 3, dtype=float)
    coordinates *= (PARENT_COUNT - 1) / (count - 1)
    return coordinates


def sample(values, coordinates, order):
    'Sample one scalar or a list of components on the received cube.'
    cube = values.reshape((PARENT_COUNT,) * 3 + values.shape[1:])
    if values.ndim == 1:
        return map_coordinates(cube, coordinates, order=order, mode='nearest').ravel()
    return np.column_stack([
        map_coordinates(cube[..., i], coordinates, order=order, mode='nearest').ravel()
        for i in range(values.shape[1])
    ])


def pad_cube(values, count, padding, exterior=0.0):
    trailing = values.shape[1:]
    extended = np.empty((count + 2 * padding,) * 3 + trailing)
    extended[...] = exterior
    inside = (slice(padding, padding + count),) * 3
    extended[inside] = values.reshape((count,) * 3 + trailing)
    return extended.reshape((-1,) + trailing)


def contract_covector(coupling, covector):
    return np.einsum('nca,nc->a', coupling, covector)


def apply_nodal(matrix, vector):
    return np.einsum('nij,nj->ni', matrix, vector)


class NodalDynamics(FullNodalDynamics):
    'Same potential and kinetic kernels as R345, including the held ports.'

    def __init__(self, count=48, padding=0, order=3):
        if count < 8 or count % 2 or padding < 0:
            raise ValueError('Use an even grid of at least eight nodes per side')
        super().__init__(derivative_step=1e-5)
        self.count = count + 2 * padding
        self.count_inside = count
        self.padding = padding
        self.order = order
        self.received_h = tuple(self.h)
        self.coordinates = None if count == PARENT_COUNT else grid_coordinates(count)
        if count != PARENT_COUNT:
            self.h = [h * (PARENT_COUNT - 1) / (count - 1) for h in self.received_h]
        self.shape = (5, self.count**3, 4)
        self.nfield = int(np.prod(self.shape))
        # Configuration.potential still records every full nodal covector.
        # Only the unused projections onto the former 60-parameter basis are bypassed.
        self.c._pull = lambda atom, gradient: np.zeros(52 if atom == 0 else 2)
        model = self.c.model
        if self.coordinates is not None or padding:
            for key, old_h, new_h in zip(('S', 'F'), (self.received_h[0], self.received_h[1]),
                                         (self.h[0], self.h[1]), strict=True):
                model.m.d[key + '_h'] = np.array(new_h)
                model.m.d[key + '_b_reference'] = self.transport(model.m.d[key + '_b_reference'])
                # node_offset is already a nodal integral in the received port.
                offset_density = model.m.d[key + '_node_offset'] / old_h**3
                model.m.d[key + '_node_offset'] = new_h**3 * self.transport(offset_density)
            model.m.d['grid_shape'] = np.full(3, self.count, dtype=int)
            positions = np.indices((self.count,) * 3, dtype=float).reshape(3, -1).T
            positions -= (self.count - 1) / 2
            model.m.unit = positions / np.linalg.norm(positions, axis=1)[:, None]

    def transport(self, values, exterior=0.0):
        if self.coordinates is not None:
            values = sample(values, self.coordinates, self.order)
        if self.padding:
            values = pad_cube(values, self.count_inside, self.padding, exterior)
        return values

    def initial(self):
        with np.load(ROOT / 'resultats/INITIAL_NODAL_R345.npz') as saved:
            received = saved['state']
        if self.count_inside == PARENT_COUNT and not self.padding:
            return received
        shape = (5, PARENT_COUNT**3, 4)
        field_size = int(np.prod(shape))
        fields = [self.transport(a, VACUUM) for a in received[:field_size].reshape(shape)]
        rates = [self.transport(a) for a in received[field_size:2 * field_size].reshape(shape)]
        x = received[-54:-30]
        v = received[-30:].copy()
        result, _ = self.project(self.pack(fields, rates, x, v))
        n, t, x, v = self.unpack(result)
        # Preserve the zero total translation momentum preparation on each mesh.
        # The centres, angular rates and transported nodal rates are held fixed.
        kinetic = self.kinetic(n, t, x, v)
        mass = np.zeros((3, 3))
        for field, embedding, h in zip(n, kinetic['E'], self.h, strict=True):
            _, _, rigid_mass = self.kernel.mass(field, h)
            translation = embedding[:, 3:6]
            mass += translation.T @ rigid_mass @ translation
        correction = np.linalg.solve(mass, -kinetic['p'][3:6])
        v[3:6] += correction
        self.initial_translation_correction = correction
        self.initial_translation_momentum_before_correction = kinetic['p'][3:6]
        return result

    def evaluate(self, y, diagnostics=False):
        started = time.perf_counter()
        y, projection = self.project(y)
        n, t, x, v = self.unpack(y)
        kinetic = self.kinetic(n, t, x, v, gradients=True)
        potential, potential_gradients, center_gradient, geometry = self.potential(n, x)

        # The derivative of the momentum is taken at zero generalized acceleration.
        speed_squared = np.sum(t * t, axis=2)[..., None]
        angle = self.eps * np.sqrt(speed_squared)
        cos = np.cos(angle)
        sin_over_speed = self.eps * np.sinc(angle / np.pi)
        plus = self.kinetic(n * cos + t * sin_over_speed,
                            t * cos - n * speed_squared * sin_over_speed,
                            x + self.eps * v[6:], v)
        minus = self.kinetic(n * cos - t * sin_over_speed,
                             t * cos + n * speed_squared * sin_over_speed,
                             x - self.eps * v[6:], v)
        momentum = kinetic['p']
        rigid_residual = (plus['p'] - minus['p']) / (2 * self.eps)
        rigid_residual[:3] -= np.cross(momentum[:3], v[:3]) + np.cross(momentum[3:6], v[3:6])
        rigid_residual[3:6] -= np.cross(momentum[3:6], v[:3])
        rigid_residual[6:] += center_gradient - kinetic['gq']

        rigid_mass = np.zeros((30, 30))
        schur = np.zeros((30, 30))
        forcing = -rigid_residual.copy()
        null_force_at_rest = np.r_[np.zeros(6), center_gradient].copy()
        blocks = []
        for i, (field, embedding, h) in enumerate(zip(n, kinetic['E'], self.h, strict=True)):
            residual = (plus['pi'][i] - minus['pi'][i]) / (2 * self.eps)
            residual += potential_gradients[i] - kinetic['gn'][i]
            residual -= field * np.sum(field * residual, axis=1)[:, None]
            mass, coupling, local_rigid_mass = self.kernel.mass(field, h)
            inverse = constrained_inverse(mass, field)
            # Only six columns survive the embedding in this preparation.
            coupling = np.ascontiguousarray(coupling[:, :, 3:])
            embedding = embedding[3:, :]
            local_rigid_mass = local_rigid_mass[3:, 3:]
            inverse_coupling = inverse @ coupling
            subtracted = coupling.reshape(-1, 6).T @ inverse_coupling.reshape(-1, 6)
            rigid_mass += embedding.T @ local_rigid_mass @ embedding
            schur += embedding.T @ (local_rigid_mass - subtracted) @ embedding
            forcing += embedding.T @ contract_covector(coupling, apply_nodal(inverse, residual))
            null_force_at_rest -= embedding.T @ contract_covector(
                inverse_coupling, potential_gradients[i])
            blocks.append((mass, coupling, inverse, embedding, residual))

        schur = (schur + schur.T) / 2
        lower = np.linalg.cholesky(schur)
        acceleration = np.linalg.solve(lower.T, np.linalg.solve(lower, forcing))
        field_acceleration = []
        rigid_balance = rigid_residual + rigid_mass @ acceleration
        nodal_defect = 0.0
        energy_derivative = (center_gradient + kinetic['gq']) @ v[6:] + momentum @ acceleration
        for i, (mass, coupling, inverse, embedding, residual) in enumerate(blocks):
            corrected = residual + coupling @ (embedding @ acceleration)
            tangent = -apply_nodal(inverse, corrected)
            rigid_balance += embedding.T @ contract_covector(coupling, tangent)
            nodal_balance = corrected + apply_nodal(mass, tangent)
            nodal_balance -= n[i] * np.sum(n[i] * nodal_balance, axis=1)[:, None]
            nodal_defect = max(nodal_defect, float(np.max(np.abs(nodal_balance))))
            ambient = tangent - n[i] * speed_squared[i]
            field_acceleration.append(ambient)
            energy_derivative += np.sum((potential_gradients[i] + kinetic['gn'][i]) * t[i])
            energy_derivative += np.sum(kinetic['pi'][i] * ambient)

        rigid_defect = float(np.max(np.abs(rigid_balance)))
        if nodal_defect >= 2e-7 or rigid_defect >= 3e-6:
            raise ArithmeticError(f'EL residual: nodal {nodal_defect}, rigid {rigid_defect}')
        self.calls += 1
        self.last = {
            'H_native': float(potential + kinetic['T']),
            'U_native': float(potential),
            'T_native': float(kinetic['T']),
            'Hamiltonian_time_derivative_native': float(energy_derivative),
            'field_EL_inf_native': nodal_defect,
            'rigid_EL_inf_native': rigid_defect,
            'body_momentum': momentum[:6],
            'spatial_angular_Casimir_native': float(momentum[:3] @ momentum[:3]),
            'geometry': geometry,
            'constraint_projection_defect': projection,
            'rigid_acceleration': acceleration,
            'maximum_field_acceleration_native': max(float(np.max(np.linalg.norm(a, axis=1)))
                                                      for a in field_acceleration),
            'RHS_seconds': time.perf_counter() - started,
        }
        if diagnostics:
            eigenvalues, eigenvectors = np.linalg.eigh(schur)
            self.last.update({
                'Schur_eigenvalues_native': eigenvalues,
                'unreduced_rigid_mass_trace': float(np.trace(rigid_mass)),
                'Schur_trace': float(np.trace(schur)),
                'Schur_rhs_native': forcing,
                'Schur_rhs_in_eigenbasis': eigenvectors.T @ forcing,
                'potential_derivative_along_mass_compensating_lift_at_rest': null_force_at_rest,
                'initial_translation_correction': getattr(self, 'initial_translation_correction', np.zeros(3)),
                'translation_momentum_before_correction': getattr(
                    self, 'initial_translation_momentum_before_correction', momentum[3:6]),
            })
        return self.pack(t, field_acceleration, v[6:], acceleration)
