'Stream the rigid Schur balance and test a local definition of the frames.'

import argparse

from dynamics_nodal_scripts import NodalDynamics, apply_nodal, contract_covector
from trials_nodal import OUTPUT, install_protocol
from fields_and_references import dump
from solve_acceleration_nodal import constrained_inverse
import numpy as np


def momentum_drift(model, n, t, x, v):
    'Use the same geodesic time derivative as in the received R345 solver.'
    base = model.kinetic(n, t, x, v, gradients=True)
    eps = model.eps
    speed_squared = np.sum(t * t, axis=2)[..., None]
    angle = eps * np.sqrt(speed_squared)
    cosine = np.cos(angle)
    sine = eps * np.sinc(angle / np.pi)
    plus = model.kinetic(n * cosine + t * sine,
                         t * cosine - n * speed_squared * sine, x + eps * v[6:], v)
    minus = model.kinetic(n * cosine - t * sine,
                          t * cosine + n * speed_squared * sine, x - eps * v[6:], v)
    p = base['p']
    rigid = (plus['p'] - minus['p']) / (2 * eps)
    rigid[:3] -= np.cross(p[:3], v[:3]) + np.cross(p[3:6], v[3:6])
    rigid[3:6] -= np.cross(p[3:6], v[:3])
    rigid[6:] -= base['gq']
    field = [(pplus - pminus) / (2 * eps) - g
             for pplus, pminus, g in zip(plus['pi'], minus['pi'], base['gn'], strict=True)]
    return rigid, field


def probe(count, order, with_motion):
    model = NodalDynamics(count, order=order)
    y, _ = model.project(model.initial())
    fields, rates, x, v = model.unpack(y)
    _, _, _, _, _, embeddings = model.geometry(x)
    U, gradients, gx, geometry = model.potential(fields, x)
    rigid_gradient = np.r_[np.zeros(6), gx]
    if with_motion:
        drift_rigid, drift_fields = momentum_drift(model, fields, rates, x, v)
        forcing = -(rigid_gradient + drift_rigid)
    forcing_rest = -rigid_gradient.copy()
    mass = np.zeros((30, 30))
    schur = np.zeros_like(mass)
    direction = np.sin(.37 * np.arange(30) + .23)
    direction /= np.linalg.norm(direction)
    direct_quadratic = 0.0
    frame_constraints = []
    for i, (field, gradient, embedding, h) in enumerate(zip(
            fields, gradients, embeddings, model.h, strict=True)):
        nodal_mass, full_coupling, local_mass = model.kernel.mass(field, h)
        inverse = constrained_inverse(nodal_mass, field)
        coupling = np.ascontiguousarray(full_coupling[:, :, 3:])
        del full_coupling
        mobility = inverse @ coupling
        columns = coupling.reshape(-1, 6)
        frame_metric = columns.T @ mobility.reshape(-1, 6)
        frame_metric = (frame_metric + frame_metric.T) / 2
        local = local_mass[3:, 3:]
        E = embedding[3:]
        mass += E.T @ local @ E
        schur += E.T @ (local - frame_metric) @ E

        motion_due_to_force = apply_nodal(inverse, gradient)
        frame_forcing = contract_covector(coupling, motion_due_to_force)
        forcing_rest += E.T @ frame_forcing
        if with_motion:
            forcing += E.T @ contract_covector(
                coupling, apply_nodal(inverse, gradient + drift_fields[i]))

        lifted_rate = -(mobility @ (E @ direction))
        direct_quadratic += model.kernel.quadratic(field, h, lifted_rate, embedding @ direction)[0]

        # Holonomic local preparation: C0^T(n-n0)=0 in each atom's chart.
        # At rest and at n0, C0 is the current coupling. The corresponding
        # acceleration constraint is C0^T n_ddot=0. The reaction has no power
        # for admissible velocities and does not add kinetic energy.
        multiplier = np.linalg.solve(frame_metric, frame_forcing)
        constrained_field_acceleration = -motion_due_to_force + mobility @ multiplier
        constraint_residual = contract_covector(coupling, constrained_field_acceleration)
        balance = gradient + apply_nodal(nodal_mass, constrained_field_acceleration) - coupling @ multiplier
        balance -= field * np.sum(field * balance, axis=1)[:, None]
        frame_constraints.append({
            'atom': i,
            'constraint_gram_minimum_eigenvalue': float(np.linalg.eigvalsh(frame_metric)[0]),
            'acceleration_constraint_inf': float(np.max(abs(constraint_residual))),
            'tangent_field_balance_inf': float(np.max(abs(balance))),
            'reaction_multiplier_before_rigid_term': multiplier,
        })
        del nodal_mass, inverse, coupling, mobility, columns, lifted_rate, constrained_field_acceleration

    schur = (schur + schur.T) / 2
    mass = (mass + mass.T) / 2
    direct_error = abs(direct_quadratic - direction @ schur @ direction)
    free = np.linalg.solve(schur, forcing_rest)
    corrected = np.linalg.solve(mass, -rigid_gradient)
    corrected_defect = np.max(abs(rigid_gradient + mass @ corrected))
    if (direct_error > 1e-7 or corrected_defect > 1e-8
            or max(row['acceleration_constraint_inf'] for row in frame_constraints) > 1e-7
            or max(row['tangent_field_balance_inf'] for row in frame_constraints) > 1e-8
            or min(row['constraint_gram_minimum_eigenvalue'] for row in frame_constraints) <= 0):
        raise ArithmeticError('The direct mass or local constrained balance test failed')
    result = {
        'revision': 346, 'full': True, 'count': count, 'spline_order': order,
        'h': model.h, 'potential': U, 'geometry': geometry,
        'free_schur_spectrum': np.linalg.eigvalsh(schur),
        'free_schur_trace': float(np.trace(schur)),
        'frame_constrained_rigid_mass_spectrum_at_anchor': np.linalg.eigvalsh(mass),
        'frame_constrained_rigid_mass_trace': float(np.trace(mass)),
        'direct_quadratic_and_schur_absolute_defect': float(direct_error),
        'moving_initial_state': None if not with_motion else {
            'rigid_forcing': forcing, 'rigid_acceleration': np.linalg.solve(schur, forcing),
            'translation_correction': getattr(model, 'initial_translation_correction', np.zeros(3)),
        },
        'identical_rest_anchor': {'free_rigid_forcing': forcing_rest, 'free_rigid_acceleration': free,
                                  'constrained_rigid_acceleration': corrected,
                                  'constrained_rigid_balance_inf': float(corrected_defect)},
        'frame_constraints': frame_constraints,
        'number_of_local_frame_constraints': 30,
        'same_potential_and_kinetic_action': True,
        'new_holonomic_definition_of_frames_tested_at_rest_only': True,
        'physical_validation': False,
    }
    dump(OUTPUT / f'LIMITE_N{count}_ORDRE{order}.json', result)
    print('N', count, 'order', order, 'Tr(S)', np.trace(schur), 'min(D)', np.linalg.eigvalsh(mass)[0],
          '|force rest|', np.linalg.norm(forcing_rest), '|a free|', np.linalg.norm(free),
          '|a constrained|', np.linalg.norm(corrected),
          '|a moving|' if with_motion else 'au repos',
          np.linalg.norm(result['moving_initial_state']['rigid_acceleration']) if with_motion else '', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('count', type=int, nargs='+')
    parser.add_argument('--order', type=int, choices=(1, 3), default=3)
    parser.add_argument('--moving', action='store_true')
    options = parser.parse_args()
    install_protocol()
    for size in options.count:
        probe(size, options.order, options.moving)
