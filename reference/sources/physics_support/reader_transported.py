'Preserve the R321 contact reader at its own resolution during grid transfer.'

from dynamics_nodal_scripts import NodalDynamics, apply_nodal, contract_covector, ROOT
from trials_nodal import OUTPUT
from fields_and_references import dump
from solve_acceleration_nodal import constrained_inverse
import numpy as np
from scipy.ndimage import map_coordinates


def tensor_map(field, matrix, source_count):
    'Apply a separable one-axis map on the three spatial axes.'
    cube = field.reshape((source_count,) * 3 + (4,))
    for axis in range(3):
        moved = np.moveaxis(cube, axis, 0)
        transformed = (matrix @ moved.reshape(source_count, -1)).reshape(
            (matrix.shape[0],) + moved.shape[1:])
        cube = np.moveaxis(transformed, 0, axis)
    return cube.reshape(-1, 4)


def sphere_project(field):
    return field / np.linalg.norm(field, axis=1)[:, None]


class FixedReader:
    'Original contact terms and their variational adjoint on the fine grid.'

    def __init__(self, source_fields, count, order, received):
        self.count = count
        self.received = received
        points = [np.linspace(0, 47, count)]
        self.P = np.column_stack([map_coordinates(column, points, order=order, mode='nearest')
                                  for column in np.eye(48)])
        self.L = np.linalg.solve(self.P.T @ self.P, self.P.T)
        self.left_inverse_defect = float(np.max(abs(self.L @ self.P-np.eye(48))))
        self.one_axis_condition = float(np.linalg.cond(self.P))
        raw_fields = np.array([tensor_map(field, self.P, 48) for field in source_fields])
        self.weight = np.linalg.norm(raw_fields, axis=2)[..., None]
        if self.left_inverse_defect > 1e-12 or float(np.min(self.weight)) < .2:
            raise ArithmeticError('Prolongation lost rank or left the normalizable chart')
        self.anchor = raw_fields / self.weight

    def contact_potential(self, fields, pose):
        reduced = np.array([tensor_map(field*weight, self.L, self.count)
                            for field, weight in zip(fields, self.weight, strict=True)])
        norms = np.linalg.norm(reduced, axis=2)[..., None]
        if not np.all(np.isfinite(norms)) or np.min(norms) <= 1e-12:
            raise ValueError('Field outside the normalizable reader chart')
        observed = reduced/norms
        energy, gradients, pose_gradient, _ = self.received.potential(observed, pose)
        own = self.received.c.model.action.rows
        interaction = float(np.longdouble(energy)-sum(row[0] for row in own))
        field_gradients = []
        for i, (gradient, (_, self_gradient), field, norm, weight) in enumerate(zip(
                gradients, own, observed, norms, self.weight, strict=True)):
            g = gradient-self_gradient
            g = (g-field*np.sum(g*field, axis=1)[:, None])/norm
            field_gradients.append(weight*tensor_map(g, self.L.T, 48))
        ports = self.received.c.model.last['contacts']
        return interaction, field_gradients, pose_gradient, observed, ports


def probe():
    with np.load(ROOT/'resultats/INITIAL_NODAL_R345.npz') as saved:
        state = saved['state']
    n = state[:5*48**3*4].reshape(5, -1, 4)
    n = np.array([sphere_project(field) for field in n])
    pose = state[-54:-30].copy()
    del state
    received = NodalDynamics()
    E0, _, g0, _ = received.potential(n, pose)
    interaction0 = float(np.longdouble(E0)-sum(row[0] for row in received.c.model.action.rows))
    ports0 = received.c.model.last['contacts']
    fine = NodalDynamics(64)
    embeddings = fine.geometry(pose)[-1]
    reports = []
    for order in (1, 3):
        reader = FixedReader(n, 64, order, received)
        interaction, g, gx, observed, ports = reader.contact_potential(reader.anchor, pose)
        field_error = float(np.max(abs(observed-n)))
        energy_error = abs(interaction-interaction0)
        pose_error = float(np.max(abs(gx-g0)))
        port_error = max(abs(row[key]-old[key]) for row, old in zip(ports, ports0, strict=True)
                         for key in ('sulfur_port', 'F_port', 'delta'))
        if field_error > 1e-12 or energy_error > 1e-8 or pose_error > 1e-6 or port_error > 1e-9:
            raise ArithmeticError('The received contact reader was not preserved at its anchor')

        direction = np.sin(.013*np.arange(reader.anchor.size)).reshape(reader.anchor.shape)
        direction -= reader.anchor*np.sum(reader.anchor*direction, axis=2)[..., None]
        direction /= np.linalg.norm(direction)
        analytic = sum(float(np.sum(grad*step)) for grad, step in zip(g, direction, strict=True))
        finite = []
        for step in (1e-4, 5e-5):
            plus = np.array([sphere_project(a+step*d) for a, d in zip(reader.anchor, direction, strict=True)])
            minus = np.array([sphere_project(a-step*d) for a, d in zip(reader.anchor, direction, strict=True)])
            finite.append((reader.contact_potential(plus, pose)[0]-reader.contact_potential(minus, pose)[0])/(2*step))
        derivative_error = max(abs(value-analytic) for value in finite)
        if derivative_error > 2e-5*max(1, abs(analytic)):
            raise ArithmeticError('The contact reader adjoint failed its directional test')

        rigid_gradient = np.r_[np.zeros(6), gx]
        forcing = -rigid_gradient.copy()
        mass = np.zeros((30, 30))
        for field, gradient, E, h in zip(reader.anchor, g, embeddings, fine.h, strict=True):
            _, self_gradient = fine.c.model.action.action(field, h)
            A, C, G = fine.kernel.mass(field, h)
            H = constrained_inverse(A, field)
            mass += E.T @ G @ E
            forcing += E.T @ contract_covector(C, apply_nodal(H, gradient+self_gradient))
        acceleration = np.linalg.solve(mass, -rigid_gradient)
        report = {'field_interpolation_order': order, 'fine_count': 64,
                  'reader_count': 48, 'one_axis_condition': reader.one_axis_condition,
                  'left_inverse_defect': reader.left_inverse_defect,
                  'received_field_replay_inf': field_error,
                  'interaction_energy_absolute_defect': energy_error,
                  'port_values_inf_defect': port_error,
                  'rigid_gradient_inf_defect': pose_error,
                  'adjoint_analytic_directional': analytic, 'adjoint_finite_differences': finite,
                  'adjoint_maximum_absolute_defect': derivative_error,
                  'free_mass_compensating_force_norm_at_rest': float(np.linalg.norm(forcing)),
                  'combined_defined_frame_acceleration_norm_at_rest': float(np.linalg.norm(acceleration)),
                  'combined_rigid_balance_inf': float(np.max(abs(mass@acceleration+rigid_gradient))),
                  'physical_validation': False}
        reports.append(report)
        print('fixed reader', order, 'ports', port_error, 'gradient', pose_error,
              'free force', report['free_mass_compensating_force_norm_at_rest'],
              'defined frames acceleration', report['combined_defined_frame_acceleration_norm_at_rest'], flush=True)
    dump(OUTPUT/'LECTEUR_FIXE_ET_REPERES_R346.json', {'full': True, 'at_rest': True, 'rows': reports,
         'separate_fixed_reader_transport_candidate': True, 'physical_validation': False})


if __name__ == '__main__':
    probe()
