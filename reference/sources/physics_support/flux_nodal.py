'Discrete energy split across the received cube on the padded trajectory.\n\nThe exterior action uses precisely the stencils missing from the cropped action.\nIts power is computed by differentiating this separate action along the common flow.'

import ctypes
from pathlib import Path

import numpy as np

from dynamics_nodal_scripts import NodalDynamics, ROOT, PARENT_COUNT
from refit_dependencies import ARRAY, load_c


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Unexpected received kernel layout for {old!r}')
    return source.replace(old, new, 1)


def exterior_sources():
    'Keep the parent algebra and cut the loops by exact stencil support.'
    source = ROOT / 'sources'
    kinetic = (source / 'cinetique_complete_R343.c').read_text().split('int bst_general_mass(')[0]
    kinetic = replace_once(kinetic, 'int bst_general_q(int n,', 'int bst_exterior_q(int n,int pad,')
    marker = 'if(!valid)continue;'
    kinetic = replace_once(kinetic, marker, marker + '''
   int interior=1;
   for(int a=0;a<3;a++){
    const int neighbor=coord[a]+sign[a];
    if(coord[a]<pad||coord[a]>=n-pad||neighbor<pad||neighbor>=n-pad)interior=0;
   }
   if(interior)continue;
''')
    potential = (source / 'action_parametree_R331.c').read_text()
    potential = replace_once(potential, 'int bst_action_params(int n,', 'int bst_exterior_u(int n,int pad,')
    potential = replace_once(potential, 'const double *core = field+center;', '''const double *core = field+center;
                const int inside=x>=pad&&y>=pad&&z>=pad&&x<n-pad&&y<n-pad&&z<n-pad;''')
    potential = replace_once(potential, 'if (coord[a]+1 == n) continue;', '''if (coord[a]+1 == n) continue;
                    if(inside&&coord[a]+1<n-pad) continue;''')
    potential = replace_once(potential, 'int valid = 1;', 'int valid = 1, inside_stencil=inside;')
    potential = replace_once(potential, 'index[a] = center+sign*stride[a];', '''index[a] = center+sign*stride[a];
                        if(coord[a]+sign<pad||coord[a]+sign>=n-pad)inside_stencil=0;''')
    potential = replace_once(potential, 'if (!valid) continue;', 'if (!valid||inside_stencil) continue;')
    generated = ROOT / 'results' / 'R346' / 'kernels'
    generated.mkdir(parents=True, exist_ok=True)
    files = []
    for name, body in [('cinetique_exterieure_R346.c', kinetic), ('action_exterieure_R346.c', potential)]:
        path = generated / name
        if path.exists() and path.read_text() != body:
            raise ValueError(f'Exterior kernel {name} differs from its exact parent transformation')
        if not path.exists():
            path.write_text(body)
        files.append(path)
    return files


class ExteriorKernel:
    def __init__(self, model):
        self.padding = model.padding
        self.count = model.count
        self.beta = model.c.beta
        kinetic, potential = exterior_sources()
        self.q, self.kinetic_hash = load_c(kinetic, 'bst_exterior_q',
            [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
             ARRAY, ARRAY, ARRAY, ARRAY, ARRAY, ARRAY, ctypes.POINTER(ctypes.c_double)])
        self.u, self.potential_hash = load_c(potential, 'bst_exterior_u',
            [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
             ARRAY, ARRAY, ARRAY])

    def quadratic(self, field, h, rate, local):
        gn = np.zeros_like(field)
        gt = np.zeros_like(field)
        gv = np.zeros(9)
        total = ctypes.c_double()
        status = self.q(self.count, self.padding, h, 1.0, self.beta,
                        field, rate, local, gn, gt, gv, ctypes.byref(total))
        if status:
            raise ValueError('Exterior kinetic stencil outside gnomonic chart')
        return total.value, gn, gt, gv

    def potential(self, field, h):
        gradient = np.zeros_like(field)
        total = np.zeros(3)
        if self.u(self.count, self.padding, h, 1.0, self.beta, field, gradient, total):
            raise ValueError('Exterior potential stencil outside gnomonic chart')
        return total[0], gradient


def power(kinetic, potential_gradients, center_gradient, t, v, a, rigid_acceleration):
    by_field = np.array([
        np.sum((gu + gt) * velocity) + np.sum(momentum * acceleration)
        for gu, gt, velocity, momentum, acceleration in zip(
            potential_gradients, kinetic['gn'], t, kinetic['pi'], a, strict=True)
    ])
    rigid = float((kinetic['gq'] + center_gradient) @ v[6:] + kinetic['p'] @ rigid_acceleration)
    return float(by_field.sum() + rigid), by_field, rigid


class ExteriorEnergy:
    def __init__(self, model):
        if model.count_inside != PARENT_COUNT or model.padding <= 0:
            raise ValueError('The cut is defined for a padded received lattice')
        self.model = model
        self.inner = NodalDynamics()
        self.exterior = ExteriorKernel(model)
        self.cut = slice(model.padding, model.padding + PARENT_COUNT)

    def crop(self, values):
        cube = values.reshape((5,) + (self.model.count,) * 3 + (4,))
        return np.ascontiguousarray(cube[:, self.cut, self.cut, self.cut]).reshape(5, -1, 4)

    def snapshot(self, state):
        d = self.model
        y, _ = d.project(state)
        n, t, x, v = d.unpack(y)
        derivative = d.evaluate(y)
        _, a, _, acc = d.unpack(derivative)
        full = d.last

        received_kernel = d.kernel
        try:
            d.kernel = self.exterior
            exterior_kinetic = d.kinetic(n, t, x, v, gradients=True)
        finally:
            d.kernel = received_kernel
        potentials = [self.exterior.potential(field, h) for field, h in zip(n, d.h, strict=True)]
        exterior_potential = sum(p[0] for p in potentials)
        outward, nodal, rigid = power(exterior_kinetic, [p[1] for p in potentials],
                                      np.zeros(24), t, v, a, acc)

        ni, ti, ai = self.crop(n), self.crop(t), self.crop(a)
        ui, gradients, gx, _ = self.inner.potential(ni, x)
        ki = self.inner.kinetic(ni, ti, x, v, gradients=True)
        inner_power, _, _ = power(ki, gradients, gx, ti, v, ai, acc)
        exterior_energy = float(exterior_potential + exterior_kinetic['T'])
        inner_energy = float(ui + ki['T'])
        energy_defect = float(full['H_native'] - (exterior_energy + inner_energy))
        power_defect = float(full['Hamiltonian_time_derivative_native'] - (outward + inner_power))
        if abs(energy_defect) > 3e-9 or abs(power_defect) > 3e-7:
            raise ArithmeticError(f'Discrete cut mismatch: energy {energy_defect}, power {power_defect}')
        return {'outer_energy': exterior_energy, 'inner_energy': inner_energy,
                'exterior_U': float(exterior_potential), 'exterior_T': float(exterior_kinetic['T']),
                'outward_power': outward, 'inner_power': inner_power,
                'full_power': full['Hamiltonian_time_derivative_native'],
                'exterior_nodal_power_by_atom': nodal, 'exterior_rigid_power': rigid,
                'energy_partition_defect': energy_defect, 'power_partition_defect': power_defect,
                'exterior_kinetic_source_sha256': self.exterior.kinetic_hash,
                'exterior_potential_source_sha256': self.exterior.potential_hash}
