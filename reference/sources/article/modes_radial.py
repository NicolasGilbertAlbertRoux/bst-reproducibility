'Linear modes q and s, in the radial subspace with fixed links.'

import numpy as np
from scipy.linalg import eigh
from stiffness_radial import coefficients


def modes(sol, parameters, radius, elements, common):
    'Static Hessian P1 for spherical variations with limited support.'
    nodes = np.linspace(sol.x[0], radius, elements + 1)
    step = nodes[1] - nodes[0]
    ids_f = np.r_[-1, np.arange(elements - 1), -1]
    ids_s = np.r_[np.arange(elements - 1, 2 * elements - 1), -1]
    hessienne = np.zeros((2 * elements - 1, 2 * elements - 1))
    mass = np.zeros_like(hessienne)
    coordinates, weights = np.polynomial.legendre.leggauss(4)
    d_f = np.array([-1., 1., 0., 0.]) / step
    d_s = np.array([0., 0., -1., 1.]) / step

    for i, (left, right) in enumerate(zip(nodes[:-1], nodes[1:], strict=True)):
        index = np.r_[ids_f[i:i + 2], ids_s[i:i + 2]]
        local = np.zeros((4, 4))
        weights_local = np.zeros((4, 4))
        r_points = (left + right + step * coordinates) / 2
        for r, (f, df, s, ds), t, w in zip(r_points, sol.sol(r_points).T, coordinates, weights, strict=True):
            base = np.array([1 - t, 1 + t]) / 2
            vf = np.r_[base, 0., 0.]
            vs = np.r_[0., 0., base]
            exp2s, beta, dbeta = coefficients(s, parameters)
            g, dg, d2g = np.sin(f)**2, np.sin(2 * f), 2 * np.cos(2 * f)
            grad = 2 * df * (beta * dg * vf + dbeta * g * vs)
            ff = (1 + beta * df**2) * d2g + beta * (dg**2 + g * d2g) / r**2
            fs = dbeta * dg * (df**2 + g / r**2)
            nonlinear = parameters.eta * r**2 * exp2s
            ss = 2 * dbeta * g * (df**2 + .5 * g / r**2) + nonlinear * ds**4 + parameters.rappel * r**2
            local += w * step / 2 * (
                (r**2 + 2 * beta * g) * np.outer(d_f, d_f)
                + (parameters.kappa * r**2 + 3 * nonlinear * ds**2) * np.outer(d_s, d_s)
                + np.outer(d_f, grad) + np.outer(grad, d_f)
                + ff * np.outer(vf, vf) + fs * (np.outer(vf, vs) + np.outer(vs, vf))
                + ss * np.outer(vs, vs)
                + 2 * nonlinear * ds**3 * (np.outer(d_s, vs) + np.outer(vs, d_s)))
            w_time = np.exp(-4*s) if common else 1.
            ms = 8.+4.*parameters.eta*exp2s*ds**2 if common else 8.
            weights_local += w*step/2*w_time*((r*r+2*beta*g)*np.outer(vf,vf)+ms*r*r*np.outer(vs,vs))
        active = index >= 0
        indices = np.ix_(index[active], index[active])
        hessienne[indices] += local[np.ix_(active, active)]
        mass[indices] += weights_local[np.ix_(active, active)]
    values = eigh(hessienne, mass, subset_by_index=(0, 1), eigvals_only=True)
    return np.sqrt(values)
