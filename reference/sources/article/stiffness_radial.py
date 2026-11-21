'BST static radial sector with non-linear gradient stiffness.'

from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson, solve_bvp
from scipy.linalg import eigh


@dataclass(frozen=True)
class Parameters:
    kappa: float = 2.0
    eta: float = 0.0
    rappel: float = 0.0
    floor: float = 0.0


def coefficients(s, parameters):
    exp2s = np.exp(2 * np.clip(s, -100., 40.))
    derivative = 2 * (1 - parameters.floor) * exp2s
    return exp2s, parameters.floor + derivative / 2, derivative


def equations(r, y, parameters):
    f, df, s, ds = y
    exp2s, beta, d_beta = coefficients(s, parameters)
    g = np.sin(f)**2
    dg = np.sin(2 * f)
    mass = r**2 + 2 * beta * g
    d_mass = 2 * r + 2 * (d_beta * ds * g + beta * dg * df)
    potential = dg * (1 + beta * (df**2 + g / r**2))
    current = (parameters.kappa + parameters.eta * exp2s * ds**2) * ds
    source = d_beta * g * (df**2 + g / (2 * r**2)) / r**2
    dds = (source + parameters.rappel * s
           - 1.5 * parameters.eta * exp2s * ds**4 - 2 * current / r
           ) / (parameters.kappa + 3 * parameters.eta * exp2s * ds**2)
    return np.array([df, (potential - d_mass * df) / mass, ds, dds])


def solve(parameters, degree=1, radius=160., tolerance=1e-7, precedent=None, origin=1e-4):
    grid = np.geomspace(origin, radius, 360)
    mu = np.sqrt(parameters.rappel / parameters.kappa)
    if precedent is None:
        sizes = np.geomspace(0.55, 1.55, degree) if degree > 1 else [0.95]
        f = sum(2 * np.arctan((a / grid)**2) for a in sizes)
        df = sum(-4 * a**2 * grid / (grid**4 + a**4) for a in sizes)
        s = -0.1 / np.sqrt(1 + grid**2)
        ds = 0.1 * grid / (1 + grid**2)**1.5
        initial = np.array([f, df, s, ds])
    else:
        boundary = precedent.x[-1]
        initial = precedent.sol(np.minimum(grid, boundary))
        outside = grid > boundary
        r = grid[outside]
        initial[0, outside] *= (boundary / r)**2
        initial[1, outside] = -2 * initial[0, outside] / r
        initial[2, outside] *= boundary / r * np.exp(-mu * (r - boundary))
        initial[3, outside] = -(1 / r + mu) * initial[2, outside]

    def bordure(left, right):
        exp2s, _, dbeta = coefficients(left[2], parameters)
        source_center = 1.5 * dbeta * left[1]**4 + parameters.rappel * left[2]
        flux_center = (parameters.kappa + parameters.eta * exp2s * left[3]**2) * left[3]
        return np.array([left[0] - origin * left[1] - degree * np.pi,
                         (flux_center - origin * source_center / 3) / parameters.kappa,
                         radius * right[1] + 2 * right[0],
                         right[3] + (1 / radius + mu) * right[2]])

    return solve_bvp(lambda r, y: equations(r, y, parameters), bordure, grid, initial,
                     tol=tolerance, max_nodes=8500)


def measurements(sol, parameters):
    output = dict(converge=bool(sol.success), nodes=int(sol.x.size),
                  residual=float(np.max(sol.rms_residuals)), radius_box=float(sol.x[-1]))
    if not sol.success:
        return output
    r = np.geomspace(sol.x[0], sol.x[-1], 16385)
    f, df, s, ds = sol.sol(r)
    exp2s, beta, d_beta = coefficients(s, parameters)
    g = np.sin(f)**2
    e4 = g * (df**2 + g / (2 * r**2))

    def integre(v):
        return float(4 * np.pi * simpson(v, x=r))

    # First-order analytical exterior for F and s. Contributions
    # external quartics are evaluated by checking the can radius.
    mu = np.sqrt(parameters.rappel / parameters.kappa)
    e2 = integre(.5 * (r * df)**2 + g) + 4 * np.pi * r[-1] * f[-1]**2
    e4_matter = integre(beta * e4)
    e4_field = integre(.25 * parameters.eta * r**2 * exp2s * ds**4)
    e_grad = integre(.5 * parameters.kappa * (r * ds)**2)
    e_rappel = integre(.5 * parameters.rappel * (r * s)**2)
    queue_rappel = np.pi * parameters.rappel * (r[-1] * s[-1])**2 / mu if mu else 0.
    e_grad += 2 * np.pi * parameters.kappa * r[-1] * (1 + mu * r[-1]) * s[-1]**2 - queue_rappel
    e_rappel += queue_rappel
    energy = e2 + e4_matter + e4_field + e_grad + e_rappel
    q_matter = integre(d_beta * e4)
    q = q_matter + 2 * e4_field
    flux = 4 * np.pi * r[-1]**2 * (parameters.kappa + parameters.eta * exp2s[-1] * ds[-1]**2) * ds[-1]
    flux_inner = 4 * np.pi * r[0]**2 * (parameters.kappa + parameters.eta * exp2s[0] * ds[0]**2) * ds[0]
    flux_expected = q + integre(parameters.rappel * r**2 * s)
    charge_abs = -g * df
    degree = 2 / np.pi * simpson(charge_abs, x=r)
    radius_rms = np.sqrt(simpson(r**2 * charge_abs, x=r) / simpson(charge_abs, x=r))
    output.update(E2=e2, E4_matter=e4_matter, E4_field=e4_field, E_gradient=e_grad,
                  E_rappel=e_rappel, E=energy, Q_s=q, Q_over_E=q / energy,
                  flux=flux, flux_inner=flux_inner, flux_expected=flux_expected,
                  defect_flux_over_E=(flux - flux_inner - flux_expected) / energy,
                  defect_virial_over_E=(e2 + e_grad - e4_matter - e4_field + 3 * e_rappel) / energy,
                  degree=float(degree), radius_rms=float(radius_rms), s_center=float(s[0]),
                  ratio_stiffness_current_max=float(np.max(1 + parameters.eta * exp2s * ds**2 / parameters.kappa)))
    return output


def variation_second(sol, parameters, radius, elements):
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
            weights_local += w * step / 2 * r**2 * (np.outer(vf, vf) + np.outer(vs, vs))
        active = index >= 0
        indices = np.ix_(index[active], index[active])
        hessienne[indices] += local[np.ix_(active, active)]
        mass[indices] += weights_local[np.ix_(active, active)]
    values = eigh(hessienne, mass, subset_by_index=(0, 1), eigvals_only=True)
    return dict(radius_perturbations=radius, elements=elements, values_eigen=values.tolist())
