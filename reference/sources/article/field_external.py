'Outdoor spherical field, scales, and optical reading.'

from math import exp, log1p, pi, sqrt

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq, minimize_scalar


def attenuation_relative(r_over_radius, radius_over_scope):
    'Log of the ratio (a/a-N)(r) / (a/a-N)(2R), source-independent.'
    x = radius_over_scope
    return -(r_over_radius - 2) * x + log1p(r_over_radius * x) - log1p(2 * x)


def scope_finite():
    kappa, rappel = 2., 8.
    scope = sqrt(kappa / rappel)
    target = .99
    x_limit = brentq(lambda x: exp(attenuation_relative(10., x)) - target, 0., 1.)
    cases = []
    for radius in (.005, .05, .5, 5., 50., 500.):
        x = radius / scope
        log_force = attenuation_relative(10., x)
        cases.append(dict(radius=radius, radius_over_scope=x,
                        log10_ratio_acceleration=log_force / np.log(10),
                        ratio_acceleration=float(exp(log_force)),
                        stiffness_min_pour_illustration_1pct=rappel * (radius / x_limit)**2))
    # Recalculates the shape factor of a homogeneous ball by quadrature.
    deviation_factor_shape = 0.
    for x in (.01, .1, 1., 4.):
        analytic = 3 * (x * np.cosh(x) - np.sinh(x)) / x**3
        numerical = quad(lambda v: 3 * v * np.sinh(x * v) / x, 0., 1.)[0]
        deviation_factor_shape = max(deviation_factor_shape, abs(analytic / numerical - 1))
    assert deviation_factor_shape < 1e-10
    return dict(kappa=kappa, rappel=rappel, scope=scope, cases=cases,
                scope_over_radius_illustration_1pct=1 / x_limit,
                deviation_quadrature_factor_of_shape=deviation_factor_shape)


def solve_external(chi, compactness):
    "Vacuum q, k's '0. t'R/r, s'compassion'u', normalized current 1-deficit."
    def slope(t, u, deficit):
        coeff = chi * exp(2 * compactness * u) * t**4
        flux = 1 - deficit
        w = brentq(lambda w: w + coeff * w**3 - flux, 0., flux)
        return w, coeff

    def equation(t, state):
        w, coeff = slope(t, *state)
        return -w, .5 * compactness * coeff * w**4

    solution = solve_ivp(equation, (0., 1.), (0., 0.), rtol=3e-11,
                         atol=(1e-13, 1e-21), method="DOP853", dense_output=True)
    assert solution.success

    def read(distance_over_radius):
        t = 1 / distance_over_radius
        u, deficit = solution.sol(t)
        w, coeff = slope(t, u, deficit)
        # Stable expression to calculate a difference as small as 1e-14.
        return dict(r_over_R=distance_over_radius, s=float(compactness * u),
                    gradient_over_newton=float(w),
                    loss_relative_gradient=float(deficit + coeff * w**3))
    return read


def scope_long(eta, kappa):
    cases = []
    for relation in ('compactness_constant', 'density_constant'):
        for radius in (.05, .5, 5., 50., 500., 5000.):
            # The second sequence keeps E/R3 constant, with low compactness.
            compactness = 1e-3 if relation == 'compactness_constant' else 1e-3 * (radius / 5000.)**2
            chi = eta / kappa * (compactness / radius)**2
            read = solve_external(chi, compactness)
            cases.append(dict(relation=relation, R=radius, A_over_R=compactness,
                            E_asymptotic=4 * pi * kappa * radius * compactness,
                            chi=chi, readout=[read(x) for x in (1., 2., 10.)]))
    test = solve_external(0., .001)
    assert test(2.)['loss_relative_gradient'] == 0.
    assert abs(test(2.)["s"] + .0005) < 1e-12
    return dict(kappa=kappa, eta=eta, cases=cases,
                convention='Gradient of s compared to the term Newtonian. External pass with imposed asymptotic flow.')


def log_indice_and_derivative(u, reader):
    if reader == 'exponential_R409':
        return 2 * u, 2.
    if reader == 'corrected_order_2':
        return 2 * u - u**2 / 4, 2 - u / 2
    return 3 * log1p(u / 2) - log1p(-u / 2), 3 / (2 + u) + 1 / (2 - u)


def deviation(epsilon, reader):
    'Insecutes the trajectory n(r) of the radius with b fixed to infinity.'
    def equation(_, state):
        x, dx = state
        log_n, dlog_n = log_indice_and_derivative(epsilon * x, reader)
        return dx, epsilon * dlog_n * exp(2 * log_n) - x

    def output(_, state):
        return state[0]

    output.direction = -1
    output.terminal = True
    solution = solve_ivp(equation, (0., 4.), (0., 1.), events=output,
                         rtol=2.5e-14, atol=2.5e-15, method="DOP853")
    assert solution.success and len(solution.t_events[0]) == 1
    return float(solution.t_events[0][0] - pi)


def optical():
    cases = {}
    for reader, n2, n3 in (('exponential_R409', 2., 4 / 3),
                            ('corrected_order_2', 7 / 4, 5 / 6),
                            ("Schwarzschild", 7 / 4, 1.)):
        c2 = pi * (n2 + 2)
        c3 = 32 / 3 + 16 * n2 + 4 * n3
        eps = np.array([.0006, .0009, .0014, .0022, .0035, .005, .007])
        angles = np.array([deviation(e, reader) for e in eps])
        cubic = (angles - 4 * eps - c2 * eps**2) / eps**3
        extrapolated = float(np.polynomial.Polynomial.fit(eps, cubic, 3)(0.))
        assert abs(extrapolated - c3) < .003
        cases[reader] = dict(n2=n2, n3=n3, deviation_c2=c2, deviation_c3=c3,
                            deviation_c3_extrapolated=extrapolated,
                            trajectoires=[dict(GM_over_bc2=float(e), angle_rad=float(a),
                                                coefficient_order_3_finite=float(c))
                                           for e, a, c in zip(eps, angles, cubic, strict=True)])
    return cases


def orbits_of_light(eta, kappa):
    'Formal empty to rm, m-GM/c2, with a source inside.'
    results = []
    for mass in (1., 10., 100., 1000., 1e4):
        chi = 4 * eta / (kappa * mass**2)
        read = solve_external(chi, 2.)
        for corrected in (False, True):
            def radius_optical(r_over_m):
                s = read(r_over_m)["s"]
                return r_over_m * exp(-s - (s**2 / 16 if corrected else 0.))

            orbit = minimize_scalar(radius_optical, bounds=(1., 10.),
                                     method="bounded", options={"xatol": 1e-10})
            results.append(dict(m=mass, chi_a_r_m=chi, readout_corrected=corrected,
                                  orbit_in_domaine=bool(orbit.x > 1.0001),
                                  r_over_m=float(orbit.x), b_critique_over_m=float(orbit.fun)))
    return dict(cases=results, schwarzschild=3 * sqrt(3),
                asymptotic_exponential=2 * exp(1),
                asymptotic_corrected=exp(2 * (2 - sqrt(2)) - (2 - sqrt(2))**2 / 4) / (2 - sqrt(2)))
