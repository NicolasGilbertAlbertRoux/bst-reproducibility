'Checks mode rate, light speeds and deviation.'

import argparse
import json
from math import exp, pi, sqrt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import least_squares, minimize_scalar

from field_external import optical, solve_external
from dynamics import Model, compile_kernel, curl, curl_adjoint, divergence, left
from modes_radial import modes
from stiffness_radial import Parameters


def modes_rescales(table):
    data = []
    for deg in (1,2):
        rows = table[table[:,0]==deg]
        interp = CubicHermiteSpline(rows[:,1],rows[:,[2,4]],rows[:,[3,5]])
        for common in (False,True):
            for elements in (128,256):
                for delta in (0.,.12,-.12):
                    d = exp(delta)
                    def sol(r):
                        f,s = interp(r/d).T
                        df,ds = interp(r/d,1).T
                        return np.array([f,df/d,s+delta,ds/d])
                    omega = modes(SimpleNamespace(x=rows[:,1]*d,sol=sol),
                                  Parameters(eta=1000.), 6.*d, elements, common)
                    if delta == 0.:
                        base = omega
                    ratio = omega/base
                    predicted = d if common else 1/d
                    assert max(abs(ratio-predicted)) < 2e-8
                    data.append(dict(degree=deg, law_field='common' if common else "R410",
                                     delta_s=delta, elements=elements, omega=omega.tolist(),
                                     ratio_computed=ratio.tolist(), ratio_expected=predicted))
    return data


def test_waves(binary, law, s0, steps):
    n, length, mode = 11, 1., 3
    grid = np.arange(1, n, dtype=float)
    m = Model(binary,n,length,law)
    q = np.zeros((n,n,n,4))
    q[...,0] = 1.
    v = np.zeros_like(q)
    s = np.full((n,n,n),s0)
    u = np.zeros_like(s)
    flux = np.zeros((n,n,n,3))
    flux[left(0,1)+(0,)] = (np.sin(pi*mode*grid/n)[:,None,None]
                            *np.sin(2*pi*grid/n)[None,:,None])
    a = curl_adjoint(flux)
    a *= 2e-6/np.max(abs(a))
    p = np.zeros_like(a)
    norm = np.sum(a*a)
    # No initial electrical pulse. The gauge is resolved with q and s free.
    state = (q,v,s,u,a,p)
    c = exp(2*s0) if law=='common' else exp(s0+s0*s0/16)
    eigenvalue = 4*(np.sin(pi*mode/(2*n))**2+np.sin(pi/n)**2)
    assert np.max(abs(curl_adjoint(curl(a))-eigenvalue*a)) < 2e-19
    assert np.max(abs(divergence(a))) < 2e-20
    omega = c*np.sqrt(eigenvalue)/m.h
    t = np.linspace(0.,4*pi/omega,steps+1)
    observations = [1.]
    for _ in t[1:]:
        state = m.step(state,t[1])
        observations.append(float(np.sum(state[4]*a)/norm))
    y = np.asarray(observations)
    fit = least_squares(lambda x: np.cos(x[0]*omega*t)-y,[1.],gtol=1e-14,xtol=1e-14,ftol=1e-14)
    error = abs(fit.x[0]-1)
    assert error < 2e-5
    return dict(law_field=law, s_uniforme=s0, step_field=steps, speed_theorique=c,
                speed_computed=float(c*fit.x[0]), error_relative=float(error),
                variation_s_max=float(np.max(abs(state[2]-s0))),
                time=t[::4].tolist(), amplitude=y[::4].tolist())


def field_fort():
    result = []
    for mass in (10.,100.,1000.,10000.):
        read = solve_external(500./mass**2,1.)
        orbit = minimize_scalar(lambda x: x*exp(-2*read(x)["s"]), bounds=(1.,10.),
                                method="bounded",options={"xatol":1e-10})
        result.append(dict(GM_over_c2=mass,r_over_m=float(orbit.x),b_over_m=float(orbit.fun)))
    return dict(cases=result, b_critique_asymptotic=2*exp(1), schwarzschild=3*sqrt(3))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entree",type=Path,required=True)
    p.add_argument("--binary",type=Path,required=True)
    p.add_argument("--sortie",type=Path,required=True)
    a = p.parse_args()
    if not a.binary.is_file():
        compile_kernel(Path(__file__).with_name("liens_jauge_R411.c"),a.binary)
    table = np.loadtxt(a.input,delimiter=",",skiprows=1)
    mode_data = modes_rescales(table)
    print('q-modes measured in homologous boxes',flush=True)
    data = dict(modes=mode_data, waves=[test_waves(a.binary,law,s0,n)
                for law in ("R410",'common') for s0 in (-.12,0.,.12) for n in (100,200)],
                lens=optical(), field_fort=field_fort())
    a.output.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print('Transverse speeds and integration of controlled rays',flush=True)


if __name__=="__main__":
    main()
