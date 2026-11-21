'Three-dimensional ring R411 from the geodetic links of R390.'

import argparse
import ctypes
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
from scipy.fft import dstn
from scipy.interpolate import CubicHermiteSpline

from topologie import preimages


AXES = (0, 1, 2)
PLANES = ((0, 1), (0, 2), (1, 2))
INTERIOR = (slice(1, -1),) * 3
ARRAY = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
ETA, KAPPA, INERTIA, G = 1000., 2., 8., .3
TARGETS = [np.array([-.57, .49, .31, -.36]), np.array([-.64, -.19, .36, .38])]


class SortieDeCarte(ValueError):
    pass


def left(*axes):
    return tuple(slice(None, -1) if i in axes else slice(None) for i in AXES)


def right(axis):
    return tuple(slice(1, None) if i == axis else slice(None) for i in AXES)


def gradient(x):
    out = np.zeros((*x.shape, 3))
    for i in AXES:
        out[left(i) + (i,)] = np.diff(x, axis=i)
    return out


def divergence(j):
    return sum(j[..., i] - np.roll(j[..., i], 1, axis=i) for i in AXES)


def curl(a):
    out = np.zeros_like(a)
    for k, (i, j) in enumerate(PLANES):
        out[left(i, j) + (k,)] = np.diff(a[left(j) + (j,)], axis=i) - np.diff(a[left(i) + (i,)], axis=j)
    return out


def curl_adjoint(b):
    out = np.zeros_like(b)
    for k, (i, j) in enumerate(PLANES):
        f = b[..., k]
        out[..., i] += f - np.roll(f, 1, axis=j)
        out[..., j] -= f - np.roll(f, 1, axis=i)
    return out


def edge_average(x):
    out = np.zeros((*x.shape, 3))
    for i in AXES:
        out[left(i) + (i,)] = (x[left(i)] + x[right(i)]) / 2
    return out


def face_average(x):
    e = edge_average(x)
    out = np.zeros_like(e)
    for k, (i, j) in enumerate(PLANES):
        f = e[..., i]
        out[left(i, j) + (k,)] = (f[left(i, j)] + f[right(j)][left(i)]) / 2
    return out


def edge_adjoint(j):
    return sum((j[..., i] + np.roll(j[..., i], 1, axis=i)) / 2 for i in AXES)


def face_adjoint(f):
    out = np.zeros(f.shape[:-1])
    for k, (i, j) in enumerate(PLANES):
        b = f[..., k]
        c = b + np.roll(b, 1, axis=i)
        out += (c + np.roll(c, 1, axis=j)) / 4
    return out


def squared_gradient(d):
    return edge_adjoint(d**2)


def charge(q, v, mass):
    p = np.einsum("...ij,...j->...i", mass, v)
    return q[..., 1] * p[..., 2] - q[..., 2] * p[..., 1]


def rotate(q, phase):
    a, b = np.cos(phase), np.sin(phase)
    out = q.copy()
    out[..., 1] = a * q[..., 1] - b * q[..., 2]
    out[..., 2] = b * q[..., 1] + a * q[..., 2]
    return out


def native(binary):
    lib = ctypes.CDLL(str(binary.resolve()))
    fn = lib.bst_links_ledger
    fn.argtypes = ([ctypes.c_int, ctypes.c_double, ctypes.c_double, ARRAY, ctypes.c_int, ctypes.c_int]
                   + [ARRAY] * 14)
    fn.restype = ctypes.c_int
    solve = lib.bst_acceleration
    solve.argtypes = [ctypes.c_int] + [ARRAY] * 7
    solve.restype = ctypes.c_int
    return fn, solve


def compile_kernel(source, binary, threshold=0.):
    binary.parent.mkdir(parents=True, exist_ok=True)
    flags = ["-O3", "-dynamiclib"] if sys.platform == "darwin" else ["-O3", "-shared", "-fPIC", "-fopenmp"]
    cc = "clang" if sys.platform == "darwin" else "gcc"
    subprocess.run([cc, *flags, f"-DR383_MIN_GEODESIC_DOT={threshold:g}",
                    str(source), "-lm", "-o", str(binary)], check=True)


def configuration(table, n, length, degree, shift=0., perturbation=True):
    rows = table[table[:, 0] == degree]
    spline = CubicHermiteSpline(rows[:, 1], rows[:, [2, 4]], rows[:, [3, 5]])
    x = np.linspace(-length/2, length/2, n)
    xyz = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1)
    r = np.linalg.norm(xyz, axis=-1)
    f, s = np.moveaxis(spline(np.maximum(r, rows[0, 1])), -1, 0)
    f[r == 0] = degree * np.pi
    angle = np.divide(np.sin(f), r, out=np.zeros_like(r), where=r>0)
    q = np.concatenate((np.cos(f)[..., None], angle[..., None] * xyz), axis=-1)
    # Non-spherical discontinuations with inner support.
    v = np.zeros_like(q)
    u = np.zeros_like(s)
    if perturbation:
        gaussian = np.exp(-r**2 / (.33 if degree == 1 else .23)**2)
        x, y, z = np.moveaxis(xyz, -1, 0)
        q[..., 0] += .006 * gaussian * (x - y) / .1
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        v[..., 1] = .015 * gaussian * z / .1
        v -= q * np.sum(q * v, axis=-1, keepdims=True)
        u[:] = .03 * gaussian * (x*y) / .05
    v_fixed = np.zeros_like(v)
    u_fixed = np.zeros_like(u)
    v_fixed[INTERIOR], u_fixed[INTERIOR] = v[INTERIOR], u[INTERIOR]
    return q, v_fixed, np.ascontiguousarray(s+shift), u_fixed


class Model:
    def __init__(self, binary, n, length, law):
        self.n, self.h, self.law = n, length/(n-1), law
        self.cell = self.h**3
        self.fn, self.solve = native(binary)
        freq = 4*np.sin(np.pi*np.arange(1, n-1)/(2*(n-1)))**2
        self.laplacian_eig = freq[:, None, None] + freq[None, :, None] + freq[None, None, :]

    def matter(self, q, v, s, u, angles, rates, static=False):
        gu, gt, adv = (np.zeros_like(q) for _ in range(3))
        mass = np.zeros((*s.shape, 4, 4))
        energy = np.zeros(3)
        source = np.zeros_like(s)
        current = np.zeros_like(angles)
        if self.law == 'common':
            log_deriv = np.full_like(s, -4.)
            weight = np.exp(-4*s)
        else:
            log_deriv, weight = np.zeros_like(s), np.ones_like(s)
        status = self.fn(self.n, self.h, 1., np.exp(2*s), 1, 2 if static else 1, q, v,
                         gu, gt, mass, adv, energy,
                         u, source, current, angles, rates, weight, log_deriv)
        if status:
            raise SortieDeCarte('a link is taken out of the geodesic map')
        return gu, gt, mass, adv, energy, source, current

    def edge_mass(self, s):
        es = edge_average(s)
        if self.law == 'common':
            return self.h/G**2 * np.exp(-2*es), np.full_like(es, -2.)
        return self.h/G**2 * np.exp(-es**2/8), -es/4

    def poisson(self, rho):
        phi = np.zeros_like(rho)
        phi[INTERIOR] = dstn(dstn(rho[INTERIOR], type=1, norm="ortho") / self.laplacian_eig,
                            type=1, norm="ortho")
        return gradient(phi)

    def initial(self, q, v, s, u):
        a = np.zeros((*s.shape, 3))
        _, _, mass, _, _, _, _ = self.matter(q, v, s, u, a, a)
        return q, v, s, u, a, self.poisson(charge(q, v, mass))

    def terms(self, state, derivatives):
        q, v, s, u, a, p = state
        m_a, dm_a = self.edge_mass(s)
        a_dot = p/m_a
        gu, gt, mass, adv, matter_energy, q_source, current = self.matter(q, v, s, u, a, a_dot)
        k_a = np.exp(2*face_average(s))/(self.h*G**2)
        b = curl(a)
        e_elec = .5*p*a_dot
        e_magn = .5*k_a*b*b
        d = gradient(s)/self.h
        z = squared_gradient(d)
        exp2 = np.exp(2*s)
        energy4s = ETA/4*exp2*z*z
        ms = INERTIA
        if self.law == 'common':
            weight_s = np.exp(-4*s)
            added = 4*ETA*weight_s*exp2*z
            ms = INERTIA*weight_s+added
        if not derivatives:
            return np.r_[matter_energy, .5*self.cell*np.sum(ms*u*u),
                         KAPPA/2*self.cell*np.sum(d*d), self.cell*np.sum(energy4s),
                         e_elec.sum(), e_magn.sum()], charge(q, v, mass)
        acceleration = np.zeros_like(v)
        if self.solve(self.n, q, v, gu, gt, mass, adv, acceleration):
            raise ValueError("masse tangentielle non positive")
        ds_forcing = (q_source + edge_adjoint(dm_a*e_elec) - 2*face_adjoint(e_magn))/self.cell
        ds_forcing += divergence((KAPPA+ETA*edge_average(exp2*z))*d)/self.h-2*energy4s
        if self.law == 'common':
            dms_ds = -4*ms+2*added
            dms_dt = dms_ds*u+8*ETA*weight_s*exp2*edge_adjoint(d*gradient(u)/self.h)
            ds_forcing += .5*dms_ds*u*u-dms_dt*u
            ds_forcing -= divergence(4*ETA*edge_average(weight_s*exp2*u*u)*d)/self.h
        sddot = np.zeros_like(s)
        sddot[INTERIOR] = (ds_forcing/ms)[INTERIOR]
        return v, acceleration, u, sddot, a_dot, current-curl_adjoint(k_a*b)

    @staticmethod
    def advance(state, rhs, time):
        q, v, s, u, a, p = (x + time*y for x, y in zip(state, rhs, strict=True))
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        v -= np.sum(q*v, axis=-1, keepdims=True)*q
        return q, v, s, u, a, p

    def step(self, state, dt):
        a = self.terms(state, True)
        b = self.terms(self.advance(state, a, dt/2), True)
        c = self.terms(self.advance(state, b, dt/2), True)
        d = self.terms(self.advance(state, c, dt), True)
        increment = tuple((x+2*y+2*z+w)/6 for x,y,z,w in zip(a,b,c,d,strict=True))
        return self.advance(state, increment, dt)

    def controls(self, state, time, topological=False):
        q, v, s, u, a, p = state
        e, rho = self.terms(state, False)
        defect = (divergence(p)+rho)[INTERIOR]
        scalar_gradient = squared_gradient(gradient(s)/self.h)
        local = ETA*np.exp(2*s)*scalar_gradient
        if self.law == 'common':
            cone = np.sqrt((KAPPA+3*local)/(INERTIA+4*local))
        else:
            cone = np.sqrt((KAPPA+3*local)/INERTIA)*np.exp(-s-s**2/16)
        data = dict(t=time, E=float(e.sum()), E_parts=e.tolist(),
                    Gauss_max=float(np.max(abs(defect))),
                    Gauss_over_rho=float(np.linalg.norm(defect)/max(np.linalg.norm(rho[INTERIOR]), 1e-30)),
                    rho_max=float(np.max(abs(rho))),
                    max_a=float(np.max(abs(a))), max_p=float(np.max(abs(p))),
                    norm_error=float(np.max(abs(np.sum(q*q, axis=-1)-1))),
                    min_s=float(np.min(s)), max_s=float(np.max(s)),
                    angle_geodesic_max=float(max(np.max(np.arccos(np.clip(
                        np.sum(q[left(i)] * rotate(q[right(i)], a[left(i)+(i,)]), axis=-1), -1, 1)))
                        for i in AXES)),
                    max_cone_scalar_over_light_static=float(np.max(cone)))
        if topological:
            data["topologie"] = [preimages(q, y, length=self.h*(self.n-1)) for y in TARGETS]
        return data


def run(binary, table, n, length, degree, law, duration, steps):
    model = Model(binary, n, length, law)
    state = model.initial(*configuration(table, n, length, degree))
    initial = state
    # Covariance test under a time-independent local transformation.
    q, v, s, u, a, p = state
    phase = .19*np.sin(np.indices(s.shape)[1]*.18)*np.cos(np.indices(s.shape)[0]*.14)
    gauged = rotate(q, phase), rotate(v, phase), s, u, a-gradient(phase), p
    eg, rg = model.terms(gauged, False)
    e0, rho = model.terms(state, False)
    covariance = dict(E=float(np.max(abs(eg-e0))/e0.sum()),
                      rho=float(np.max(abs(rg-rho))))
    assert covariance["E"] < 2e-11 and covariance["rho"] < 2e-11
    dt = duration/steps
    records = [model.controls(state, 0., True)]
    milestones = set(np.round(np.linspace(0, steps, 9)).astype(int)[1:])
    completed = True
    for i in range(1, steps+1):
        try:
            state = model.step(state, dt)
        except SortieDeCarte:
            completed = False
            records.append(model.controls(state, (i-1)*dt, True))
            break
        if i in milestones:
            records.append(model.controls(state, i*dt, i == steps))
    record = dict(N=n, L=length, h=model.h, degree=degree, law_field=law,
                  t=duration, dt=dt, step_field=steps, time_final=records[-1]["t"],
                  duration_reached=completed, covariance=covariance, trajectory=records,
                  max_drift_E=max(abs(x["E"]/records[0]["E"]-1) for x in records),
                  max_gauss_over_rho=max(x["Gauss_sur_rho"] for x in records),
                  variation_q_max=float(np.max(np.linalg.norm(state[0]-initial[0], axis=-1))),
                  variation_s_max=float(np.max(abs(state[2]-initial[2]))))
    return record, state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--entree", type=Path, required=True)
    p.add_argument("--sortie", type=Path, required=True)
    p.add_argument("--loi", choices=("R410", 'common'), required=True)
    p.add_argument("--degre", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--longueur", type=float, default=2.4)
    p.add_argument("--duree", type=float, required=True)
    p.add_argument("--pas", type=int, required=True)
    p.add_argument("--etat", type=Path)
    args = p.parse_args()
    table = np.loadtxt(args.input, delimiter=",", skiprows=1)
    r, state = run(args.binary, table, args.n, args.length_field, args.degree, args.law_field, args.duration_field, args.step_field)
    args.output.write_text(json.dumps(r, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    if args.state_field:
        n = args.n
        coords = np.linspace(-args.length_field/2, args.length_field/2, n)
        np.savez_compressed(args.state_field, x=coords, q=state[0][:, :, n//2],
                            s=state[2][:, :, n//2], u=state[3][:, :, n//2])
    print(args.output.name, "dE", r["max_derive_E"], "Gauss", r['max_gauss_over_rho'],
          "variation_q", r["variation_q_max"], "variation_s", r["variation_s_max"],
          "degres", [x["degree"] for x in r['trajectory'][-1]["topologie"]], flush=True)


if __name__ == "__main__":
    main()
