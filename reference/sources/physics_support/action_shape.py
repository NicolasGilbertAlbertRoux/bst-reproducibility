'Nonlinear shape inertia on the same SE(3) action and nodal variables.'
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

from sweep_sf4 import read
from dynamics_size import core_projection
from window_core import OUTPUT,WindowMolecule
from solve_acceleration_nodal import constrained_inverse
import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation

FIT = OUTPUT/'INERTIE_FORME_R360.json'


def metric(x,theta,mass):
    'Lab-frame metric of relative distance rates, holomorphic in the chart.'
    n = len(x)-1
    a = x[1:]-x[0]
    r = np.sqrt(np.einsum('ij,ij->i',a,a))
    d = a/r[:,None]
    dr = np.zeros((n,len(x),3),dtype=x.dtype)
    for i in range(n):
        dr[i,0] = -d[i]
        dr[i,i+1] = d[i]
    dr = dr.reshape(n,-1)
    mu = dr.mean(axis=0)
    dev = dr-mu
    G = theta[0]*n*np.outer(mu,mu)+theta[1]*(dev.T@dev)
    for i,j in combinations(range(n),2):
        b = a[i]-a[j]
        u = d[i]@d[j]
        w = theta[2]*(1.-u*u)+theta[3]*((1.-u)/2.)**2
        b /= np.sqrt(b@b)
        g = np.zeros_like(x)
        g[i+1],g[j+1] = b,-b
        G += w*np.outer(g.ravel(),g.ravel())/(n-1)
    return mass*G


def variations(x,xdot,theta,mass):
    G = metric(x,theta,mass)
    q = xdot.ravel()
    units = np.eye(x.size).reshape(x.size,*x.shape)
    eps = 1e-25
    dg = np.array([np.imag(metric(x+1j*eps*e,theta,mass))/eps for e in units])
    Gdot = np.einsum('i,ijk->jk',q,dg)
    T_gradient = .5*np.einsum('j,ijk,k->i',q,dg,q)
    return G,Gdot,T_gradient


class ShapeMolecule(WindowMolecule):
    def __init__(self,species):
        super().__init__(species)
        fit = read(FIT)
        self.theta = np.array(fit['parameters_SF6_fitted_and_transferred_without_changes'][:-1])
        self.mass = fit['m_ref_native']

    def mobility(self,task):
        n,h,O = task
        amp = self.amplitude(self.radius(n,h)[0])
        A,C,G = self.kernel.mass(n,h,amp)
        u = n[self.mask]
        A,C,G = A[self.mask],C[self.mask,:,3:],G[3:,3:]
        H = constrained_inverse(A,u)@C
        X = C.reshape(-1,6)
        D = G-X.T@H.reshape(-1,6)
        D = (D+D.T)*.5
        Z = np.linalg.solve(D,np.vstack((np.zeros((3,3)),O.T)))
        return SimpleNamespace(H=H,Z=Z,R=O@Z[3:],D=D)

    def evaluate(self,y):
        dy,d = super().evaluate(y)
        y,_ = self.project(y)
        s,a = self.unpack(y),self.unpack(dy)
        O = Rotation.from_quat(s.q).as_matrix()
        mobility = list(self.pool.map(self.mobility,zip(s.n,self.h,O,strict=True)))
        R = block_diag(*(m.R for m in mobility))
        G,Gdot,dT = variations(s.x,a.x,self.theta,self.mass)
        q = a.x.ravel()
        x0 = np.einsum('nij,nj->ni',O,a.v[:,3:]+np.cross(s.v[:,:3],s.v[:,3:])).ravel()
        rhs = dT-Gdot@q-G@x0
        f = np.linalg.solve(np.eye(len(G))+G@R,rhs)
        dv = np.stack([m.Z@w for m,w in zip(mobility,f.reshape(-1,3),strict=True)])
        field = a.t.copy()
        for i,m in enumerate(mobility):
            field[i,self.mask] -= m.H@dv[i]
        corrected = self.pack(a.n,field,a.x,a.q,a.v+dv)
        acc = x0+R@f
        p = G@q
        P = p.reshape(-1,3).sum(axis=0)
        J = np.cross(s.x,p.reshape(-1,3)).sum(axis=0)
        invariant = max(np.max(abs(P)),np.max(abs(J)))
        work = .5*(q@Gdot@q)+p@acc+q@f
        block = max(np.max(abs(m.D@z-np.r_[np.zeros(3),r.T@w]))
                    for m,z,r,w in zip(mobility,dv,O,f.reshape(-1,3),strict=True))
        defec = max(np.max(abs(f+G@acc+Gdot@q-dT)),block)
        if max(invariant,abs(work),defec)>2e-10:
            raise ArithmeticError(f'Shape virtual work or Schur correction failed {invariant} {work} {defec}')
        extra = .5*(q@p)
        d['H_native'] += extra
        d['T_native'] += extra
        d['P_world'] += P
        d['J_world'] += J
        d['energy_derivative'] += work
        d['shape_kinetic_energy_native'] = extra
        d['shape_force_world_l2'] = np.linalg.norm(f)
        d['shape_work_and_Schur_defects'] = [abs(work),defec,invariant]
        return corrected,d

    def energy(self,y):
        s = self.unpack(self.project(y)[0])
        q = np.einsum('nij,nj->ni',Rotation.from_quat(s.q).as_matrix(),s.v[:,3:]).ravel()
        return super().energy(y)+.5*(q@metric(s.x,self.theta,self.mass)@q)


def check_pullback(x,theta,mass):
    from inertia_shape_R360 import features,site_jet
    a = x[1:]-x[0]
    r = np.linalg.norm(a,axis=1)
    d = a/r[:,None]
    X = site_jet(r,d,5*len(r))
    J = np.vstack((np.zeros((3,X.shape[-1])),X.reshape(-1,X.shape[-1])))
    e = mass*np.einsum('i,ijk->jk',theta,features(r,d,X))
    defect = np.max(abs(J.T@metric(x,theta,mass)@J-e))
    if defect>4e-12:
        raise ArithmeticError('The nonlinear shape action differs from the spectral pencil')
    return defect
