'Matrix-free tangent Hessians of the native potential at fixed contact curvature.'

import ctypes
from pathlib import Path

from prepare_contact_prestressed import bare_curvature
from refit_dependencies import ARRAY,load_c
from relax_nodes import ROOT,product,projected,retract
import numpy as np


SOURCE = Path(__file__).with_name('hessien_nodal_R354.c')


class FieldHessian:
    def __init__(self,model,n,r=None,D=None,frames=None):
        self.n = n
        self.model = model
        self.profiled = r is None
        self.D = D
        self.frames = frames
        _,_ = model.evaluate(n,True,r=r,D=D,frames=frames,keep=True)
        model.fixed = model.last['kappa_native'].copy()
        self.r = model.last['radii_native'].copy()
        self.kr = model.fixed+np.array([bare_curvature(model.model.well,v) for v in self.r])
        self.ports = model.port_cache if D is None and frames is None else model.ports(
            n,model.D if D is None else D,model.frames if frames is None else frames)
        self.fun,self.source_hash = load_c(SOURCE,'bst_action_hessian_r354',
            [ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ARRAY,ARRAY,ARRAY],
            (ROOT/'sources'/'action_parametree_R331.c',))
        jobs = zip(n,model.h,strict=True)
        raw = np.array([w[1] for w in model.pool.map(lambda z:model.m.action.action(*z),jobs)])
        for j,row in enumerate(model.last['contacts']):
            for i,p in zip((0,j+1),self.ports[j],strict=True):
                raw[i] -= row['kappa']*row['delta_native']*p['gradient_field']
        self.normal = np.sum(raw*n,axis=2,keepdims=True)
        self.coeff = model.fixed if r is not None else model.fixed*(self.kr-model.fixed)/self.kr
        self.calls = 0

    def _one(self,task):
        field,v,h = task
        g = np.zeros_like(field)
        code = self.fun(self.model.count,h,1.,self.model.model.common['beta'],
                        np.ascontiguousarray(field),np.ascontiguousarray(v),g)
        if code:
            raise ValueError('The Hessian evaluation left the R331 field chart')
        return g

    def __call__(self,v):
        rows = self.model.pool.map(self._one,zip(self.n,v,self.model.h,strict=True))
        g = np.array(list(rows))-self.normal*v
        for j,pair in enumerate(self.ports):
            S,F = [p['gradient_field'] for p in pair]
            dv = self.coeff[j]*(product(S,v[0])+product(F,v[j+1]))
            g[0] += dv*S
            g[j+1] += dv*F
        self.calls += 1
        return projected(self.n,g,self.model.mask)

    def test(self,v,exact=False):
        v = projected(self.n,v,self.model.mask)
        v /= np.linalg.norm(v)
        calculated = self(v)
        rows = []
        for eps in (2e-4,1e-4):
            p = retract(self.n,v,eps,self.model.mask)
            m = retract(self.n,v,-eps,self.model.mask)
            r = None if self.profiled else self.r
            _,gp = self.model.evaluate(p,exact,r=r,D=self.D,frames=self.frames)
            _,gm = self.model.evaluate(m,exact,r=r,D=self.D,frames=self.frames)
            fd = projected(self.n,(gp-gm)/(2*eps),self.model.mask)
            rows.append({'step':eps,'exact_current_curvature_in_reference':exact,
                         'relative_l2_defect':float(np.linalg.norm(calculated-fd)/np.linalg.norm(fd)),
                         'directional_curvature_analytic':product(v,calculated),
                         'directional_curvature_centered':product(v,fd)})
        return rows


def pcg(apply,b,precondition,maximum=36,relative_tolerance=2e-3,shift=0.):
    x = np.zeros_like(b)
    r = b.copy()
    z = precondition(r)
    p = z.copy()
    rz = product(r,z)
    scale = float(np.linalg.norm(b))
    history = [float(np.linalg.norm(r))/max(scale,1e-40)]
    negative = False
    for _ in range(maximum):
        if history[-1]<=relative_tolerance:
            break
        hp = apply(p)+shift*p
        curvature = product(p,hp)
        if curvature<=0.:
            negative = True
            break
        a = rz/curvature
        x += a*p
        r -= a*hp
        history.append(float(np.linalg.norm(r))/max(scale,1e-40))
        if history[-1]<=relative_tolerance:
            break
        z = precondition(r)
        new_rz = product(r,z)
        p = z+(new_rz/rz)*p
        rz = new_rz
    return x,{'relative_residual_history':history,'negative_curvature_encountered':negative,
              'positive_shift_native':shift,'matvecs':len(history)-1+int(negative)}
