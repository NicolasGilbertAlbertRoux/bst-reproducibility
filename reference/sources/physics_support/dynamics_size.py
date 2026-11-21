'Integrate the R355 size-weighted action on independent moving site frames.'

from collections import namedtuple
from types import SimpleNamespace
from time import perf_counter

from sweep_sf4 import read
from contact_opposition_sf4 import ConditionedSF4,FINAL,REPORT
from dynamics_full import NodalKinetic
from dynamics_nodal_scripts import VACUUM,pad_cube
from solve_acceleration_nodal import constrained_inverse
from flux_nodal import ExteriorKernel
from response_of_frames import ROOT,STATE,SIZES
from release_collar import face_interior
from relax_nodes import NodalSF6
from size_dynamics import Radius
import numpy as np
from scipy.spatial.transform import Rotation


OUTPUT = ROOT/'results'/'R357'
CORE = 48
phase = namedtuple('Phase','n t x q v')
Site = namedtuple('Site','energy momentum radius weight weight_rate cusp_power edge a dv defect hdot schur')


def rotate_quaternion(q,w):
    xyz,s = q[:,:3],q[:,3:]
    return .5*np.column_stack((s*w+np.cross(xyz,w),-np.sum(xyz*w,axis=1)))


def field_inner(a,b):
    return float(np.sum(a*b))


def core_projection(n,t):
    t -= n*np.sum(n*t,axis=-1,keepdims=True)
    return t


class Molecule:
    'One full nodal field, position and SO(3) frame per molecular site.'
    def __init__(self,species='SF6',padding=0,workers=7,jet=2e-5):
        self.species = species
        self.padding = padding
        self.count = CORE+2*padding
        self.path = STATE if species=='SF6' else FINAL
        self.model = NodalSF6(workers=workers) if species=='SF6' else ConditionedSF4()
        self.pool = self.model.pool
        self.beta = self.model.model.common['beta']
        self.kernel = NodalKinetic(self.beta)
        self.radius = Radius()
        self.jet = jet
        self.law = read(SIZES)['calibration_of_size_law']
        self.p = self.law['power']
        self.ref = self.law['reference_F_support_radius_native']
        self.mask = face_interior(self.count)
        self.model.mask = self.mask if not padding else np.ones(CORE**3,dtype=bool)
        with np.load(self.path) as s:
            n,r,d,O,h = [s[k] for k in ('fields','radii_native','directions','frames','h_native')]
            self.model.model.geometry.d['r0'] = float(s['contact_intercept_native'])
        if species=='SF4':
            self.model.opposition_coefficient = read(REPORT)['contact_calibrations'][-1]['opposition_coefficient_native']
        if not np.array_equal(self.model.h,h):
            raise ValueError('The saved and evaluated native spacings differ')
        self.h = h
        self.nsites = len(n)
        self.shape = (self.nsites,self.count**3,4)
        self.nfield = int(np.prod(self.shape))
        self.n0 = self.pad(n,VACUUM)
        self.x0 = np.vstack((np.zeros(3),r[:,None]*d))
        self.q0 = Rotation.from_matrix(np.concatenate((np.eye(3)[None],O))).as_quat()
        self.exterior = {CORE: self.exterior_kernel(CORE)} if padding else {}
        self.calls = 0

    def close(self):
        self.model.close()

    def pad(self,n,fill):
        return np.array([pad_cube(a,CORE,self.padding,fill) for a in n]) if self.padding else n.copy()

    def crop(self,n,count=CORE):
        pad = (self.count-count)//2
        if not pad:
            return n
        cut = slice(pad,pad+count)
        v = n.reshape((self.nsites,)+(self.count,)*3+(4,))
        return np.ascontiguousarray(v[:,cut,cut,cut]).reshape(self.nsites,-1,4)

    def exterior_kernel(self,count):
        pad = (self.count-count)//2
        if count<CORE or pad<=0 or count+2*pad!=self.count:
            raise ValueError('A cut must contain the contact domain and leave an exterior')
        return ExteriorKernel(SimpleNamespace(padding=pad,count=self.count,c=SimpleNamespace(beta=self.beta)))

    def pack(self,n,t,x,q,v):
        return np.r_[n.ravel(),t.ravel(),x.ravel(),q.ravel(),v.ravel()]

    def unpack(self,y):
        m,k = self.nfield,self.nsites
        return phase(y[:m].reshape(self.shape),y[m:2*m].reshape(self.shape),
            y[2*m:2*m+3*k].reshape(k,3),y[2*m+3*k:2*m+7*k].reshape(k,4),y[2*m+7*k:].reshape(k,6))

    def project(self,y):
        n,t,x,q,v = self.unpack(y)
        nn = np.linalg.norm(n,axis=-1,keepdims=True)
        qq = np.linalg.norm(q,axis=-1,keepdims=True)
        if not np.isfinite(y).all() or min(float(nn.min()),float(qq.min()))<.9:
            raise ArithmeticError('The stage left the regular S3 or SO(3) chart')
        n = n/nn
        normal_rate = np.sum(n*t,axis=-1,keepdims=True)
        t = t-n*normal_rate
        n[:,~self.mask] = self.n0[:,~self.mask]
        t[:,~self.mask] = 0.
        defect = max(np.max(abs(nn-1.)),np.max(abs(qq-1.)),np.max(abs(normal_rate)))
        return self.pack(n,t,x,q/qq,v),float(defect)

    def rest(self):
        return self.pack(self.n0,np.zeros_like(self.n0),self.x0,self.q0,np.zeros((self.nsites,6)))

    def potential(self,n,x,O):
        relative_x = (x[1:]-x[0])@O[0]
        r = np.linalg.norm(relative_x,axis=1)
        d = relative_x/r[:,None]
        relative_O = O[0].T@O[1:]
        U,g = self.model.evaluate(self.crop(n),True,r=r,D=d,frames=relative_O,pose=True)
        state = self.model.last
        dg = state['direction_gradient']
        tangential = dg-d*np.sum(d*dg,axis=1)[:,None]
        F = -(state['radial_force_native'][:,None]*d+tangential/r[:,None])@O[0].T
        torque = state['frame_rotation_gradient_world']@O[0].T
        force = np.vstack((-F.sum(axis=0),F))
        tau = np.vstack((-(torque+np.cross(x[1:]-x[0],F)).sum(axis=0),torque))
        if self.padding:
            ex = self.exterior[CORE]
            outside = list(self.pool.map(lambda z:ex.potential(*z),zip(n,self.h,strict=True)))
            U += sum(a[0] for a in outside)
            g = self.pad(g,0.)+np.array([a[1] for a in outside])
        return float(U),g,force,tau,r

    def site(self,task):
        n,t,h,v,gu,load = task
        radius,cusp,edge,dr = self.radius(n,h,t)
        w = (radius/self.ref)**self.p
        log_gradient = (self.p/radius)*dr
        log_rate = field_inner(log_gradient,t)
        raw = np.r_[np.zeros(3),v]
        Q,gn,gt,gv = self.kernel.quadratic(n,h,t,raw)
        speed2 = np.sum(t*t,axis=1)[:,None]
        angle = self.jet*np.sqrt(speed2)
        cos = np.cos(angle)
        sin = self.jet*np.sinc(angle/np.pi)
        plus = self.kernel.quadratic(n*cos+t*sin,h,t*cos-n*speed2*sin,raw)
        minus = self.kernel.quadratic(n*cos-t*sin,h,t*cos+n*speed2*sin,raw)
        pt,pv = .5*gt,.5*gv[3:]
        gt_dot = (plus[2]-minus[2])/(2.*self.jet)
        gv_dot = (plus[3][3:]-minus[3][3:])/(2.*self.jet)
        coadjoint = np.r_[np.cross(pv[:3],v[:3])+np.cross(pv[3:],v[3:]),
                          np.cross(pv[3:],v[:3])]
        rn = .5*(gt_dot+log_rate*gt-gn-Q*log_gradient)+gu/w
        rb = .5*gv_dot+log_rate*pv-coadjoint-load/w
        mask = self.mask
        ni = n[mask]
        rn = core_projection(ni,rn[mask])
        A,C,G = self.kernel.mass(n,h)
        A = A[mask]
        C = np.ascontiguousarray(C[mask,:,3:])
        G = G[3:,3:]
        I = constrained_inverse(A,ni)
        H = I@C
        Cf = C.reshape(-1,6)
        D = G-Cf.T@H.reshape(-1,6)
        D = .5*(D+D.T)
        eigenvalues = np.linalg.eigvalsh(D)
        if eigenvalues[0]<=0.:
            raise ArithmeticError('The native six-dimensional site Schur block lost positivity')
        inv_rn = np.einsum('nij,nj->ni',I,rn)
        dv = np.linalg.solve(D,-rb+Cf.T@inv_rn.ravel())
        tangential = -inv_rn-H@dv
        a = np.zeros_like(n)
        a[mask] = tangential-ni*speed2[mask]
        nodal_residual = core_projection(ni,rn+np.einsum('nij,nj->ni',A,tangential)+C@dv)
        body_residual = rb+G@dv+Cf.T@tangential.ravel()
        defect = np.array([np.max(abs(nodal_residual))*w,np.max(abs(body_residual))*w])
        # Time derivative of the same kinetic energy and potential on this flow.
        T = .5*w*Q
        hdot = (field_inner(.5*w*(gn+Q*log_gradient)+gu,t)
                +w*field_inner(pt,a)+w*float(pv@dv)-float(load@v))
        cusp_power = abs(T*self.p/radius)*cusp
        return Site(T,w*pv,radius,w,w*log_rate,cusp_power,edge,a,dv,defect,hdot,w*eigenvalues[0])

    def evaluate(self,y):
        start = perf_counter()
        y,projection = self.project(y)
        s = self.unpack(y)
        O = Rotation.from_quat(s.q).as_matrix()
        U,gu,F,tau,r = self.potential(s.n,s.x,O)
        body = np.column_stack((np.einsum('nji,nj->ni',O,tau),np.einsum('nji,nj->ni',O,F)))
        sites = list(self.pool.map(self.site,zip(s.n,s.t,self.h,s.v,gu,body,strict=True)))
        T = sum(z.energy for z in sites)
        p = np.array([z.momentum for z in sites])
        P = np.einsum('nij,nj->ni',O,p[:,3:])
        J = np.einsum('nij,nj->ni',O,p[:,:3])+np.cross(s.x,P)
        xdot = np.einsum('nij,nj->ni',O,s.v[:,3:])
        qdot = rotate_quaternion(s.q,s.v[:,:3])
        acceleration = np.array([z.dv for z in sites])
        dy = self.pack(s.t,np.array([z.a for z in sites]),xdot,qdot,acceleration)
        diagnostic = {
            'H_native':T+U,'T_native':T,'U_native':U,'P_world':P.sum(axis=0),'J_world':J.sum(axis=0),
            'pose_x_native':s.x.copy(),'pose_quaternion_xyzw':s.q.copy(),'radii_native':r,
            'size_native':[z.radius for z in sites],'size_multipliers':[z.weight for z in sites],
            'size_multiplier_rates':[z.weight_rate for z in sites],
            'abs_cusp_power_bound':sum(z.cusp_power for z in sites),
            'energy_derivative':sum(z.hdot for z in sites),
            'field_and_body_EL_inf':np.max([z.defect for z in sites],axis=0),
            'edge_dot_min':min(z.edge for z in sites),
            'site_schur_min_eigenvalues':[z.schur for z in sites],
            'projection_inf':projection,'rhs_seconds':perf_counter()-start,
        }
        if (max(diagnostic['field_and_body_EL_inf'])>1e-8 or abs(diagnostic['energy_derivative'])>2e-7
                or diagnostic['abs_cusp_power_bound']>2e-8):
            raise ArithmeticError(f'Native EL, work or radius cusp gate failed: {diagnostic}')
        self.calls += 1
        return dy,diagnostic

    def step(self,y,dt,k1):
        k2,_ = self.evaluate(y+.5*dt*k1)
        k3,_ = self.evaluate(y+.5*dt*k2)
        k4,_ = self.evaluate(y+dt*k3)
        return self.project(y+(dt/6.)*(k1+2*k2+2*k3+k4))

    def reverse(self,y):
        s = self.unpack(y)
        return self.pack(s.n,-s.t,s.x,s.q,-s.v)

    def energy(self,y):
        y,_ = self.project(y)
        s = self.unpack(y)
        O = Rotation.from_quat(s.q).as_matrix()
        U,gu,F,tau,_ = self.potential(s.n,s.x,O)
        def site(z):
            n,t,h,v = z
            R,*_ = self.radius(n,h)
            return .5*(R/self.ref)**self.p*self.kernel.quadratic(n,h,t,np.r_[np.zeros(3),v])[0]
        return float(U+sum(self.pool.map(site,zip(s.n,s.t,self.h,s.v,strict=True))))

    def size_power_controls(self,y):
        s = self.unpack(y)
        rows = []
        for i in (0,1):
            n,t,h,v = s.n[i],s.t[i],self.h[i],np.r_[np.zeros(3),s.v[i]]
            R,kink,edge,g = self.radius(n,h,t)
            dR = field_inner(g,t)
            w = (R/self.ref)**self.p
            Q,gn,gt,gv = self.kernel.quadratic(n,h,t,v)
            a = -n*np.sum(t*t,axis=1)[:,None]
            naive = .5*w*(field_inner(gn,t)+field_inner(gt,a))
            extra = .5*w*Q*self.p*dR/R
            def at(eps):
                p = n+eps*t
                p /= np.linalg.norm(p,axis=1)[:,None]
                u = core_projection(p,t.copy())
                rr = self.radius(p,h)[0]
                return rr,.5*(rr/self.ref)**self.p*self.kernel.quadratic(p,h,u,v)[0]
            tests = []
            for step in (2e-4,1e-4):
                plus,minus = at(step),at(-step)
                rfd,tfd = ((a-b)/(2.*step) for a,b in zip(plus,minus,strict=True))
                tests.append({'step':step,'radius_defect':abs(rfd-dR),'kinetic_work_defect':abs(tfd-naive-extra),
                              'work_defect_without_size_term':abs(tfd-naive)})
            rows.append({'site':i,'radius':R,'directional_cusp_bound':kink,'dradius':dR,
                         'kinetic_size_work_term':extra,'tests':tests})
        return rows


def prepare_motion(model,amplitude=1.):
    s = model.unpack(model.rest())
    n,t,x,q,v = s.n.copy(),s.t.copy(),s.x.copy(),s.q.copy(),s.v.copy()
    k = len(x)
    tfield = np.indices((model.count,)*3,dtype=float).reshape(3,-1).T-.5*(model.count-1)
    for i,h in enumerate(model.h):
        xyz = h*tfield
        gate = np.maximum(1.-np.sum(xyz*xyz,axis=1)/3.5**2,0.)**3
        shape = np.sin(.31*xyz[:,0,None]-.19*xyz[:,1,None]+.71*np.arange(4)+.23*i)
        t[i] = amplitude*.035*gate[:,None]*shape
        t[i] = core_projection(n[i],t[i])
    d = x[1:]/np.linalg.norm(x[1:],axis=1)[:,None]
    offset = .017*amplitude*np.cos(.89*np.arange(k-1)+.2)
    x[1:] += d*offset[:,None]
    axes = np.sin(.5*np.arange(3*k).reshape(k,3)+.3)
    q = (Rotation.from_rotvec(.03*amplitude*axes)*Rotation.from_quat(q)).as_quat()
    v[:] = .024*amplitude*np.cos(.51*np.arange(k*6).reshape(k,6)+.3)
    return model.project(model.pack(n,t,x,q,v))[0]


def errors(model,ref,y):
    a,b = model.unpack(ref),model.unpack(y)
    qerr = (Rotation.from_quat(a.q).inv()*Rotation.from_quat(b.q)).magnitude()
    return {'field_node_max':float(np.max(np.linalg.norm(a.n-b.n,axis=-1))),
            'field_rate_node_max':float(np.max(np.linalg.norm(a.t-b.t,axis=-1))),
            'center_native_max':float(np.max(np.linalg.norm(a.x-b.x,axis=1))),
            'frame_angle_rad_max':float(np.max(qerr)),
            'body_rate_max':float(np.max(np.linalg.norm(a.v-b.v,axis=1)))}


def work_and_covariance(model,y,dy,diag):
    s,a = model.unpack(y),model.unpack(dy)
    O = Rotation.from_quat(s.q).as_matrix()
    U,gu,F,tau,_ = model.potential(s.n,s.x,O)
    Udot = field_inner(gu,s.t)-field_inner(F,a.x)-field_inner(tau,np.einsum('nij,nj->ni',O,s.v[:,:3]))
    def potential(z):
        n,t,x,q,v = model.unpack(model.project(z)[0])
        return model.potential(n,x,Rotation.from_quat(q).as_matrix())[0]
    work = []
    for eps in (2e-4,1e-4):
        f = (potential(y+eps*dy)-potential(y-eps*dy))/(2.*eps)
        e = (model.energy(y+eps*dy)-model.energy(y-eps*dy))/(2.*eps)
        work.append({'step':eps,'native_potential_flow_derivative':Udot,'potential_FD':f,'potential_defect':abs(f-Udot),
                     'Hamiltonian_FD':e,'Hamiltonian_defect':abs(e-diag['energy_derivative'])})
    R = Rotation.from_rotvec([.53,-.38,.77])
    moved = model.pack(s.n,s.t,R.apply(s.x)+np.array([.3,-.18,.55]),(R*Rotation.from_quat(s.q)).as_quat(),s.v)
    b,d2 = model.evaluate(moved)
    m = model.unpack(b)
    transformed_p = R.apply(diag['P_world'])
    expected_J = R.apply(diag['J_world'])+np.cross([.3,-.18,.55],transformed_p)
    covariance = {'H_defect':abs(d2['H_native']-diag['H_native']),'P_defect':np.linalg.norm(d2['P_world']-transformed_p),
        'J_defect':np.linalg.norm(d2['J_world']-expected_J),
        'field_acceleration_inf_defect':np.max(abs(m.t-a.t)),
        'center_rate_inf_defect':np.max(abs(m.x-R.apply(a.x))),
        'body_acceleration_inf_defect':np.max(abs(m.v-a.v))}
    if (max(z['potential_defect'] for z in work)>2e-7 or max(z['Hamiltonian_defect'] for z in work)>2e-7
            or max(covariance.values())>2e-8):
        raise ArithmeticError(f'Finite-difference work or rigid covariance failed: {work} {covariance}')
    return work,covariance
