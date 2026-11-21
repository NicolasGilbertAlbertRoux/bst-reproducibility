'Relax the SF6 static chart with the received action and profile the six radii.'

import argparse
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

from angles_internal import pullback
from sweep_sf4 import read
from collection_curves import OUTPUT as R352, ROOT
from fields_and_references import dump
from contacts_reactive import ReactiveContacts
from trials_nodal import digest
from precision_sum import AccurateSumAction
from prepare_contact_prestressed import segments
from refit_dependencies import install
from free_rotations import CALIBRATION
import numpy as np
from scipy.optimize import brentq


OUTPUT = ROOT/'results'/'R353'
SOURCE = Path(__file__).resolve()
RADIAL = R352/'PORTS_MULTICAS_R352.json'
GRAD_TOL = (1e-6,1e-7)
ENERGY_ROUNDOFF = 2e-12
HESS_STEP = 4e-5
FD_MODEL = None


def well_curvature_floor(well,lower,upper):
    s = well.radius_sum
    bounds = []
    for a,b,poly in segments(well):
        l,h = max(a,lower/s),min(b,upper/s)
        if l>=h:
            continue
        curve = poly.deriv(2)/s**2
        tests = [l-a,h-a,*[t.real for t in curve.deriv().roots()
                           if abs(t.imag)<1e-11 and l-a<t.real<h-a]]
        bounds.extend(curve(tests))
    # The inherited quadratic tails have constant second derivatives.
    bounds.extend([well.left_tail_curvature/s**2,well.right_tail_curvature/s**2])
    return float(min(bounds))


class SF6:
    def __init__(self):
        self.common = read(CALIBRATION)['candidate']
        root = read(RADIAL)['relative_offset_after_conditioning_on_molecule']['SF6']
        self.offset = root['applied_offset_native']
        self.m = ReactiveContacts('SF6','current','mobile')
        c = self.common
        install(self.m,c['beta'],c['kangle'],c['gamma'])
        self.m.action.action = AccurateSumAction(c['beta'])
        self.n = self.m.nstate
        self.geometry = self.m.m
        self.anchor = self.m.state.x.copy()
        self.radial = self.n+3*np.arange(6)
        self.active = np.setdiff1d(np.arange(len(self.anchor)),self.radial)
        self.bounds = self.m.state.system.bounds.copy()
        self.anchor[self.radial] = np.log(
            np.array([z['radius_native'] for z in root['radial_replay']])/self.geometry.rbase)
        self.geometry.d['r0'] = float(self.geometry.d['r0'])+self.offset
        self.well = self.geometry.trial.well
        self.r_limits = self.geometry.rbase[:,None]*np.exp(self.bounds[self.radial])
        self.well_floor = well_curvature_floor(self.well,*[f(self.r_limits[:,i]) for f,i in ((min,0),(max,1))])
        self.calls = {'fixed':0,'current':0}
        self.last = None

    def fix_predictor(self,kappas):
        self.geometry.d['stiffness'] = np.asarray(kappas)/self.m.factor

    def profile(self,z,law='current'):
        x = self.anchor.copy()
        x[self.active] = z
        self.m.law = law
        self.m.evaluate(x)
        self.calls[law] += 1
        p = self.m.reduction
        contacts = p['contacts']
        D = p['directions']
        gd = p['Gram_gradient_direction'].copy()
        gq = p['own_state_gradient'].copy()
        E = sum(self.m.last['atomic_self_energies'])+self.m.last['Gram_energy']
        r = []
        forces = []
        curvatures = []
        b_values = []
        k_values = []
        for j,(c,(lo,hi)) in enumerate(zip(contacts,self.r_limits,strict=True)):
            b = float(self.geometry.d['r0'])+c['port_sum']
            k = c['kappa']
            if k+self.well_floor<=0.:
                raise ValueError('Radial convexity is outside the certified window')
            def f(v):
                return self.well.separation_energy_derivative(v)[1]+k*(v-b)
            v = brentq(f,lo,hi,xtol=1e-13)
            U,Wd = self.well.separation_energy_derivative(v)
            delta = v-b
            E += U+.5*k*delta**2
            gq += -k*delta*c['port_state_gradient']+.5*delta**2*c['kappa_state_gradient']
            gd[j] += -k*delta*c['port_direction_gradient']+.5*delta**2*c['kappa_direction_gradient']
            r.append(v)
            forces.append(Wd+k*delta)
            curvatures.append(k+self.well_floor)
            b_values.append(b)
            k_values.append(k)
        r = np.asarray(r)
        x[self.radial] = np.log(r/self.geometry.rbase)
        w = np.einsum('ia,iac->ic',x[self.n:].reshape(6,3)[:,1:],self.geometry.tangents)
        gD = gd-D*np.sum(D*gd,axis=1)[:,None]
        gangular = np.einsum('iac,ic->ia',self.geometry.tangents,
                            pullback(self.geometry.D0,w,gD)).ravel()
        g = np.r_[gq,gangular]
        self.last = {'energy_native':float(E),'gradient_profiled':g,'coordinates':x,
            'radii_native':r,'directions':D,'radial_force_native':forces,
            'radial_curvature_lower_bound':min(curvatures),'reference_b_native':b_values,'kappa_native':k_values}
        return float(E),g

    def replay(self,z):
        E,g = self.profile(z)
        info = self.last.copy()
        e,h = self.m.evaluate(info['coordinates'])
        radial = [r['radial_force'] for r in self.m.last['contacts']]
        if abs(E-e)>2e-9 or max(abs(g-h[self.active]))>2e-8 or max(abs(h[self.radial]))>2e-8:
            raise ArithmeticError('The profiled calculation did not reproduce the full action')
        info.update({'energy_replay_defect':abs(E-e),
            'gradient_replay_inf_defect':float(max(abs(g-h[self.active]))),
            'full_gradient':h,'full_radial_forces':radial,
            'F_frame_torque_inf':float(np.max(np.abs([r['F_frame_conjugate_torque_native'] for r in self.m.last['contacts']]))),
            'native_angles_deg':self.geometry.geometry(self.m.last)['all_pair_angles_deg'],
            'max_sulfur_turn_rad':float(np.max(abs(self.m.state.Z@z[:50]))),
            'distance_to_coordinate_bounds':float(np.min(np.minimum(info['coordinates']-self.bounds[:,0],
                                                                     self.bounds[:,1]-info['coordinates'])))})
        return info


def hessian_column(task):
    z,j,h,g0,law = task
    x = z.copy()
    x[j] += h
    _,g = FD_MODEL.profile(x,law)
    return (g-g0)/h


def hessian(model,z,indices,workers):
    global FD_MODEL
    FD_MODEL = model
    _,g = model.profile(z,'fixed')
    tasks = [(z,int(j),HESS_STEP,g,'fixed') for j in indices]
    with get_context('fork').Pool(workers) as pool:
        columns = pool.map(hessian_column,tasks,chunksize=1)
    raw = np.asarray(columns).T[indices]
    return .5*(raw+raw.T),float(np.max(abs(raw-raw.T)))


def precondition(H):
    scale = 1./np.sqrt(np.maximum(abs(np.diag(H)),1e-4))
    eigen,Q = np.linalg.eigh(scale[:,None]*H*scale[None,:])
    inverse = (Q/np.maximum(eigen,.06))@Q.T
    return scale,inverse,eigen


def optimize(model,z,indices,H,law,max_steps,name):
    scales,B,eig = precondition(H)
    E,g = model.profile(z,law)
    trace = []
    start = perf_counter()
    bfgs0 = B.copy()
    for i in range(max_steps):
        state = max(abs(g[:model.n]))
        angle = max(abs(g[model.n:]))
        if state<GRAD_TOL[0] and (len(indices)==model.n or angle<GRAD_TOL[1]):
            break
        gscaled = g[indices]*scales
        direction = np.zeros_like(z)
        direction[indices] = -scales*(B@gscaled)
        if g@direction>=0.:
            B = bfgs0.copy()
            direction[indices] = -scales*(B@gscaled)
        turn = np.max(abs(model.m.state.Z@direction[:50]))
        alpha = min(1.,.12/max(1e-30,turn),.12/max(1e-30,np.max(abs(direction[50:]))))
        accepted = False
        for attempt in range(16):
            trial = z+alpha*direction
            try:
                e,d = model.profile(trial,law)
            except (ValueError,ArithmeticError):
                alpha *= .5
                continue
            armijo = E+1e-4*float(g@(trial-z))
            stable_root_step = (abs(e-E)<=ENERGY_ROUNDOFF and
                np.linalg.norm(d[indices]*scales)<.65*np.linalg.norm(gscaled))
            if e<=armijo or stable_root_step:
                accepted = True
                break
            alpha *= .5
        if not accepted:
            break
        dx = (trial[indices]-z[indices])/scales
        dg = (d[indices]-g[indices])*scales
        ys = float(dx@dg)
        if ys>1e-11*np.linalg.norm(dx)*np.linalg.norm(dg):
            C = np.eye(len(indices))-np.outer(dx,dg)/ys
            B = C@B@C.T+np.outer(dx,dx)/ys
        z,E,g = trial,e,d
        row = {'step':i+1,'energy_native':E,'state_gradient_inf':float(max(abs(g[:model.n]))),
            'angle_gradient_inf':float(max(abs(g[model.n:]))),'step_scale':alpha,
            'radii_native':model.last['radii_native'],'elapsed_seconds':perf_counter()-start}
        trace.append(row)
        if i<4 or i%4==3:
            print(name,law,row['step'],E,row['state_gradient_inf'],row['angle_gradient_inf'],flush=True)
        if i%4==3:
            dump(OUTPUT/f'REPRISE_{name}_{law}_R353.json',{'coordinates':model.last['coordinates'],
                 'accepted_steps':len(trace),'record_full':False})
    return z,{'law':law,'iterations':len(trace),'trace':trace,'scaled_predictor_hessian_eigenvalues':eig,
              'stationarity_on_active_coordinates':bool(max(abs(g[:model.n]))<GRAD_TOL[0]
                 and (len(indices)==model.n or max(abs(g[model.n:]))<GRAD_TOL[1]))}


def main(case,workers):
    model = SF6()
    name = 'FIELDS_FIXES_ANGLES' if case=='fields' else 'FIELDS_ET_ANGLES_LIBRES'
    if case=='fields':
        x = model.anchor
        indices = np.arange(model.n)
        parent = RADIAL
    else:
        parent = OUTPUT/'CHAMPS_FIXES_ANGLES_R353.json'
        x = np.array(read(parent)['final']['coordinates'])
        indices = np.arange(len(model.active))
    z = x[model.active]
    first = model.replay(z)
    if case=='fields' and abs(first['energy_native']-839.449736868604)>2e-8:
        raise ArithmeticError('Initial R352 contact energy did not replay')
    print(name,'start',first['energy_native'],'|g|',max(abs(first['gradient_profiled'])),
          'well second derivative floor',model.well_floor,flush=True)
    model.fix_predictor(first['kappa_native'])
    H,asym = hessian(model,z,indices,workers)
    print(name,'predictor Hessian',np.linalg.eigvalsh(H)[[0,-1]],'asymmetry',asym,flush=True)
    z,predictor = optimize(model,z,indices,H,'fixed',30 if case=='fields' else 42,name)
    z,corrector = optimize(model,z,indices,H,'current',24 if case=='fields' else 36,name)
    end = model.replay(z)
    ds = [SOURCE,parent,RADIAL,CALIBRATION,ROOT/'scripts'/'contacts_reactifs_R338.py',
          ROOT/'scripts'/'cinetique_preparations_R337.py',ROOT/'scripts'/'precision_somme_R342.py',
          ROOT/'results'/'INVERSION_PREPARATIONS_R336.json']
    record = {'revision':353,'full':True,'physical_validation':False,'case':case,
        'released_indices_in_profiled_chart':indices,'F_frames_held':True,
        'stationarity_tolerances_native':GRAD_TOL,'radial_reference_offset_native':model.offset,
        'common_model_constants':{k:model.common[k] for k in ('beta','gamma','kangle','scale_native_per_angstrom')},
        'initial':first,'final':end,'fixed_curvature_predictor':predictor,'full_current_curvature_corrector':corrector,
        'predictor_hessian_absolute_asymmetry':asym,'serial_evaluations':model.calls,
        'parallel_hessian_evaluations':len(indices),'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in ds}}
    dump(OUTPUT/f'{name}_R353.json',record)
    print(name,'full',end['energy_native'],np.max(abs(end['gradient_profiled'][:model.n])),
          np.max(abs(end['gradient_profiled'][model.n:])),end['radii_native']/model.common['scale_native_per_angstrom'],flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('case',choices=('fields','all'))
    p.add_argument('--workers',type=int,default=6)
    a = p.parse_args()
    main(a.case,a.workers)
