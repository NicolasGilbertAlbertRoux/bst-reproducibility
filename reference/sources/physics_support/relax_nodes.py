'Relax the seven S3 fields on the R353 contact law with profiled radii.'

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

from sweep_sf4 import read
from fields_and_references import dump
from trials_nodal import digest
from forces_outside_map import field_energy
from match_contact import TransferredContact
from relax_sf6 import ROOT
import numpy as np
from scipy.fft import dstn,idstn
from scipy.optimize import brentq


R353 = ROOT/'results'/'R353'
OUTPUT = ROOT/'results'/'R354'
ANCHOR = R353/'CONTACT_POLI_R353.json'
SOURCE = Path(__file__).resolve()
TARGET_L2 = 3e-6


def product(a,b):
    return float(np.dot(a.ravel(),b.ravel()))


def projected(field,vector,mask):
    result = vector-field*np.sum(field*vector,axis=-1,keepdims=True)
    result[:,~mask] = 0.
    return result


def retract(field,direction,step,mask):
    point = field+step*direction
    point /= np.linalg.norm(point,axis=-1,keepdims=True)
    point[:,~mask] = field[:,~mask]
    return point


class NodalSF6:
    def __init__(self,workers=6):
        self.model = TransferredContact()
        self.m = self.model.m
        self.x0 = np.array(read(ANCHOR)['final']['coordinates'])
        self.energy0,_ = self.m.evaluate(self.x0)
        self.r0 = np.array(self.m.last['radii'])
        self.D = np.array(self.m.last['D'])
        self.frames = self.m.frames.copy()
        self.n0 = np.array(self.m.state.fields)
        self.n0 /= np.linalg.norm(self.n0,axis=2,keepdims=True)
        self.count = round(self.n0.shape[1]**(1/3))
        self.mask = np.array(self.model.geometry.trial.mask,dtype=bool)
        self.h = np.array([self.m.h(i) for i in range(7)])
        self.fixed = np.array([c['kappa'] for c in self.m.last['contacts']])
        self.port_cache = self.ports(self.n0,self.D,self.frames)
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.last = None

    def close(self):
        self.pool.shutdown()

    def ports(self,n,d,frames):
        result = []
        for j,D in enumerate(d,1):
            q = (self.m.port(n[0],0,D),self.m.port(n[j],j,-frames[j-1].T@D))
            result.append(tuple({**p,'offset':p['value']-product(p['gradient_field'],n[i])}
                                for p,i in zip(q,(0,j),strict=True)))
        return result

    def _curvature(self,task):
        n,i,p,own = task
        return self.m.curvature(n,i,p,own)

    def evaluate(self,n,exact,r=None,D=None,frames=None,pose=False,keep=False):
        d = self.D if D is None else np.asarray(D)
        f = self.frames if frames is None else np.asarray(frames)
        own = list(self.pool.map(lambda v:self.m.action.action(v[0],v[1]),zip(n,self.h,strict=True)))
        raw = np.array([row[1] for row in own])
        ports = self.port_cache if D is None and frames is None and not pose else self.ports(n,d,f)
        if exact:
            tasks = [(n[i],i,port['p'],own[i]) for j,pair in enumerate(ports,1)
                     for i,port in zip((0,j),pair,strict=True)]
            curvatures = list(self.pool.map(self._curvature,tasks))
        else:
            curvatures = [None]*12
        diff = d@d.T-self.model.geometry.target
        U = sum(row[0] for row in own)+.5*self.model.geometry.weight*np.sum(diff**2)
        direction_gradient = 2.*self.model.geometry.weight*diff@d
        frame_gradient = np.zeros((6,3))
        rows = []
        radii = []
        force = []
        kvalues = []
        for j,((S,F),(lo,hi)) in enumerate(zip(ports,self.model.r_limits,strict=True)):
            i = j+1
            Cs,Cf = curvatures[2*j:2*j+2]
            if exact:
                k = self.m.factor*(Cs['absolute']+Cf['absolute']+self.m.pair_curvature)
            else:
                k = self.fixed[j]
            if k+self.model.well_floor<=0.:
                raise ValueError('Loss of positive radial curvature over the inherited interval')
            sp = S['offset']+product(S['gradient_field'],n[0])
            fp = F['offset']+product(F['gradient_field'],n[i])
            b = float(self.model.geometry.d['r0'])+sp+fp
            def g(u):
                return self.model.well.separation_energy_derivative(u)[1]+k*(u-b)
            radius = brentq(g,lo,hi,xtol=1e-13) if r is None else r[j]
            W,Wd = self.model.well.separation_energy_derivative(radius)
            delta = radius-b
            contact = .5*k*delta**2
            rho = -k*delta
            sig = .5*delta**2*self.m.factor
            raw[0] += rho*S['gradient_field']
            raw[i] += rho*F['gradient_field']
            if exact:
                raw[0] += sig*Cs['gradient_field']
                raw[i] += sig*Cf['gradient_field']
            if pose:
                dg_s = rho*S['gradient_direction']
                dg_f = rho*F['gradient_direction']
                if exact:
                    dg_s = dg_s+sig*Cs['gradient_direction']
                    dg_f = dg_f+sig*Cf['gradient_direction']
                direction_gradient[j] += dg_s-f[j]@dg_f
                df = -f[j].T@d[j]
                frame_gradient[j] = np.cross(-d[j],f[j]@dg_f)
            U += W+contact
            radii.append(radius)
            force.append(Wd+k*delta)
            kvalues.append(k)
            if keep:
                rows.append({'b_native':b,'delta_native':delta,'sulfur_port':sp,'fluorine_port':fp,
                             'kappa':k,'signed_curvatures':None if not exact else [Cs['signed'],Cf['signed']]})
        self.last = {'energy_native':float(U),'radii_native':np.array(radii),'radial_force_native':np.array(force),
                     'kappa_native':np.array(kvalues),'contacts':rows,'direction_gradient':direction_gradient,
                     'frame_rotation_gradient_world':frame_gradient,'self_energies_native':[float(v[0]) for v in own]}
        return U,projected(n,raw,self.mask)


class Smoother:
    def __init__(self,n,h,beta,mask):
        m = round(n.shape[1]**(1/3))
        self.m = m
        self.mask = mask
        k = 4.*np.sin(.5*np.pi*np.arange(1,m-1)/(m-1))**2
        self.denominator = (k[:,None,None]+k[None,:,None]+k[None,None,:]+.002)[...,None]
        self.weights = []
        for field,spacing in zip(n,h,strict=True):
            grid = field.reshape(m,m,m,4)
            rate = np.zeros(grid.shape[:-1])
            for a in range(3):
                dq = np.sum(np.diff(grid,axis=a)**2,axis=-1)/spacing**2
                L = [slice(None)]*3
                R = [slice(None)]*3
                L[a],R[a] = slice(None,-1),slice(1,None)
                rate[tuple(L)] += .5*dq
                rate[tuple(R)] += .5*dq
            self.weights.append(1./np.sqrt(spacing*(1.+beta*rate[...,None])))
        if np.any(mask.reshape(m,m,m)[[0,-1]]) or np.any(mask.reshape(m,m,m)[:,[0,-1]]) or np.any(mask.reshape(m,m,m)[:,:,[0,-1]]):
            raise ValueError('The smoothing box requires fixed outer grid faces')

    def apply(self,n,q):
        z = np.zeros_like(q)
        inside = (slice(1,-1),)*3
        for a,b,w in zip(q,z,self.weights,strict=True):
            rhs = (a.reshape(self.m,self.m,self.m,4)*w)[inside]
            mode = dstn(rhs,type=1,norm='ortho',axes=(0,1,2),workers=2)
            box = idstn(mode/self.denominator,type=1,norm='ortho',axes=(0,1,2),workers=2)
            b.reshape(self.m,self.m,self.m,4)[inside] = box*w[inside]
        return projected(n,z,self.mask)


def two_loop(gradient,n,history,smoother):
    q = gradient.copy()
    a = []
    for s,y,inv in reversed(history):
        weight = inv*product(s,q)
        q -= weight*y
        a.append(weight)
    r = smoother.apply(n,q)
    for (s,y,inv),weight in zip(history,reversed(a),strict=True):
        r += (weight-inv*product(y,r))*s
    return -projected(n,r,smoother.mask)


def snapshot(e,g,last,iteration):
    return {'iteration':iteration,'energy_native':float(e),'field_gradient_l2':float(np.linalg.norm(g)),
            'field_gradient_inf':float(np.max(abs(g))),'field_gradient_l2_by_site':np.linalg.norm(g.reshape(7,-1),axis=1),
            'radii_native':last['radii_native'],'radial_force_inf_native':float(np.max(abs(last['radial_force_native'])))}


def solve(model,n,exact,steps,goal=TARGET_L2,r=None):
    smoother = Smoother(n,model.h,model.model.common['beta'],model.mask)
    history = []
    e,g = model.evaluate(n,exact,r=r)
    trace = [snapshot(e,g,model.last,0)]
    started = perf_counter()
    for iteration in range(1,steps+1):
        if trace[-1]['field_gradient_l2']<=goal:
            break
        d = two_loop(g,n,history,smoother)
        derivative = product(g,d)
        if derivative>=0.:
            history.clear()
            d = -smoother.apply(n,g)
            derivative = product(g,d)
        step = min(1.,.11/max(float(np.max(np.linalg.norm(d,axis=2))),1e-30))
        accepted = False
        for trials in range(16):
            nn = retract(n,d,step,model.mask)
            try:
                in_field,gn = model.evaluate(nn,exact,r=r)
            except (ValueError,ArithmeticError):
                step *= .5
                continue
            if in_field<=e+1e-4*step*derivative or (abs(in_field-e)<2e-12 and np.linalg.norm(gn)<.8*np.linalg.norm(g)):
                accepted = True
                break
            step *= .5
        if not accepted:
            break
        s = projected(nn,nn-n,model.mask)
        y = gn-projected(nn,g,model.mask)
        sy = product(s,y)
        if sy>1e-12*np.linalg.norm(s)*np.linalg.norm(y):
            history.append((s,y,1./sy))
            history = history[-5:]
        n,e,g = nn,in_field,gn
        rec = snapshot(e,g,model.last,iteration)
        rec.update({'step':step,'trials':trials+1})
        trace.append(rec)
        if iteration<=2 or iteration%10==0:
            print('current' if exact else 'predictor',iteration,float(e),rec['field_gradient_l2'],
                  rec['field_gradient_inf'],round(perf_counter()-started,2),flush=True)
    return n,trace


def edge_minimum(fields,m):
    result = []
    for field in fields:
        g = field.reshape(m,m,m,4)
        result.append(min(float(np.min(np.sum(np.take(g,range(1,m),axis=a)*
                                            np.take(g,range(m-1),axis=a),axis=-1))) for a in range(3)))
    return result


def main(a):
    model = NodalSF6(a.workers)
    try:
        E,g = model.evaluate(model.n0,True,keep=True)
        start = snapshot(E,g,model.last,0)
        if abs(float(E)-model.energy0)>2e-8 or abs(start['field_gradient_l2']-21.661289054376596)>2e-6:
            raise ArithmeticError(f'R353 nodal anchor did not replay: {start}')
        print('R353 replay',start,'active nodes per field',int(model.mask.sum()),flush=True)
        n = model.n0.copy()
        inputs = [SOURCE,ANCHOR,R353/'LECTURE_POLIE_R353.json',
                  ROOT/'scripts'/'raccorder_contact_R353.py',ROOT/'scripts'/'contacts_reactifs_R338.py',
                  ROOT/'scripts'/'precision_somme_R342.py',ROOT/'sources'/'action_parametree_R331.c']
        if a.seed:
            inputs.append(a.seed.resolve())
            with np.load(a.seed) as archive:
                n = archive['fields']
            model.evaluate(n,True)
            model.fixed = model.last['kappa_native'].copy()
        n,fast = solve(model,n,False,a.predictor,goal=.5*TARGET_L2)
        n,exact = solve(model,n,True,a.current)
        E,g = model.evaluate(n,True,pose=True,keep=True)
        final = snapshot(E,g,model.last,exact[-1]['iteration'])
        r = model.last['radii_native'].copy()
        if abs(float(field_energy(model.model,n,r,model.D))-float(E))>2e-8:
            raise ArithmeticError('Independent fixed-radius energy did not replay the nodal potential')
        defect = float(np.max(abs(np.sum(n*n,axis=-1)-1.)))
        boundary = float(np.max(abs(n[:,~model.mask]-model.n0[:,~model.mask])))
        overlap = edge_minimum(n,model.count)
        if defect>1e-12 or boundary!=0. or min(overlap)<=0.:
            raise ArithmeticError('The field left the inherited nodal constraints')
        target = OUTPUT/f'ETAT_{a.label}_R354.npz'
        np.savez_compressed(target,fields=n,radii_native=r,directions=model.D,frames=model.frames,h_native=model.h)
        report = {'revision':354,'full':True,'physical_validation':False,'label':a.label,
            'grid_count_per_axis':model.count,'independent_nodes_per_field':int(model.mask.sum()),
            'active_field_dofs':int(7*model.mask.sum()*3),
            'directions_and_fluorine_frames_held':True,'unchanged_R353_contact_and_energy_law':True,
            'beta':model.model.common['beta'],'scale_native_per_angstrom':model.model.common['scale_native_per_angstrom'],
            'initial_R353':start,'fixed_curvature_predictor':fast,'current_curvature_corrector':exact,
            'final':{**final,**model.last},'target_nodal_gradient_l2':TARGET_L2,
            'passes_nodal_gradient_target':bool(final['field_gradient_l2']<=TARGET_L2),
            'norm_defect_inf':defect,'fixed_boundary_defect_inf':boundary,'minimum_adjacent_field_dot_by_site':overlap,
            'fields_sha256':digest(target),'fields_filename':target.name,
            'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in inputs}}
        dump(OUTPUT/f'RELAXATION_{a.label}_R354.json',report)
        print('Nodal replay complete',final,flush=True)
    finally:
        model.close()


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--label',default='N48')
    p.add_argument('--seed',type=Path)
    p.add_argument('--predictor',type=int,default=150)
    p.add_argument('--current',type=int,default=32)
    p.add_argument('--workers',type=int,default=7)
    main(p.parse_args())
