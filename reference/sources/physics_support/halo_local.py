'Matched pulse and reference on the localized action with a prepared halo.'

import argparse
from pathlib import Path

from sweep_sf4 import read
from fields_and_references import dump
from kinetic_local import LocalKinetic,LocalRadius
from dynamics_size import CORE,Molecule,field_inner
from trials_nodal import digest
from flux_mobile import CUTS,ledger,outside,prepare_pulse
from motion_local import CALIBRATION,GATE,OUTPUT,ROOT,LocalMolecule
from relax_nodes import projected,solve
from trajectories_moving import timeline
import numpy as np
from scipy.spatial.transform import Rotation

PREP = OUTPUT/'HALO_PREPARE_R358.npz'
PREP_LOG = OUTPUT/'PREPARATION_HALO_R358.json'
DURATION = .36
CASES = {'wave_fin':(.03,True),'control_fin':(.03,False),'wave_grossier':(.06,True)}


class Halo(LocalMolecule):
    def __init__(self):
        Molecule.__init__(self,'SF6',padding=4)
        law = next(row for row in read(CALIBRATION)['gate_catalog_all_results']
                   if tuple(row['thresholds'])==GATE)
        self.p,self.ref = (law[k] for k in ('exponent','reference_radius_native'))
        self.kernel,self.radius = LocalKinetic(self.beta,*GATE),LocalRadius(*GATE)
        self.exterior[50] = self.exterior_kernel(50)


class HeldCore:
    def __init__(self,flow):
        self.flow = flow
        self.model = flow.model.model
        self.h = flow.h
        self.O = Rotation.from_quat(flow.q0).as_matrix()
        self.mask = flow.mask.copy()
        p = (flow.count-44)//2
        self.mask.reshape((flow.count,)*3)[(slice(p,p+44),)*3] = False

    def evaluate(self,n,exact,r=None):
        U,g,F,tau,r = self.flow.potential(n,self.flow.x0,self.O)
        self.last = self.flow.model.last
        return U,projected(n,g,self.mask)


def prepare():
    m = Halo()
    try:
        held = HeldCore(m)
        n,trace = solve(held,m.n0,True,26,goal=3e-6)
        temp = PREP.with_name(PREP.stem+'.tmp.npz')
        np.savez_compressed(temp,fields=n)
        temp.replace(PREP)
        if trace[-1]['field_gradient_l2']>=trace[0]['field_gradient_l2']:
            raise ArithmeticError('Halo relaxation did not reduce the free field residual')
        dump(PREP_LOG,{'revision':358,'full':True,'physical_validation':False,
             'scope':'N56 halo field and old face relaxation, inner N44 and all site poses held',
             'trace':trace,'active_preparation_nodes':int(held.mask.sum()),'field_sha256':digest(PREP),
             'input_sha256':{str(f.relative_to(ROOT)):digest(f) for f in (Path(__file__),m.path,CALIBRATION)}})
        print('Halo relaxation U and residual',trace[0]['energy_native']-trace[-1]['energy_native'],
              trace[0]['field_gradient_l2'],trace[-1]['field_gradient_l2'],flush=True)
    finally:
        m.close()


def outer_energy(m,y,cut):
    s = m.unpack(m.project(y)[0])
    ex = m.exterior[cut]
    def site(z):
        n,t,v,h = z
        q,*_ = ex.quadratic(n,h,t,np.r_[np.zeros(3),v])
        U,_ = ex.potential(n,h)
        return float(.5*q+U)
    return sum(m.pool.map(site,zip(s.n,s.t,s.v,m.h,strict=True)))


def flux(m,y,dy,d,cut):
    s = m.unpack(y)
    side = (m.count-cut)//2
    mask = np.ones((m.count,)*3,dtype=bool)
    mask[(slice(side+1,side+cut-1),)*3] = False
    minimum = np.min(s.n[:,mask.ravel(),0])
    if minimum<=GATE[1]:
        raise ArithmeticError(f'The measured exterior overlaps active kinetic loading at cut {cut}: {minimum}')
    local = {**d,'size_multipliers':np.ones(m.nsites),'size_multiplier_rates':np.zeros(m.nsites)}
    return {**outside(m,y,dy,local,cut),'minimum_exterior_field_scalar':minimum}


def fd_controls(m,y,dy,d):
    checks = []
    for cut in CUTS:
        p = d['cuts'][str(cut)]['power_native']
        for e in (2e-4,1e-4):
            fd = (outer_energy(m,y+e*dy,cut)-outer_energy(m,y-e*dy,cut))/(2.*e)
            if abs(p-fd)>2e-7:
                raise ArithmeticError('The exterior field/rigid power did not reproduce dH_ext/dt')
            checks.append({'cut':cut,'step':e,'power_absolute_defect':abs(p-fd)})
    return checks


def run(case):
    dt,pulse = CASES[case]
    m = Halo()
    try:
        with np.load(PREP) as a:
            m.n0 = a['fields']
        if digest(PREP)!=read(PREP_LOG)['field_sha256']:
            raise ArithmeticError('Prepared halo changed')
        y = prepare_pulse(m) if pulse else m.rest()
        def observed(y,dy,d,k):
            d['cuts'] = {str(i):flux(m,y,dy,d,i) for i in CUTS}
            if k in (0,round(DURATION/dt)):
                d['FD_energy_checks'] = fd_controls(m,y,dy,d)
        _,rows = timeline(m,y,DURATION,dt,observed,log=case)
        out = {'revision':358,'full':True,'physical_validation':False,'case':case,
              'dt_native':dt,'pulse':pulse,'duration_native':DURATION,'cut_grids':CUTS,
              'local_weight_one_on_every_sampled_exterior_base':True,'relaxed_exterior_held_deep_core':True,
              'finite_held_outer_faces_not_an_infinite_open_boundary':True,
              'maximum_H_drift_native':max(abs(r['H_native']-rows[0]['H_native']) for r in rows),
              'ledger':ledger(rows),'history':rows,
              'input_sha256':{str(f.relative_to(ROOT)):digest(f) for f in (Path(__file__),PREP,PREP_LOG)}}
        dump(OUTPUT/f'HALO_LOCAL_{case.upper()}_R358.json',out)
        print(case,'ledgers',out['ledger'],flush=True)
    finally:
        m.close()


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('case',choices=['prepare',*CASES])
    a = p.parse_args()
    if a.case=='prepare':
        prepare()
    else:
        run(a.case)
