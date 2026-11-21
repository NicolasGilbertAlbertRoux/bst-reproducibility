'Energy ledger at two cuts of the same shape action and two exterior thicknesses.'
import argparse
from pathlib import Path

from action_shape import FIT,ShapeMolecule
from sweep_sf4 import read
from fields_and_references import dump
from dynamics_nodal_scripts import VACUUM,pad_cube
from trials_nodal import digest
from window_core import OUTPUT,ROOT,WindowMolecule,window
from flux_mobile import CUTS,ledger,prepare_pulse
from halo_local import PREP,PREP_LOG,fd_controls,flux
from trajectories_moving import timeline
import numpy as np
from scipy.spatial.transform import Rotation

DURATION = .36
DT = .06


class ShapeHalo(ShapeMolecule):
    def __init__(self,padding):
        WindowMolecule.__init__(self,'SF6',padding=padding)
        fit = read(FIT)
        self.theta = np.asarray(fit['parameters_SF6_fitted_and_transferred_without_changes'][:-1])
        self.mass = fit['m_ref_native']
        self.exterior[50] = self.exterior_kernel(50)
        if digest(PREP)!=read(PREP_LOG)['field_sha256']:
            raise ArithmeticError('Changed halo reference')
        with np.load(PREP) as a:
            field = a['fields']
        if padding==6:
            field = np.stack([pad_cube(n,56,2,VACUUM) for n in field])
        self.n0 = field


def outside_momentum(m,y,cut):
    s = m.unpack(m.project(y)[0])
    def site(z):
        n,h,t,v = z
        p = m.exterior[cut].quadratic(n,h,t,np.r_[np.zeros(3),v])[-1]
        return .5*p[3:]
    p = np.array(list(m.pool.map(site,zip(s.n,m.h,s.t,s.v,strict=True))))
    O = Rotation.from_quat(s.q).as_matrix()
    P = np.einsum('nij,nj->ni',O,p[:,3:])
    J = np.einsum('nij,nj->ni',O,p[:,:3])+np.cross(s.x,P)
    return np.r_[P.sum(axis=0),J.sum(axis=0)]


def run(padding):
    m = ShapeHalo(padding)
    results = {}
    try:
        for pulse in (False,True):
            y = prepare_pulse(m) if pulse else m.rest()
            case = 'pulse' if pulse else 'release_only'
            def observed(y,dy,d,k):
                d['cuts'] = {str(i):flux(m,y,dy,d,i) for i in CUTS}
                for i in CUTS:
                    if .5*(i-1)*min(m.h)<=window()[1]+max(m.h):
                        raise ArithmeticError('A measured cut intersects material loading')
                    d['cuts'][str(i)]['Noether_exterior_P_and_J'] = outside_momentum(m,y,i)
                if k in (0,round(DURATION/DT)):
                    d['FD_energy_checks'] = fd_controls(m,y,dy,d)
            _,rows = timeline(m,y,DURATION,DT,observed,log=f'N{m.count} {case}')
            results[case] = {'ledger':ledger(rows),'history':rows,
                    'global_energy_drift':max(abs(r['H_native']-rows[0]['H_native']) for r in rows),
                    'global_P_drift':max(np.linalg.norm(r['P_world']-rows[0]['P_world']) for r in rows),
                    'global_J_drift':max(np.linalg.norm(r['J_world']-rows[0]['J_world']) for r in rows)}
        matched = {}
        for i in map(str,CUTS):
            wave,release = (results[k]['ledger'][i] for k in ('pulse','release_only'))
            p0,p1 = (results['pulse']['history'][j]['cuts'][i]['Noether_exterior_P_and_J'] for j in (0,-1))
            r0,r1 = (results['release_only']['history'][j]['cuts'][i]['Noether_exterior_P_and_J'] for j in (0,-1))
            matched[i] = {'pulse_only_energy_increase':wave['energy_increase_native']-release['energy_increase_native'],
                          'pulse_only_integrated_power':wave['integrated_power_native']-release['integrated_power_native'],
                          'exterior_P_and_J_net_after_release_subtraction':(p1-p0)-(r1-r0)}
        files = (Path(__file__),FIT,PREP,PREP_LOG,Path(__file__).with_name('action_forme_R360.py'))
        dump(OUTPUT/f'FLUX_N{m.count}_R360.json',{'revision':360,'full':True,'physical_validation':False,
              'scope':'outward energy ledger on two material cuts with matched field release and finite held outer faces',
              'outer_node_count':m.count,'native_time':DURATION,'dt_native':DT,'cut_grids':CUTS,
              'same_N56_preparation_embedded_with_two_extra_vacuum_layers':padding==6,
              'matched_pulse_minus_release':matched,'results':results,
              'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in files}})
        print('outward',m.count,matched,flush=True)
    finally:
        m.close()


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('padding',type=int,choices=(4,6))
    run(p.parse_args().padding)
