'Condition the contact reference by the local opposition of bond directions.'

import argparse
from pathlib import Path

from sweep_sf4 import LENGTH_TARGETS,LENGTH_UNC,read
from fields_and_references import dump
from trials_nodal import digest
from ports_and_cases import radial_interval
from refit_pose import ANGLE_TOL
from release_sf4 import (NodalSF4,OUTPUT,ROOT,TARGET_L2,angular,descend,edge_minimum,observation,
                               polish,save_state)
from relax_nodes import projected,retract
import numpy as np
from scipy.optimize import linprog


SOURCE = Path(__file__).resolve()
START = OUTPUT/'SF4_AVANT_ANGLES_R355.npz'
FAIL = OUTPUT/'RELAXATION_SF4_R355.json'
RECEIVED = ROOT/'results'/'R353'/'CONTACT_RECALIBRE_R353.json'
REPORT = OUTPUT/'OPPOSITION_SF4_R355.json'
FINAL = OUTPUT/'ETAT_SF4_OPPOSITION_R355.npz'


def opposition(d):
    opposite = np.maximum(-(d@d.T),0.)
    return opposite,np.sum(opposite**4,axis=1)


class ConditionedSF4(NodalSF4):
    def __init__(self):
        self.opposition_coefficient = 0.
        self.port_slope = 1.
        super().__init__()
        self.old_target = self.model.geometry.target.copy()
        transfer = read(RECEIVED)['law']['anchors_by_coordination']['6']
        self.port_slope = transfer['slope']
        self.model.geometry.weight = transfer['angle_stiffness_native']
        # Tolles--Gwinn gas geometry as tabulated by NIST CCCBDB.
        # The two measured symmetry angles determine all six cosines.
        a,e = np.radians([173.07,101.55])/2.
        reference = np.array([[0.,np.sin(a),-np.cos(a)],[0.,-np.sin(a),-np.cos(a)],
                              [np.sin(e),0.,-np.cos(e)],[-np.sin(e),0.,-np.cos(e)]])
        self.model.geometry.target = reference@reference.T

    def ports(self,n,d,frames):
        data = super().ports(n,d,frames)
        _,chi = opposition(d)
        rows = []
        for (s,f),c in zip(data,chi,strict=True):
            p,q = [{**r,**{k:self.port_slope*r[k] for k in ('value','offset','gradient_field','gradient_direction')}}
                   for r in (s,f)]
            p['offset'] += self.opposition_coefficient*c
            rows.append((p,q))
        return rows

    def evaluate(self,n,exact,r=None,D=None,frames=None,pose=False,keep=False):
        d = self.D if D is None else np.asarray(D)
        f = self.frames if frames is None else np.asarray(frames)
        E,g = super().evaluate(n,exact,r=r,D=d,frames=f,pose=pose,keep=True)
        if pose:
            opposite,_ = opposition(d)
            rows = self.last['contacts']
            rho = -np.array([w['kappa']*w['delta_native'] for w in rows])
            weights = -4.*self.opposition_coefficient*(rho[:,None]+rho[None,:])*opposite**3
            self.last['direction_gradient'] += weights@d
        return E,g


def interval_table(model,n,d,O):
    g = model.model
    scale = g.common['scale_native_per_angstrom']
    targets = LENGTH_TARGETS*scale
    model.evaluate(n,True,r=targets,D=d,frames=O,keep=True)
    rows = model.last['contacts']
    _,chi = opposition(d)
    bounds = np.array([radial_interval(g.well,w['kappa'],z-LENGTH_UNC*scale,z+LENGTH_UNC*scale)
                       for w,z in zip(rows,targets,strict=True)])
    p = np.array([w['sulfur_port']+w['fluorine_port'] for w in rows])-model.opposition_coefficient*chi
    return bounds,chi,p/model.port_slope


def fit(model,n,d,O):
    bounds,chi,port = interval_table(model,n,d,O)
    simple = np.column_stack((np.ones(len(port)),port))
    old = linprog(np.zeros(2),A_ub=np.vstack((simple,-simple)),b_ub=np.r_[bounds[:,1],-bounds[:,0]],
                  bounds=[(None,None),(None,None)],method='highs')
    A = np.column_stack((np.ones(len(chi)),chi))
    shifted = bounds-model.port_slope*port[:,None]
    coefs = np.linalg.lstsq(A,shifted.mean(axis=1),rcond=None)[0]
    residual = A@coefs
    if np.any(residual<shifted[:,0]) or np.any(residual>shifted[:,1]):
        linear = linprog(np.zeros(2),A_ub=np.vstack((A,-A)),b_ub=np.r_[shifted[:,1],-shifted[:,0]],
                         bounds=[(None,None),(0.,None)],method='highs')
        if not linear.success:
            raise ValueError('The opposition reference has no solution for these four inputs')
        coefs = linear.x
    if coefs[1]<=0.:
        raise ValueError('The axial opposition coefficient must be positive')
    model.model.geometry.d['r0'] = float(coefs[0])
    model.opposition_coefficient = float(coefs[1])
    return {'intercept_native':float(coefs[0]),'opposition_coefficient_native':float(coefs[1]),
            'unscaled_port_sums':port,'opposition':chi,'inverse_reference_intervals':bounds,
            'simple_affine_in_raw_port_feasible_with_either_slope_sign':bool(old.success),
            'recovered_b':A@coefs+model.port_slope*port}


def independent_energy(model,n,r,d,O):
    m,g = model.m,model.model
    own = [m.action.action(field,m.h(i)) for i,field in enumerate(n)]
    error = d@d.T-g.geometry.target
    E = sum(row[0] for row in own)+.5*g.geometry.weight*np.sum(error**2)
    _,chi = opposition(d)
    for j,D in enumerate(d):
        S = m.port(n[0],0,D)
        F = m.port(n[j+1],j+1,-O[j].T@D)
        CS = m.curvature(n[0],0,S['p'],own[0])
        CF = m.curvature(n[j+1],j+1,F['p'],own[j+1])
        k = m.factor*(CS['absolute']+CF['absolute']+m.pair_curvature)
        b = float(g.geometry.d['r0'])+model.port_slope*(S['value']+F['value'])+model.opposition_coefficient*chi[j]
        E += g.well.separation_energy_derivative(r[j])[0]+.5*k*(r[j]-b)**2
    return float(E)


def checkpoint(model,n,D,O,log,row,full):
    save_state(FINAL,n,D,O,model,row)
    files = [SOURCE,START,FAIL,RECEIVED,ROOT/'scripts/relacher_SF4_R355.py']
    log.update({'revision':355,'full':full,'physical_validation':False,'final':row,
                'state_sha256':digest(FINAL),'state_filename':FINAL.name,'active_nodes_per_field':int(model.mask.sum()),
                'port_slope_transferred_from_SF6':model.port_slope,
                'angular_stiffness_transferred_from_SF6':model.model.geometry.weight,
                'Gram_calibrated_to_SF4_gas_geometry':model.model.geometry.target,
                'input_sha256':{str(f.relative_to(ROOT)):digest(f) for f in files}})
    dump(REPORT,log)


def main(a):
    model = ConditionedSF4()
    try:
        path = FINAL if a.resume else START
        with np.load(path) as z:
            n,D,O = [z[k] for k in ('fields','directions','frames')]
        log = read(REPORT) if a.resume else {'contact_calibrations':[],'joint_cycles':[]}
        if a.resume:
            last = log['contact_calibrations'][-1]
            model.model.geometry.d['r0'] = last['intercept_native']
            model.opposition_coefficient = last['opposition_coefficient_native']
            if a.refit:
                log['replay_after_port_accounting_correction'] = {
                    'previous_calibrations':len(log['contact_calibrations']),
                    'previous_cycles':len(log['joint_cycles']),
                    'previous_state_sha256':digest(path),
                    'reason':'The inherited port diagnostic includes the opposition offset. Subtract it before fitting the coefficients.',
                }
                log['contact_calibrations'].append(fit(model,n,D,O))
        else:
            log['contact_calibrations'].append(fit(model,n,D,O))
            row = observation(model,n,D,O,0)
            log['fitted_at_held_shape'] = row
            model.fixed = model.last['kappa_native'].copy()
            n,x = descend(model,n,D,O,False,a.predictor)
            n,y = descend(model,n,D,O,True,a.corrector)
            n,z = polish(model,n,D,O,a.newton)
            log.update({'predictor':x,'corrector':y,'nodal_polish':z})
            row = observation(model,n,D,O,0)
            checkpoint(model,n,D,O,log,row,False)
        for k in range(a.cycles):
            D,O,angles = angular(model,n,D,O)
            n,steps = polish(model,n,D,O,7)
            row = observation(model,n,D,O,len(log['joint_cycles'])+1)
            log['joint_cycles'].append({'pose':angles,'nodal':steps,'observation':row})
            checkpoint(model,n,D,O,log,row,False)
            error = row['radii_angstrom']-LENGTH_TARGETS
            print('SF4 opposition',row['field_gradient_l2'],row['direction_reaction_inf'],row['frame_reaction_inf'],
                  row['radii_angstrom'],flush=True)
            if max(abs(error))>LENGTH_UNC:
                if k+1<a.cycles:
                    log['contact_calibrations'].append(fit(model,n,D,O))
            elif row['field_gradient_l2']<=TARGET_L2 and max(row['direction_reaction_inf'],row['frame_reaction_inf'])<=ANGLE_TOL:
                break
        row = observation(model,n,D,O,len(log['joint_cycles']))
        E2 = independent_energy(model,n,row['radii_native'],D,O)
        edge = edge_minimum(n,model.count)
        fixed = float(np.max(abs(n[:,~model.mask]-model.n0[:,~model.mask])))
        unit = float(np.max(abs(np.sum(n*n,axis=-1)-1.)))
        defect = abs(E2-row['energy_native'])
        if fixed!=0. or unit>1e-12 or min(edge)<=0. or defect>2e-8:
            raise ArithmeticError('Constraint or independent potential replay failed')
        log.update({'fixed_faces_defect':fixed,'field_unit_defect':unit,'edge_minima':edge,
                    'independent_energy_defect':defect,'joint_thresholds_met':bool(row['field_gradient_l2']<=TARGET_L2
                    and max(row['direction_reaction_inf'],row['frame_reaction_inf'])<=ANGLE_TOL),
                    'all_lengths_in_calibration_window':bool(max(abs(row['radii_angstrom']-LENGTH_TARGETS))<=LENGTH_UNC)})
        checkpoint(model,n,D,O,log,row,True)
    finally:
        model.close()


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--predictor',type=int,default=65)
    p.add_argument('--corrector',type=int,default=14)
    p.add_argument('--newton',type=int,default=10)
    p.add_argument('--cycles',type=int,default=3)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--refit',action='store_true')
    main(p.parse_args())
