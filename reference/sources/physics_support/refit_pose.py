'Recalibrate one shared contact intercept and release the directions and frames.'

import argparse
from pathlib import Path

from angles_internal import move,pullback
from sweep_sf4 import read
from calibrate_curves import GIVEN_R6
from fields_and_references import dump
from contacts_reactive import align
from trials_nodal import digest
from refine_contact import chart,rotate,spin_jac
from refine_nodes import refine
from ports_and_cases import radial_interval
from relax_nodes import ANCHOR,OUTPUT,ROOT,TARGET_L2,NodalSF6,edge_minimum,snapshot
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize


SOURCE = Path(__file__).resolve()
INPUT = OUTPUT/'ETAT_POLI_N48_R354.npz'
REST = OUTPUT/'NOEUDS_POLIS_R354.json'
ANGLE_TOL = 1e-7


def tangent_basis(d):
    e = np.eye(3)[np.argmin(abs(d),axis=1)]
    a = np.cross(d,e)
    a /= np.linalg.norm(a,axis=1)[:,None]
    return np.stack((a,np.cross(d,a)),axis=1)


class Pose:
    'Local fluorine reader directions and molecular bond directions are independent.'

    def __init__(self,model,n,D,O):
        self.model,self.fields = model,n
        self.geometry = model.model.geometry
        self.n = 0
        self.D0 = D.copy()
        self.O0 = O.copy()
        self.f0 = -np.einsum('nji,nj->ni',O,D)
        self.B = tangent_basis(self.f0)
        self.f = self.f0.copy()
        self.last = None

    def frames(self,D,f):
        return np.array([align(d0,d)@O@align(a,b)
                         for d0,d,O,a,b in zip(self.D0,D,self.O0,f,self.f0,strict=True)])

    def profile(self,x,law='current'):
        w = np.einsum('nia,ni->na',self.geometry.tangents,x.reshape(6,2))
        D = move(self.geometry.D0,w)
        O = self.frames(D,self.f)
        E,_ = self.model.evaluate(self.fields,law=='current',D=D,frames=O,pose=True)
        # This stored sign is the mechanical torque, the negative derivative for
        # active world-frame rotations. Its pullback cancels the F part of g_D.
        torque = self.model.last['frame_rotation_gradient_world']
        gd = self.model.last['direction_gradient']+np.cross(D,torque)
        grad = pullback(self.geometry.D0,w,gd)
        self.last = {'directions':D,'frames':O}
        return float(E),np.einsum('nia,na->ni',self.geometry.tangents,grad).ravel()

    def frame_eval(self,x,D,exact):
        w = np.einsum('nia,ni->na',self.B,x.reshape(6,2))
        f = move(self.f0,w)
        O = self.frames(D,f)
        E,_ = self.model.evaluate(self.fields,exact,D=D,frames=O,pose=True)
        torque = np.einsum('nji,nj->ni',O,self.model.last['frame_rotation_gradient_world'])
        gf = np.cross(torque,f)
        back = pullback(self.f0,w,gf)
        return E,np.einsum('nia,na->ni',self.B,back).ravel(),f

    def frame_minimum(self,D):
        x = np.zeros(12)
        baseline = self.frame_eval(x,D,True)[0]
        v = np.cos(.37*np.arange(len(x))+.17)
        v /= np.linalg.norm(v)
        predicted = float(self.frame_eval(x,D,True)[1]@v)
        fd = []
        for h in (2e-3,1e-3):
            ep = self.frame_eval(x+h*v,D,True)[0]
            em = self.frame_eval(x-h*v,D,True)[0]
            derivative = float((ep-em)/(2*h))
            fd.append({'step':h,'analytic':predicted,'finite_difference':derivative,
                       'absolute_defect':abs(predicted-derivative)})
        if max(w['absolute_defect'] for w in fd)>2e-8:
            raise ArithmeticError('Relative frame derivative did not reproduce the action')
        proof = []
        for exact,cap in ((False,26),(True,12)):
            base = self.frame_eval(x,D,exact)[0]
            def objective(u):
                energy,g,_ = self.frame_eval(u,D,exact)
                return float((energy-base)/1e-4),g/1e-4
            opt = minimize(objective,x,method='L-BFGS-B',jac=True,bounds=[(-1.1,1.1)]*12,
                           options={'maxiter':cap,'ftol':1e-11,'gtol':2e-6,'maxls':14})
            x = opt.x
            proof.append({'current_curvature':exact,'iterations':opt.nit,'evaluations':opt.nfev,
                          'final_gradient_inf_native':float(np.max(abs(opt.jac)))*1e-4,
                          'algorithmic_bounds_active':bool(np.max(abs(x))>1.1-1e-8)})
        E,grad,self.f = self.frame_eval(x,D,True)
        return {'analytic_frame_checks':fd,'two_angle_optimizers':proof,'full_coordinates':x,
                'energy_change_native':float(E-baseline),'residual_inf':float(np.max(abs(grad))),
                'rotation_about_bond_is_a_gauge':True}


def relative(pose,x,H):
    trace = []
    E,g = pose.profile(x)
    for _ in range(6):
        basis = null_space(spin_jac(pose,pose.last['directions'],np.zeros(3)).T)
        b = basis.T@g
        if np.max(abs(b))<2e-8:
            break
        dx = -basis@np.linalg.solve(basis.T@H@basis,b)
        trial = x+dx
        and_field,gt = pose.profile(trial)
        if and_field>E+3e-12 or np.linalg.norm(basis.T@gt)>=np.linalg.norm(b):
            break
        x,E,g = trial,and_field,gt
        trace.append({'energy_native':E,'gradient_inf':float(np.max(abs(g)))})
    return x,trace


def directions(pose,D):
    x = chart(pose,D)
    cols = []
    for unit in np.eye(12):
        p = pose.profile(x+1e-4*unit,'fixed')[1]
        m = pose.profile(x-1e-4*unit,'fixed')[1]
        cols.append((p-m)/(2e-4))
    H = np.array(cols).T
    H = .5*(H+H.T)
    x,before = relative(pose,x,H)
    x,world = rotate(pose,x)
    x,after = relative(pose,x,H)
    _,g = pose.profile(x)
    return pose.last['directions'],pose.last['frames'],{
        'relative_corrections_before':before,'collective_orientation':world,
        'relative_corrections_after':after,'angle_chart_force_inf':float(np.max(abs(g)))}


def calibrate(model,n,r,D,O):
    model.evaluate(n,True,r=r,D=D,frames=O,pose=True,keep=True)
    rows = model.last['contacts']
    model.fixed = model.last['kappa_native'].copy()
    scale = model.model.common['scale_native_per_angstrom']
    r0 = GIVEN_R6*scale
    dr = .003*scale
    p = [w['sulfur_port']+w['fluorine_port'] for w in rows]
    windows = [radial_interval(model.model.well,w['kappa'],r0-dr,r0+dr)-val for w,val in zip(rows,p,strict=True)]
    lo,hi = max(w[0] for w in windows),min(w[1] for w in windows)
    if not lo<=hi:
        raise ArithmeticError('The relaxed shape has no single shared intercept in the diagnostic length window')
    old = float(model.model.geometry.d['r0'])
    new = float(.5*(lo+hi))
    model.model.geometry.d['r0'] = new
    return {'old_intercept_native':old,'new_intercept_native':new,'common_interval_native':(lo,hi),
            'intervals_per_bond_native':windows,'one_constant_for_six_contacts':True,
            'contact_slope_was_held':True,'fitted_length_angstrom':GIVEN_R6,
            'diagnostic_half_width_angstrom':dr/scale}


def main(a):
    model = NodalSF6(a.workers)
    try:
        record = read(REST)
        if digest(INPUT)!=record['fields_sha256']:
            raise ArithmeticError('Wrong nodal input identity')
        with np.load(INPUT) as z:
            n,r,D,O = [z[k] for k in ('fields','radii_native','directions','frames')]
        fit = calibrate(model,n,r,D,O)
        print('One contact intercept',fit,flush=True)
        proof = []
        for k in range(a.cycles):
            pose = Pose(model,n,D,O)
            F = pose.frame_minimum(D)
            D,O,P = directions(pose,D)
            n,H = refine(model,n,a.rounds,a.matvecs,D=D,frames=O)
            E,g = model.evaluate(n,True,D=D,frames=O,pose=True,keep=True)
            angles = np.linalg.norm(np.cross(D,model.last['direction_gradient']),axis=1)
            frame = np.linalg.norm(model.last['frame_rotation_gradient_world'],axis=1)
            row = {**snapshot(E,g,model.last,k+1),'frame_reactions_inf':float(max(frame)),
                   'direction_reactions_inf':float(max(angles)),'frames':F,'directions':P,'nodal_steps':H}
            proof.append(row)
            print('Joint cycle',k+1,row['field_gradient_l2'],row['direction_reactions_inf'],
                  row['frame_reactions_inf'],row['radii_native']/model.model.common['scale_native_per_angstrom'],flush=True)
            if max(row['direction_reactions_inf'],row['frame_reactions_inf'])<ANGLE_TOL and row['field_gradient_l2']<=TARGET_L2:
                break
        f = OUTPUT/'ETAT_RECALE_N48_R354.npz'
        np.savez_compressed(f,fields=n,radii_native=model.last['radii_native'],directions=D,frames=O,h_native=model.h,
                            contact_intercept_native=fit['new_intercept_native'])
        sources = (SOURCE,INPUT,REST,ANCHOR,ROOT/'scripts'/'relaxer_noeuds_R354.py',
                   ROOT/'scripts'/'polir_noeuds_R354.py',ROOT/'scripts'/'polir_contact_R353.py')
        last = proof[-1]
        dump(OUTPUT/'POSE_RECALEE_R354.json',{'revision':354,'full':True,'physical_validation':False,
            'contact_intercept_fit':fit,'all_seven_fields_and_bond_and_F_frame_directions_released':True,
            'cycles':proof,'final':{**last,**model.last},
            'passes_declared_nodal_and_angular_tolerances':bool(
                max(last['direction_reactions_inf'],last['frame_reactions_inf'])<ANGLE_TOL and
                last['field_gradient_l2']<=TARGET_L2),
            'fixed_boundary_defect_inf':float(np.max(abs(n[:,~model.mask]-model.n0[:,~model.mask]))),
            'norm_defect_inf':float(np.max(abs(np.sum(n*n,axis=2)-1.))),
            'minimum_adjacent_field_dot_by_site':edge_minimum(n,model.count),
            'fields_filename':f.name,'fields_sha256':digest(f),
            'input_sha256':{str(q.relative_to(ROOT)):digest(q) for q in sources}})
    finally:
        model.close()


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=7)
    p.add_argument('--cycles',type=int,default=2)
    p.add_argument('--rounds',type=int,default=4)
    p.add_argument('--matvecs',type=int,default=26)
    main(p.parse_args())
