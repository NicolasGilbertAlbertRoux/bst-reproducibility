'Calibrate a coordination-dependent port response and an angular ratio for SF6.'

import argparse
from pathlib import Path

from sweep_sf4 import read
from calibrate_curves import GIVEN_R6
from fields_and_references import dump
from trials_nodal import digest
from prepare_contact_prestressed import bare_curvature
from relax_sf6 import OUTPUT,ROOT,RADIAL,SF6,hessian,optimize
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


SOURCE = Path(__file__).resolve()
FIELDS = OUTPUT/'CHAMPS_FIXES_ANGLES_R353.json'
READING = OUTPUT/'LECTURE_FIELDS_R353.json'
FREQUENCIES = {'A1g':774.,'Eg':642.,'F2g':525.}
SHIMANOUCHI = 'https://srd.nist.gov/NSRDS/NSRDS-NBS-39.pdf'


class TransferredContact(SF6):
    def __init__(self):
        super().__init__()
        fields = read(FIELDS)['final']
        ref = read(READING)['contact_inverse_diagnostic']
        rows = np.asarray(ref['conditioned_port_rows_SF6'])
        s = .25*ref['SF6_relaxed_preparation_slope_interval']['slope_interval'][1]
        intercepts = rows[:,1:]-s*rows[:,:1]
        lo,hi = float(np.max(intercepts[:,0])),float(np.min(intercepts[:,1]))
        if not 0.<s<1. or lo>=hi:
            raise ArithmeticError('No positive damped contact response in the diagnostic window')
        center = .5*(lo+hi)
        start = np.asarray(read(FIELDS)['initial']['directions'])
        E = (start[::2]-start[1::2]).T
        U,_,Vh = np.linalg.svd(E)
        Q = U@Vh
        d = np.stack((Q.T,-Q.T),axis=1).reshape(6,3)
        self.ideal_directions = d
        if np.max(abs(d@d.T-self.geometry.target))>5e-12:
            raise ValueError('The ideal octahedron does not match the inherited ligand order')
        A = np.outer(d[0],d[2])+np.outer(d[2],d[0])
        u = d@A
        r = self.common['scale_native_per_angstrom']*GIVEN_R6
        uD = (u-d*np.sum(u*d,axis=1)[:,None])/r
        delta_gram = uD@d.T+d@uD.T
        Kr = np.array(fields['kappa_native'])+bare_curvature(self.well,r)
        radial_stretch_rayleigh = float(np.sum(Kr)/len(Kr))
        shear_rayleigh_per_k = float(np.sum(delta_gram**2)/np.sum(u**2))
        ratio = FREQUENCIES['F2g']/FREQUENCIES['A1g']
        k = ratio**2*radial_stretch_rayleigh/shear_rayleigh_per_k
        received_port = self.m.port

        def transferred(field,index,direction):
            record = received_port(field,index,direction)
            for key in ('value','gradient_field','gradient_direction'):
                record[key] *= s
            return record

        self.m.port = transferred
        self.geometry.d['r0'] = center
        self.geometry.weight = k
        # State construction and its unobserved interpolation use this same geometry object.
        self.common = {**self.common,'kangle':k}
        self.law = {'family':'intercept plus port response times a coordination-dependent slope',
            'anchors_by_coordination':{'4':{'slope':1.,'angle_stiffness_native':read(FIELDS)['common_model_constants']['kangle']},
                                        '6':{'slope':s,'intercept_native':center,'angle_stiffness_native':k}},
            'allowed_intercept_interval_at_fixed_fields':(lo,hi),
            'slope_choice':'one quarter of the positive SF6 feasibility bound, a declared margin choice',
            'angle_stiffness_calibration':{'ratio_used':ratio,'used_bands_cm_inverse':[FREQUENCIES['F2g'],FREQUENCIES['A1g']],
                'reserved_Eg_cm_inverse':FREQUENCIES['Eg'],'source':SHIMANOUCHI,
                'symmetric_stretch_stiffness_per_cartesian_displacement':radial_stretch_rayleigh,
                'octahedral_F2g_stiffness_per_unit_k':shear_rayleigh_per_k,
                'same_F_site_inertial_factor_cancels_in_this_ideal_octahedral_calibration':True},
            'interpolation_between_two_coordination_cases_not_identified':True}


def angles_on_orbit(model,rotvec):
    d = model.geometry.D0
    d1 = Rotation.from_rotvec(rotvec).apply(model.ideal_directions)
    cos = np.sum(d*d1,axis=1)
    tangent = d1-d*cos[:,None]
    sine = np.linalg.norm(tangent,axis=1)
    angle = np.arctan2(sine,cos)
    w = tangent*np.divide(angle,sine,out=np.ones_like(sine),where=sine>1e-14)[:,None]
    return np.einsum('iac,ic->ia',model.geometry.tangents,w).ravel()


def orient(model,z):
    z = z.copy()
    start = np.zeros(3)
    z[model.n:] = angles_on_orbit(model,start)
    E0 = model.profile(z,'fixed')[0]
    energy_scale = 1e-4
    def f(w):
        z[model.n:] = angles_on_orbit(model,w)
        E,g = model.profile(z,'fixed')
        d = 2e-6*np.eye(3)
        J = np.column_stack([(angles_on_orbit(model,w+v)-angles_on_orbit(model,w-v))/(4e-6) for v in d])
        return (E-E0)/energy_scale,J.T@g[model.n:]/energy_scale
    opt = minimize(f,start,method='L-BFGS-B',jac=True,bounds=[(-1.1,1.1)]*3,
                   options={'maxiter':28,'ftol':2e-9,'gtol':2e-5,'maxls':14})
    z[model.n:] = angles_on_orbit(model,opt.x)
    return z,{'rotvec_world':opt.x,'predictor_reduced_gradient':opt.jac*energy_scale,
              'trials':opt.nfev,'energy_change':opt.fun*energy_scale,
              'solver_status':int(opt.status),'active_coordinate_bound':bool(np.max(abs(opt.x))>=1.1-1e-7)}


def main(workers):
    model = TransferredContact()
    source = read(FIELDS)['final']
    z0 = np.array(source['coordinates'])[model.active]
    first = model.replay(z0)
    print('Recalibrated SF6 slope and angular stiffness',model.law['anchors_by_coordination']['6'],
          'start residuals',max(abs(first['gradient_profiled'][:model.n])),
          max(abs(first['gradient_profiled'][model.n:])),flush=True)
    model.fix_predictor(first['kappa_native'])
    z,orbit = orient(model,z0)
    print('Rigid octahedral orientation stage',orbit,flush=True)
    H,asym = hessian(model,z,np.arange(len(z)),workers)
    z,predictor = optimize(model,z,np.arange(len(z)),H,'fixed',25,'RACCORDE')
    z,corrector = optimize(model,z,np.arange(len(z)),H,'current',22,'RACCORDE')
    end = model.replay(z)
    files = (SOURCE,FIELDS,READING,RADIAL,ROOT/'scripts'/'relaxer_SF6_R353.py')
    dump(OUTPUT/'CONTACT_RECALIBRE_R353.json',{'revision':353,'full':True,'physical_validation':False,
        'law':model.law,'F_frames_held':True,'initial':first,'final':end,'orientation_predictor':orbit,
        'fixed_curvature_predictor':predictor,'full_current_curvature_corrector':corrector,
        'predictor_hessian_absolute_asymmetry':asym,'serial_evaluations':model.calls,
        'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in files}})
    print('Recalibrated SF6 complete',end['radii_native']/model.common['scale_native_per_angstrom'],
          'internal/angle defects',np.max(abs(end['gradient_profiled'][:model.n])),
          np.max(abs(end['gradient_profiled'][model.n:])),flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=5)
    main(p.parse_args().workers)
