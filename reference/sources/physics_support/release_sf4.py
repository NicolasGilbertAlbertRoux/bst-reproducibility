'Use the R354 fixed-face preparation and full contact variation for SF4.'

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from angles_internal import move,pullback
from sweep_sf4 import read
from collection_native_curves import native_fields
from fields_and_references import dump
from contacts_reactive import ReactiveContacts
from dynamics_nodal_scripts import NodalDynamics
from trials_nodal import digest
from forces_outside_map import field_energy
from operator_nodes import FieldHessian,pcg
from refine_contact import chart,rotate
from precision_sum import AccurateSumAction
from refit_dependencies import install
from refit_pose import ANGLE_TOL,Pose,relative
from release_collar import face_interior
from relax_sf6 import well_curvature_floor
from relax_nodes import (ROOT,TARGET_L2,NodalSF6,Smoother,edge_minimum,product,projected,
                                retract,two_loop)
from free_rotations import CALIBRATION
import numpy as np
from scipy.optimize import minimize


OUTPUT = ROOT/'results'/'R355'
SOURCE = Path(__file__).resolve()
R352 = ROOT/'results'/'R352'
R354 = ROOT/'results'/'R354'
ANCHOR_BANK = R352/'BANQUE_SF4_N48_R352.json'
PREP = ROOT/'results'/'INITIAL_NODAL_R345.npz'


class NodalSF4(NodalSF6):
    'Retain the shared action and the SF4 parameters before any refit.'

    def __init__(self):
        self.m = ReactiveContacts('SF4','current','mobile')
        common = read(CALIBRATION)['candidate']
        geometry = self.m.m
        install(self.m,common['beta'],common['kangle'],common['gamma'])
        self.m.action.action = AccurateSumAction(common['beta'])
        well = geometry.trial.well
        r_limits = geometry.rbase[:,None]*np.exp(self.m.state.system.bounds[self.m.nstate::3])
        floor = well_curvature_floor(well,r_limits[:,0].min(),r_limits[:,1].max())
        self.model = SimpleNamespace(common=common,geometry=geometry,m=self.m,well=well,
                                     r_limits=r_limits,well_floor=floor)
        self.n0,data,beta = native_fields(NodalDynamics(48),'SF4_N48')
        self.h = np.asarray(data['h_native'])
        self.count = round(self.n0.shape[1]**(1/3))
        np.testing.assert_allclose(self.h,[self.m.h(i) for i in range(len(self.n0))],rtol=0.,atol=1e-14)
        assert beta==common['beta'] and self.count==48
        self.mask = face_interior(self.count)
        X = np.array(data['centers'])[1:]
        self.r0 = np.linalg.norm(X,axis=1)
        self.D = X/self.r0[:,None]
        self.frames = np.array(data['frames'])[1:]
        self.m.frames = self.frames
        self.port_cache = self.ports(self.n0,self.D,self.frames)
        self.pool = ThreadPoolExecutor(max_workers=len(self.n0))
        self.evaluate(self.n0,True,r=self.r0)
        self.fixed = self.last['kappa_native'].copy()

    def evaluate(self,n,exact,r=None,D=None,frames=None,pose=False,keep=False):
        E,g = super().evaluate(n,exact,r=r,D=D,frames=frames,pose=pose,keep=keep)
        # R354 reserves storage for six fluorine torques. Here four bonds are active.
        self.last['frame_rotation_gradient_world'] = self.last['frame_rotation_gradient_world'][:len(self.D)]
        return E,g


class SF4Pose(Pose):
    def profile(self,x,law='current'):
        D0,B = self.geometry.D0,self.geometry.tangents
        w = np.einsum('nia,ni->na',B,x.reshape(len(D0),2))
        D = move(D0,w)
        O = self.frames(D,self.f)
        E,_ = self.model.evaluate(self.fields,law=='current',D=D,frames=O,pose=True)
        torque = self.model.last['frame_rotation_gradient_world']
        gd = self.model.last['direction_gradient']+np.cross(D,torque)
        grad = pullback(D0,w,gd)
        self.last = {'directions':D,'frames':O}
        return float(E),np.einsum('nia,na->ni',B,grad).ravel()

    def frame_eval(self,x,D,exact):
        w = np.einsum('nia,ni->na',self.B,x.reshape(len(D),2))
        f = move(self.f0,w)
        O = self.frames(D,f)
        E,_ = self.model.evaluate(self.fields,exact,D=D,frames=O,pose=True)
        torque = np.einsum('nji,nj->ni',O,self.model.last['frame_rotation_gradient_world'])
        gf = np.cross(torque,f)
        back = pullback(self.f0,w,gf)
        return float(E),np.einsum('nia,na->ni',self.B,back).ravel(),f


def observation(model,n,D,O,iteration,r=None):
    E,g = model.evaluate(n,True,r=r,D=D,frames=O,pose=True,keep=True)
    q = model.last
    return {'iteration':iteration,'energy_native':float(E),'field_gradient_l2':float(np.linalg.norm(g)),
            'field_gradient_inf':float(np.max(abs(g))),'radii_native':q['radii_native'],
            'radii_angstrom':q['radii_native']/model.model.common['scale_native_per_angstrom'],
            'radial_force_inf':float(max(abs(q['radial_force_native']))),
            'direction_reaction_inf':float(max(np.linalg.norm(np.cross(D,q['direction_gradient']),axis=1))),
            'frame_reaction_inf':float(max(np.linalg.norm(q['frame_rotation_gradient_world'],axis=1))),
            'contacts':q['contacts'],'angles_deg':np.degrees(np.arccos(np.clip(D@D.T,-1.,1.)))}


def descend(model,n,D,O,exact,steps):
    smooth = Smoother(n,model.h,model.model.common['beta'],model.mask)
    history,rows = [],[]
    E,g = model.evaluate(n,exact,D=D,frames=O)
    for k in range(steps):
        norm = float(np.linalg.norm(g))
        if norm<=TARGET_L2:
            break
        d = two_loop(g,n,history,smooth)
        slope = product(d,g)
        if slope>=0.:
            history.clear()
            d = -smooth.apply(n,g)
            slope = product(d,g)
        step = min(1.,.10/max(float(np.max(np.linalg.norm(d,axis=2))),1e-30))
        for _ in range(16):
            nn = retract(n,d,step,model.mask)
            try:
                e,g2 = model.evaluate(nn,exact,D=D,frames=O)
            except (ValueError,ArithmeticError):
                step *= .5
                continue
            if e<=E+1e-4*step*slope:
                break
            step *= .5
        else:
            break
        s = projected(nn,nn-n,model.mask)
        y = g2-projected(nn,g,model.mask)
        sy = product(s,y)
        if sy>1e-12*np.linalg.norm(s)*np.linalg.norm(y):
            history = [*history[-4:],(s,y,1./sy)]
        n,E,g = nn,e,g2
        rows.append({'energy_native':float(E),'field_gradient_l2':float(np.linalg.norm(g)),'step':step})
        if k<2 or (k+1)%10==0:
            print('SF4 current' if exact else 'SF4 predictor',k+1,float(E),rows[-1]['field_gradient_l2'],flush=True)
    return n,rows


def polish(model,n,D,O,steps,matvecs=32):
    E,g = model.evaluate(n,True,D=D,frames=O)
    rows = []
    for k in range(steps):
        norm = float(np.linalg.norm(g))
        if norm<=TARGET_L2:
            break
        H = FieldHessian(model,n,D=D,frames=O)
        smooth = Smoother(n,model.h,model.model.common['beta'],model.mask)
        shift = max(2e-7,min(2e-3,.007*norm))
        d,linear = pcg(H,-g,lambda a:smooth.apply(n,a),maximum=matvecs,relative_tolerance=.016,shift=shift)
        slope = product(d,g)
        if slope>=0.:
            d = -smooth.apply(n,g)
            slope = product(d,g)
        step = min(1.,.09/max(float(np.max(np.linalg.norm(d,axis=2))),1e-30))
        for _ in range(14):
            nn = retract(n,d,step,model.mask)
            try:
                e,g2 = model.evaluate(nn,True,D=D,frames=O)
            except (ValueError,ArithmeticError):
                step *= .5
                continue
            if e<=E+1e-4*step*slope or (abs(e-E)<2e-12 and np.linalg.norm(g2)<.7*norm):
                break
            step *= .5
        else:
            break
        n,E,g = nn,e,g2
        rows.append({'energy_native':float(E),'field_gradient_l2':float(np.linalg.norm(g)),
                     'step':step,'linear':linear})
        print('SF4 Newton',k+1,float(E),rows[-1]['field_gradient_l2'],linear['matvecs'],flush=True)
    return n,rows


def angular(model,n,D,O):
    P = SF4Pose(model,n,D,O)
    x = np.zeros(2*len(D))
    frame_traces = []
    E,g,_ = P.frame_eval(x,D,True)
    test = np.sin(.4*np.arange(len(x))+.23)
    test /= np.linalg.norm(test)
    fd = []
    for step in (4e-4,2e-4):
        a = P.frame_eval(x+step*test,D,True)[0]
        b = P.frame_eval(x-step*test,D,True)[0]
        fd.append(abs(float(g@test)-(a-b)/(2*step)))
    if max(fd)>1e-6*max(1.,abs(g@test)):
        raise ArithmeticError('Independent frame derivative did not replay')
    model.fixed = model.last['kappa_native'].copy()
    for exact,cap in ((False,38),(True,21)):
        base = P.frame_eval(x,D,exact)[0]
        def objective(u):
            energy,grad,_ = P.frame_eval(u,D,exact)
            return (energy-base)/.001,grad/.001
        opt = minimize(objective,x,method='L-BFGS-B',jac=True,bounds=[(-1.35,1.35)]*len(x),
                       options={'maxiter':cap,'ftol':2e-12,'gtol':2e-6,'maxls':16})
        x = opt.x
        frame_traces.append({'iterations':opt.nit,'evaluations':opt.nfev,'gradient_inf':float(max(abs(opt.jac)))*.001})
    _,_,P.f = P.frame_eval(x,D,True)
    z = chart(P,D)
    model.fixed = model.last['kappa_native'].copy()
    cols = [(P.profile(z+1e-4*v,'fixed')[1]-P.profile(z-1e-4*v,'fixed')[1])/2e-4 for v in np.eye(len(z))]
    H = np.array(cols).T
    H = (H+H.T)/2.
    z,before = relative(P,z,H)
    z,world = rotate(P,z)
    z,after = relative(P,z,H)
    P.profile(z)
    return P.last['directions'],P.last['frames'],{'frame_fd_defects':fd,'frame':frame_traces,'relative_before':before,
                                               'world':world,'relative_after':after}


def save_state(path,n,D,O,model,row):
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,fields=n,directions=D,frames=O,radii_native=row['radii_native'],h_native=model.h,
                        contact_intercept_native=float(model.model.geometry.d['r0']))
    tmp.replace(path)


def main(args):
    model = NodalSF4()
    try:
        n,D,O = model.n0,model.D,model.frames
        initial = observation(model,n,D,O,0,r=model.r0)
        log = {'initial_at_R352_geometry':initial,'cycles':[]}
        pilot = OUTPUT/'SF4_AVANT_ANGLES_R355.npz'
        record = OUTPUT/'RELAXATION_SF4_R355.json'
        if args.resume:
            with np.load(args.resume) as z:
                n,D,O = [z[k] for k in ('fields','directions','frames')]
                model.model.geometry.d['r0'] = float(z['contact_intercept_native'])
            log.update(read(record) if record.exists() else {})
            log['resumed_from'] = args.resume.name
        else:
            print('SF4 initial',initial['energy_native'],initial['field_gradient_l2'],initial['radii_angstrom'],flush=True)
            n,a = descend(model,n,D,O,False,args.predictor)
            n,b = descend(model,n,D,O,True,args.corrector)
            n,c = polish(model,n,D,O,args.newton)
            row = observation(model,n,D,O,0)
            save_state(pilot,n,D,O,model,row)
            log.update({'predictor':a,'current_corrector':b,'newton':c,'held_pose':row})
            dump(record,log)
        cycles = log['cycles']
        for k in range(args.cycles):
            D,O,a = angular(model,n,D,O)
            n,b = polish(model,n,D,O,6)
            row = observation(model,n,D,O,len(cycles)+1)
            cycles.append({'observation':row,'angular':a,'field':b})
            path = OUTPUT/'ETAT_SF4_FACES_FIXES_R355.npz'
            save_state(path,n,D,O,model,row)
            dump(record,log)
            print('SF4 joint',row['field_gradient_l2'],row['direction_reaction_inf'],row['frame_reaction_inf'],
                  row['radii_angstrom'],flush=True)
            if row['field_gradient_l2']<=TARGET_L2 and max(row['direction_reaction_inf'],row['frame_reaction_inf'])<=ANGLE_TOL:
                break
        if not args.cycles:
            row = observation(model,n,D,O,len(cycles))
            path = OUTPUT/'ETAT_SF4_FACES_FIXES_R355.npz'
            save_state(path,n,D,O,model,row)
        model.m.frames = O
        independent = float(field_energy(model.model,n,row['radii_native'],D))
        defect = abs(row['energy_native']-independent)
        norm = float(np.max(abs(np.sum(n*n,axis=2)-1.)))
        fixed = float(np.max(abs(n[:,~model.mask]-model.n0[:,~model.mask])))
        edge = edge_minimum(n,model.count)
        if defect>2e-8 or norm>1e-12 or fixed!=0. or min(edge)<=0.:
            raise ArithmeticError('SF4 independent action or constraints did not replay')
        files = [SOURCE,ANCHOR_BANK,PREP,CALIBRATION,ROOT/'scripts/relaxer_noeuds_R354.py',
                 ROOT/'scripts/operateur_noeuds_R354.py',ROOT/'scripts/recaler_pose_R354.py']
        if args.resume:
            files.append(args.resume.resolve())
        log.update({'revision':355,'full':True,'physical_validation':False,'field_count':len(n),
                    'active_nodes_per_field':int(model.mask.sum()),'intercept_native':float(model.model.geometry.d['r0']),
                    'unchanged_SF4_contact_and_angular_law':True,'reference_angles_are_the_received_Gram':True,'final':row,'field_norm_defect_inf':norm,'fixed_faces_defect_inf':fixed,
                    'adjacent_field_dot_minima':edge,'independent_energy_defect':defect,'state_filename':path.name,
                    'state_sha256':digest(path),'joint_tolerances_met':bool(row['field_gradient_l2']<=TARGET_L2 and
                    max(row['direction_reaction_inf'],row['frame_reaction_inf'])<=ANGLE_TOL),
                    'input_sha256':{str(f.relative_to(ROOT)):digest(f) for f in files}})
        dump(record,log)
    finally:
        model.close()


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--predictor',type=int,default=90)
    p.add_argument('--corrector',type=int,default=20)
    p.add_argument('--newton',type=int,default=10)
    p.add_argument('--cycles',type=int,default=3)
    p.add_argument('--resume',type=Path)
    main(p.parse_args())
