'Resolve the nearly flat octahedral orientation separately from the stiff relative angles.'

from pathlib import Path

from sweep_sf4 import read
from fields_and_references import dump
from trials_nodal import digest
from modes_radial import CORRECTED
from match_contact import TransferredContact
from relax_sf6 import OUTPUT,ROOT,hessian
import numpy as np
from scipy.linalg import block_diag,null_space
from scipy.optimize import root
from scipy.spatial.transform import Rotation


def chart(model,dirs):
    d = model.geometry.D0
    cos = np.sum(d*dirs,axis=1)
    w = dirs-d*cos[:,None]
    n = np.linalg.norm(w,axis=1)
    a = np.arctan2(n,cos)
    w *= np.divide(a,n,out=np.ones_like(a),where=n>1e-14)[:,None]
    return np.einsum('iac,ic->ia',model.geometry.tangents,w).ravel()


def spin(model,dirs,v):
    return chart(model,Rotation.from_rotvec(v).apply(dirs))


def spin_jac(model,dirs,v):
    increments = 1e-6*np.eye(3)
    return np.column_stack([(spin(model,dirs,v+d)-spin(model,dirs,v-d))/(2e-6)
                            for d in increments])


def relative(model,z,H):
    trace = []
    E,g = model.profile(z)
    for _ in range(6):
        dirs = model.last['directions']
        J = spin_jac(model,dirs,np.zeros(3))
        basis = block_diag(np.eye(model.n),null_space(J.T))
        projected = basis.T@g
        if np.max(abs(projected))<2e-8:
            break
        d = -basis@np.linalg.solve(basis.T@H@basis,projected)
        alpha = 1.
        accepted = False
        for _ in range(5):
            trial = z+alpha*d
            e,r = model.profile(trial)
            if e<=E+3e-12 and np.linalg.norm(basis.T@r)<.9*np.linalg.norm(projected):
                accepted = True
                break
            alpha *= .5
        if not accepted:
            break
        z,E,g = trial,e,r
        trace.append({'energy_native':E,'field_force_inf':float(np.max(abs(g[:model.n]))),
                      'angular_force_inf':float(np.max(abs(g[model.n:]))),
                      'relative_angle_and_field_force_inf':float(np.max(abs(basis.T@g))),
                      'step':alpha})
    return z,trace


def rotate(model,z):
    E,g = model.profile(z)
    d = model.last['directions'].copy()
    now = spin_jac(model,d,np.zeros(3)).T@g[model.n:]
    if np.max(abs(now))<2e-8:
        return z,{'initial_world_rotation_gradient':now,'final_world_rotation_gradient':now,
                  'evaluations':0}
    trial = z.copy()
    work = []
    def fun(v,law='current'):
        trial[model.n:] = spin(model,d,v)
        e,grad = model.profile(trial,law)
        actual = spin_jac(model,d,v).T@grad[model.n:]
        if law=='current':
            work.append((float(np.linalg.norm(actual)),e,v.copy(),actual))
        return actual
    ds = .002*np.eye(3)
    H = np.column_stack([(fun(v,'fixed')-fun(-v,'fixed'))/.004 for v in ds])
    root(fun,np.zeros(3),jac=lambda _:H,method='hybr',options={'xtol':1e-7,'maxfev':12})
    accepted = [r for r in work if r[1]<=E+4e-12]
    score,e,v,grad = min(accepted,key=lambda u:u[0])
    trial[model.n:] = spin(model,d,v)
    return trial,{'initial_world_rotation_gradient':now,'final_world_rotation_gradient':grad,
                  'world_rotation_vector_rad':v,'energy_change':e-E,'evaluations':len(work)}


def main():
    model = TransferredContact()
    old = read(CORRECTED)['final']
    z0 = np.array(old['coordinates'])[model.active]
    model.fix_predictor(old['kappa_native'])
    H,asym = hessian(model,z0,np.arange(len(z0)),4)
    z,a = relative(model,z0,H)
    z,b = rotate(model,z)
    z,c = relative(model,z,H)
    end = model.replay(z)
    g = end['gradient_profiled']
    passed = max(abs(g[:model.n]))<1e-6 and max(abs(g[model.n:]))<1e-7
    files = (Path(__file__).resolve(),CORRECTED,ROOT/'scripts'/'raccorder_contact_R353.py',
             ROOT/'scripts'/'relaxer_SF6_R353.py')
    dump(OUTPUT/'CONTACT_POLI_R353.json',{'revision':353,'full':True,'physical_validation':False,
        'F_frames_held':True,'initial':old,'final':end,'relative_stage_before':a,'world_rotation_stage':b,
        'relative_stage_after':c,'predictor_hessian_absolute_asymmetry':asym,
        'stationarity_on_82_coordinates':bool(passed),
        'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in files}})
    print('Final SF6 chart force',max(abs(g[:model.n])),max(abs(g[model.n:])),
          'radii',end['radii_native']/model.common['scale_native_per_angstrom'],'pass',passed,flush=True)


if __name__=='__main__':
    main()
