'Observe collective rotation while retaining the five native frame responses.'

import argparse
from pathlib import Path

from sweep_sf4 import ROOT, read
from fields_and_references import dump
from test_fit import CANDIDATES, AdjustedFrames, get_candidate
from trials_nodal import digest
from solve_acceleration_nodal import constrained_inverse
from readout_inertial import REFERENCE
from refit_dependencies import skew
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation
from scipy.stats import qmc


OUTPUT = ROOT/'results'/'R349'
R348 = ROOT/'results'/'R348'
SOURCE = Path(__file__).resolve()
REF = np.array(REFERENCE['rotation']['abc_cm_inverse'])
RATIOS = REF[1:]/REF[0]
CALIBRATION = OUTPUT/'CALAGE_TORSIONS_R349.json'


def reduced(M,keep):
    M = (M+M.T)/2
    if keep==len(M):
        return M
    return M[:keep,:keep]-M[:keep,keep:]@np.linalg.solve(M[keep:,keep:],M[keep:,:keep])


def rotation_reading(I):
    eig = np.linalg.eigvalsh((I+I.T)/2)
    if eig[0]<=0:
        raise ArithmeticError('Rotation tensor has a nonpositive eigenvalue')
    ratios = eig[0]/eig[1:]
    return {'inertia_native':I,'eigenvalues_native':eig,'ratios':ratios,
            'relative_errors':ratios/RATIOS-1}


def local_matrices(model,state,with_shape):
    fields,_,x,_ = model.unpack(state)
    blocks = []
    defects = []
    for i,(n,h) in enumerate(zip(fields,model.h,strict=True)):
        A,C,G = model.kernel.mass(n,h)
        C = np.ascontiguousarray(C[:,:,3:])
        block = G[3:,3:]
        if with_shape:
            W = model.frame_columns[i]
            H = constrained_inverse(A,n)
            HC, HW = H@C,H@W
            wc = W.reshape(-1,6).T@HC.reshape(-1,6)
            gram = W.reshape(-1,6).T@HW.reshape(-1,6)
            change = C.reshape(-1,6).T@(HC-HW@np.linalg.solve(gram,wc)).reshape(-1,6)
            block = block-change
            defects.append(float(np.max(abs(change))))
        blocks.append((block+block.T)/2)
    return model.geometry(x),np.array(blocks),defects


def rigid_geometry_map(centers,frames):
    maps = np.zeros((5,6,21))
    for i,(X,O) in enumerate(zip(centers,frames,strict=True)):
        maps[i,:3,:3] = O.T
        maps[i,3:,:3] = -O.T@skew(X)
        maps[i,3:,3:6] = O.T
        maps[i,:3,6+3*i:9+3*i] = O.T
    return maps


def pose_coordinate_map(geometry):
    X,J,_,_,jets,_ = geometry
    T = np.zeros((30,21))
    T[:6,:6] = np.eye(6)
    T[:3,6:9] = np.eye(3)
    for i,(Xj,Jj,jet) in enumerate(zip(X[1:],J,jets,strict=True)):
        T[6+3*i:9+3*i,6:9] = np.linalg.solve(Jj,skew(Xj))
        inverse = np.linalg.inv(jet[0])
        T[18+3*i:21+3*i,6:9] = -inverse
        T[18+3*i:21+3*i,9+3*i:12+3*i] = inverse
    return T


def kinetic21(centers,frames,blocks):
    maps = rigid_geometry_map(centers,frames)
    result = np.einsum('nai,nab,nbj->ij',maps,blocks,maps)
    return (result+result.T)/2


def read_branches(geometry,blocks):
    centers,_,_,frames,_,E = geometry
    T = pose_coordinate_map(geometry)
    maps = rigid_geometry_map(centers,frames)
    defect = float(np.max(abs(np.array(E)[:,3:,:]@T-maps)))
    if defect>2e-12:
        raise ArithmeticError('The new coordinates do not reproduce the native pose tangent')
    D = kinetic21(centers,frames,blocks)
    dirs = centers[1:]/np.linalg.norm(centers[1:],axis=1)[:,None]
    twist = np.zeros((15,4))
    for j,d in enumerate(dirs):
        twist[3*j+3:3*j+6,j] = d
    branches = {'locked':np.zeros((15,0)), 'F_bond_twists':twist,
                'F_frames':np.eye(15)[:,3:], 'all_frames':np.eye(15)}
    result = {}
    for name,Z in branches.items():
        W = block_diag(np.eye(6),Z)
        result[name] = rotation_reading(reduced(W.T@D@W,3))
    return result,defect


def twisted_frames(centers,frames,angles):
    result = frames.copy()
    axes = centers[1:]/np.linalg.norm(centers[1:],axis=1)[:,None]
    result[1:] = Rotation.from_rotvec(axes*angles[:,None]).as_matrix()@result[1:]
    return result


def collect():
    rows = []
    candidate = get_candidate('lois_recues')
    for count in (48,64):
        model = AdjustedFrames(count,candidate)
        try:
            terminal_path = R348/f'N{count}_DT005.npz'
            with np.load(terminal_path) as saved:
                terminal = saved['state']
            for name,y in (('anchor',model.adjusted_seed),('terminal',terminal)):
                geom,G,_ = local_matrices(model,y,False)
                tested, map_defect = read_branches(geom,G)
                Gm,shape_change = (None,None) if name=='anchor' else local_matrices(model,y,True)[1:]
                row = {'count':count,'stage':name,'centers':geom[0],'frames':geom[3],
                       'native_blocks':G,'branches':tested,'coordinate_map_inf_defect':map_defect,
                       'allowed_nodal_response_change_by_atom':shape_change,
                       'all_frames_with_allowed_nodal_rates':None if Gm is None else read_branches(geom,Gm)[0]['all_frames'],
                       'source':str(terminal_path.relative_to(ROOT)) if name=='terminal' else str(CANDIDATES.relative_to(ROOT)),
                       'source_sha256':digest(terminal_path if name=='terminal' else CANDIDATES)}
                rows.append(row)
                print(count,name,'twists',tested['F_bond_twists']['relative_errors'],
                      'all',tested['all_frames']['relative_errors'],flush=True)
        finally:
            model.close()
    dump(OUTPUT/'BANQUE_ROTATION_R349.json',{'rows':rows,'reference':REFERENCE['rotation'],
         'source_sha256':digest(SOURCE),'physical_validation':False})


class MobileRotor:
    def __init__(self,row):
        self.centers = np.array(row['centers'])
        self.frames = np.array(row['frames'])
        self.blocks = np.array(row['native_blocks'])
        self.translational = np.array([reduced(block[np.ix_([3,4,5,0,1,2],[3,4,5,0,1,2])],3) for block in self.blocks])
        self.velocities = np.array([np.column_stack((-skew(X),np.eye(3))) for X in self.centers])

    def reading(self,angles):
        O = twisted_frames(self.centers,self.frames,angles)
        Q = O.transpose(0,2,1)@self.velocities
        global_T = np.einsum('nai,nab,nbj->ij',Q,self.translational,Q)
        return rotation_reading(reduced(global_T,3))


def fit():
    bank = OUTPUT/'BANQUE_ROTATION_R349.json'
    anchors = [row for row in read(bank)['rows'] if row['stage']=='anchor']
    models = [MobileRotor(row) for row in anchors]
    def errors(angles):
        return np.concatenate([m.reading(angles)['relative_errors'] for m in models])
    grid = (2*qmc.Sobol(4,scramble=True,seed=349).random_base2(12)-1)*np.pi
    coarse = np.array([max(abs(errors(angles))) for angles in grid])
    trials = []
    for start in [np.zeros(4),*grid[np.argsort(coarse)[:15]]]:
        result = least_squares(errors,start,bounds=(-np.pi,np.pi),max_nfev=500,
                               ftol=3e-13,xtol=3e-13,gtol=3e-13)
        trials.append({'angles':result.x,'errors':result.fun,'max_error':float(max(abs(result.fun))),
                       'evaluations':result.nfev,'successful':result.success})
    best = min(trials,key=lambda r:r['max_error'])
    epigraph = minimize(lambda x:x[-1],np.r_[best['angles'],best['max_error']],method='SLSQP',
        bounds=[(-np.pi,np.pi)]*4+[(0.,.1)],
        constraints={'type':'ineq','fun':lambda x:np.r_[x[-1]-errors(x[:4]),x[-1]+errors(x[:4])]},
        options={'maxiter':150,'ftol':1e-13})
    if np.all(np.isfinite(epigraph.x)) and max(abs(errors(epigraph.x[:4])))<best['max_error']:
        angles = epigraph.x[:4]
    else:
        angles = best['angles']
    check = []
    for row,model in zip(anchors,models,strict=True):
        O = twisted_frames(model.centers,model.frames,angles)
        directions = model.centers[1:]/np.linalg.norm(model.centers[1:],axis=1)[:,None]
        before = model.frames[1:].transpose(0,2,1)@directions[:,:,None]
        after = O[1:].transpose(0,2,1)@directions[:,:,None]
        direct = rotation_reading(reduced(kinetic21(model.centers,O,model.blocks),3))
        fast = model.reading(angles)
        defect = float(max(abs(direct['ratios']-fast['ratios'])))
        if max(defect,float(np.max(abs(before-after))))>1e-11:
            raise ArithmeticError('Twisting changed the bond reader or the condensed action')
        check.append({'count':row['count'],'direct':direct,'reduction_agreement':defect,
                      'contact_direction_inf_defect':float(np.max(abs(before-after)))})
    candidate = get_candidate('lois_recues')
    x = np.array(candidate['center_chart'])
    centers = models[0].centers
    axes = centers[1:]/np.linalg.norm(centers[1:],axis=1)[:,None]
    x[12:] = (Rotation.from_rotvec(axes*angles[:,None])*Rotation.from_rotvec(x[12:].reshape(4,3))).as_rotvec().ravel()
    candidate.update({'center_chart':x,'label':'torsions_common_R349','parent':'lois_recues_R348'})
    dump(CALIBRATION,{'twists_rad':angles,'twists_deg':np.degrees(angles),'declared_twist_window_rad':[-np.pi,np.pi],
         'coarse_trials':len(grid),'best_coarse_max_relative_error':float(np.min(coarse)),'refinements':trials,
         'anchors':check,'reference_ratios':RATIOS,'candidate':candidate,
         'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (CANDIDATES,bank)},
         'source_sha256':digest(SOURCE),'physical_validation':False})
    print('common twists',np.degrees(angles),'mesh errors',[r['direct']['relative_errors'] for r in check],flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('bank','fit'))
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    {'bank':collect,'fit':fit}[args.action]()
