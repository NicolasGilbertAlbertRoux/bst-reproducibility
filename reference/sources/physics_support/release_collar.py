'Extend the stationarity test to all nodes inside the fixed outer faces.'

import argparse
from pathlib import Path

from fields_and_references import dump
from trials_nodal import digest
from forces_outside_map import field_energy
from refine_nodes import refine
from refit_pose import ANGLE_TOL,Pose,directions
from relax_nodes import ANCHOR,OUTPUT,ROOT,TARGET_L2,NodalSF6,edge_minimum,snapshot
import numpy as np


SOURCE = Path(__file__).resolve()
STATE = OUTPUT/'ETAT_RECALE_N48_R354.npz'
COLLAR = OUTPUT/'VARIATION_COLLIER_R354.json'


def face_interior(m):
    inside = np.zeros((m,m,m),dtype=bool)
    inside[1:-1,1:-1,1:-1] = True
    return inside.ravel()


def main(args):
    with np.load(STATE) as z:
        n,r,D,O,h = [z[k] for k in ('fields','radii_native','directions','frames','h_native')]
        c = float(z['contact_intercept_native'])
    model = NodalSF6(workers=7)
    try:
        original = model.mask.copy()
        model.mask = face_interior(model.count)
        model.model.geometry.d['r0'] = c
        E,g = model.evaluate(n,True,D=D,frames=O)
        start = snapshot(E,g,model.last,0)
        n,first = refine(model,n,args.rounds,args.matvecs,D=D,frames=O,goal=.5*TARGET_L2)
        cycle = []
        for k in range(1,args.cycles+1):
            pose = Pose(model,n,D,O)
            F = pose.frame_minimum(D)
            D,O,P = directions(pose,D)
            n,steps = refine(model,n,4,args.matvecs,D=D,frames=O,goal=.5*TARGET_L2)
            E,g = model.evaluate(n,True,D=D,frames=O,pose=True,keep=True)
            angle = float(max(np.linalg.norm(np.cross(D,model.last['direction_gradient']),axis=1)))
            frame = float(max(np.linalg.norm(model.last['frame_rotation_gradient_world'],axis=1)))
            row = {**snapshot(E,g,model.last,k),'frames':F,'directions':P,'nodal_steps':steps,
                   'direction_reaction_inf_native':angle,'frame_reaction_inf_native':frame}
            cycle.append(row)
            print('Expanded interior',k,row['field_gradient_l2'],angle,frame,flush=True)
            if max(angle,frame)<=ANGLE_TOL and row['field_gradient_l2']<=TARGET_L2:
                break
        assert cycle
        boundary = float(np.max(abs(n[:,~model.mask]-model.n0[:,~model.mask])))
        norm = float(np.max(abs(np.sum(n*n,axis=-1)-1.)))
        edges = edge_minimum(n,model.count)
        r1 = model.last['radii_native']
        model.m.frames = O
        alternate = float(field_energy(model.model,n,r1,D))
        if boundary!=0. or norm>1e-12 or min(edges)<=0. or abs(float(E)-alternate)>2e-8:
            raise ArithmeticError('The expanded-interior energy or constraint did not replay')
        f = OUTPUT/'ETAT_FACES_AVANT_POLI_R354.npz'
        np.savez_compressed(f,fields=n,radii_native=r1,directions=D,frames=O,h_native=h,contact_intercept_native=c)
        files = (SOURCE,STATE,COLLAR,ANCHOR,ROOT/'scripts/polir_noeuds_R354.py',ROOT/'scripts/recaler_pose_R354.py',
                 ROOT/'scripts/relaxer_noeuds_R354.py')
        dump(OUTPUT/'INTERIEUR_AVANT_POLI_R354.json',{'revision':354,'full':True,'physical_validation':False,
            'scope':'N48, all interior field nodes, fixed outer faces, inherited R354 joint contact intercept',
            'active_nodes_per_field':int(model.mask.sum()),'original_active_nodes_per_field':int(original.sum()),
            'active_field_dofs':int(7*model.mask.sum()*3),'start':start,
            'first_fixed_direction_relaxation':first,'joint_cycles':cycle,
            'final':{**cycle[-1],**model.last,'radii_angstrom':r1/model.model.common['scale_native_per_angstrom']},
            'fixed_faces_defect_inf':boundary,'unit_constraint_defect_inf':norm,
            'minimum_adjacent_field_dot_by_site':edges,'independent_energy_absolute_defect':abs(float(E)-alternate),
            'passes_joint_tolerances':bool(max(angle,frame)<=ANGLE_TOL and cycle[-1]['field_gradient_l2']<=TARGET_L2),
            'fields_filename':f.name,'fields_sha256':digest(f),
            'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in files}})
    finally:
        model.close()


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rounds',type=int,default=10)
    p.add_argument('--matvecs',type=int,default=34)
    p.add_argument('--cycles',type=int,default=2)
    main(p.parse_args())
