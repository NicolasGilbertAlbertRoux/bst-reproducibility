'Refine the nodal relaxation with matrix-free Newton steps and the exact action.'

import argparse
from pathlib import Path

from sweep_sf4 import read
from fields_and_references import dump
from trials_nodal import digest
from operator_nodes import FieldHessian,pcg
from relax_nodes import (ANCHOR,OUTPUT,ROOT,TARGET_L2,NodalSF6,Smoother,
    edge_minimum,product,projected,retract,snapshot)
import numpy as np


SOURCE = Path(__file__).resolve()


def refine(model,n,rounds,matvecs,D=None,frames=None,goal=TARGET_L2,verbose=True):
    E,g = model.evaluate(n,True,D=D,frames=frames)
    trace = [snapshot(E,g,model.last,0)]
    for k in range(1,rounds+1):
        if trace[-1]['field_gradient_l2']<=goal:
            break
        H = FieldHessian(model,n,D=D,frames=frames)
        pre = Smoother(n,model.h,model.model.common['beta'],model.mask)
        damping = max(2e-7,min(2e-3,.007*trace[-1]['field_gradient_l2']))
        accepted = False
        for _ in range(3):
            d,linear = pcg(H,-g,lambda v:pre.apply(n,v),maximum=matvecs,
                           relative_tolerance=.018,shift=damping)
            slope = product(g,d)
            if slope>=0.:
                d = -pre.apply(n,g)
                slope = product(g,d)
            step = min(1.,.08/max(float(np.max(np.linalg.norm(d,axis=2))),1e-30))
            for _ in range(10):
                trial = retract(n,d,step,model.mask)
                try:
                    e,h = model.evaluate(trial,True,D=D,frames=frames)
                except (ValueError,ArithmeticError):
                    step *= .5
                    continue
                if e<=E+1e-4*step*slope or (abs(e-E)<2e-12 and np.linalg.norm(h)<.7*np.linalg.norm(g)):
                    accepted = True
                    break
                step *= .5
            if accepted:
                break
            damping *= 12.
        if not accepted:
            break
        n,E,g = trial,e,h
        row = snapshot(E,g,model.last,k)
        row.update({'step':step,'directional_derivative_native':slope,'linear_system':linear})
        trace.append(row)
        if verbose:
            print('Newton',k,row['energy_native'],row['field_gradient_l2'],'step',step,
                  'matvecs',linear['matvecs'],'res',linear['relative_residual_history'][-1],
                  'negative',linear['negative_curvature_encountered'],flush=True)
    return n,trace


def main(a):
    p = OUTPUT/f'ETAT_{a.seed}_R354.npz'
    seed_report = OUTPUT/f'RELAXATION_{a.seed}_R354.json'
    model = NodalSF6(a.workers)
    try:
        with np.load(p) as f:
            n = f['fields']
        if read(seed_report)['fields_sha256']!=digest(p):
            raise ArithmeticError('The checkpoint differs from its recorded identity')
        H = FieldHessian(model,n)
        i = np.arange(n.size,dtype=float)
        direction = np.cos(.271*i).reshape(n.shape)
        del i
        audits = {'fixed':H.test(direction),'reactive':H.test(direction,exact=True)}
        if max(z['relative_l2_defect'] for z in audits['fixed'])>2e-6:
            raise ArithmeticError(f'The frozen-curvature Hessian did not replay: {audits}')
        print('Hessian directional defects',audits,flush=True)
        n,trace = refine(model,n,a.rounds,a.matvecs)
        E,g = model.evaluate(n,True,pose=True,keep=True)
        last = model.last
        final = snapshot(E,g,last,trace[-1]['iteration'])
        data = OUTPUT/'ETAT_POLI_N48_R354.npz'
        np.savez_compressed(data,fields=n,radii_native=last['radii_native'],directions=model.D,
                            frames=model.frames,h_native=model.h)
        paths = (p,seed_report,SOURCE,ANCHOR,ROOT/'scripts'/'operateur_noeuds_R354.py',
                 ROOT/'scripts'/'hessien_nodal_R354.c',ROOT/'scripts'/'relaxer_noeuds_R354.py')
        dump(OUTPUT/'NOEUDS_POLIS_R354.json',{'revision':354,'full':True,'physical_validation':False,
            'N48_field_only_with_profiled_radii':True,'directions_and_fluorine_frames_held':True,
            'newton_steps_with_reactive_energy_and_gradient':trace,'directional_Hessian_checks':audits,
            'final':{**final,**last},'passes_nodal_gradient_target':bool(final['field_gradient_l2']<=TARGET_L2),
            'target_nodal_gradient_l2':TARGET_L2,
            'norm_defect_inf':float(np.max(abs(np.sum(n*n,axis=2)-1.))),
            'fixed_boundary_defect_inf':float(np.max(abs(n[:,~model.mask]-model.n0[:,~model.mask]))),
            'minimum_adjacent_field_dot_by_site':edge_minimum(n,model.count),
            'fields_filename':data.name,'fields_sha256':digest(data),
            'input_sha256':{str(q.relative_to(ROOT)):digest(q) for q in paths}})
        print('Nodal polishing',final,'radii A',final['radii_native']/model.model.common['scale_native_per_angstrom'],flush=True)
    finally:
        model.close()


if __name__=='__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--seed',default='N48_PILOTE')
    a.add_argument('--rounds',type=int,default=9)
    a.add_argument('--matvecs',type=int,default=25)
    a.add_argument('--workers',type=int,default=7)
    main(a.parse_args())
