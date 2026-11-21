'Forward, matched-step and time-reversed tests of the native R357 action.'

import argparse
from pathlib import Path
from time import perf_counter

from fields_and_references import dump
from dynamics_size import (CORE,OUTPUT,ROOT,SIZES,Molecule,errors,
                                    prepare_motion,work_and_covariance)
from trials_nodal import digest
from kernel_curves import CurveKinetic
from rotation_prepared import GATES
import numpy as np


SOURCE = Path(__file__).resolve()
CASES = {'SF6':{'duration':.24,'steps':(.04,.02,.01)},
         'SF4':{'duration':.24,'steps':(.04,.02)}}
COMMON_SAMPLES = (.0,.08,.16,.24)
PARENTS = (
    'sources/cinetique_complete_R343.c','sources/action_parametree_R331.c',
    'scripts/noyau_courbes_R352.py','scripts/relaxer_noeuds_R354.py',
    'scripts/travail_inertie_taille_R355.py',
    'scripts/derivee_taille_R357.c','scripts/taille_dynamique_R357.py',
    'scripts/dynamique_taille_R357.py',
)


def hashes(model,source,extra=()):
    paths = [source,model.path,SIZES,*(ROOT/f for f in PARENTS),*extra]
    return {str(f.relative_to(ROOT)):digest(f) for f in paths}


def timeline(model,y,duration,dt,observed=None,log=None):
    steps = round(duration/dt)
    if abs(steps*dt-duration)>2e-13:
        raise ValueError('The matched window is not an integer number of steps')
    rows = []
    projection = 0.
    elapsed = perf_counter()
    for k in range(steps+1):
        derivative,d = model.evaluate(y)
        d['time_native'] = round(k*dt,12)
        d['accepted_step_projection_inf'] = projection
        if observed is not None:
            observed(y,derivative,d,k)
        rows.append(d)
        if log is not None and (k==0 or k==steps or k%max(2,steps//4)==0):
            print(log,k,'of',steps,'t',d['time_native'],'H delta',
                  d['H_native']-rows[0]['H_native'],'wall seconds',round(perf_counter()-elapsed,1),flush=True)
        if k<steps:
            y,projection = model.step(y,dt,derivative)
    return y,rows


def history_bounds(rows,excitation):
    head = rows[0]
    return {
        'max_energy_drift_native':max(abs(q['H_native']-head['H_native']) for q in rows),
        'max_energy_drift_over_excitation':max(abs(q['H_native']-head['H_native']) for q in rows)/excitation,
        'max_energy_drift_over_whole_H':max(abs(q['H_native']-head['H_native']) for q in rows)/abs(head['H_native']),
        'P_drift_world_l2_max':max(np.linalg.norm(q['P_world']-head['P_world']) for q in rows),
        'J_drift_world_l2_max':max(np.linalg.norm(q['J_world']-head['J_world']) for q in rows),
        'J_drift_world_relative_max':max(np.linalg.norm(q['J_world']-head['J_world']) for q in rows)/np.linalg.norm(head['J_world']),
        'max_accepted_projection_inf':max(q['accepted_step_projection_inf'] for q in rows),
        'max_instantaneous_abs_Hdot':max(abs(q['energy_derivative']) for q in rows),
        'max_EL_inf':np.max([q['field_and_body_EL_inf'] for q in rows],axis=0),
        'minimum_edge_dot':min(q['edge_dot_min'] for q in rows),
        'minimum_site_schur_eigenvalue':min(min(q['site_schur_min_eigenvalues']) for q in rows),
    }


def replay_kinetic(model,y):
    s = model.unpack(y)
    old = CurveKinetic(model.beta,[])
    checks = []
    for i in (0,1):
        args = (s.n[i],model.h[i],s.t[i],np.r_[np.zeros(3),s.v[i]])
        a,b = model.kernel.quadratic(*args),old.quadratic(*args)
        defect = max(float(np.max(abs(x-z))) for x,z in zip(a,b,strict=True))
        if defect>5e-11:
            raise ArithmeticError('The empty-window R352 kinetic law did not replay')
        checks.append(defect)
    return checks


def compare_steps(model,phases,returns,steps):
    table = []
    for j in range(len(phases)-1):
        table.append({'steps':[steps[j],steps[j+1]],'matched_endpoint_differences':{
            str(t):errors(model,phases[j][t],phases[j+1][t]) for t in COMMON_SAMPLES if t!=0.}})
    if len(table)>1:
        final = str(COMMON_SAMPLES[-1])
        a,b = (p['matched_endpoint_differences'][final] for p in table[:2])
        order = {k:float(np.log2(a[k]/b[k])) if b[k]>0. and a[k]>0. else None for k in a}
    else:
        order = None
    return {'step_pairs':table,'observed_order_at_final_endpoint':order,'reverse_at_all_steps':returns}


def run(species):
    case = CASES[species]
    dt_values = case['steps']
    duration = case['duration']
    model = Molecule(species)
    start = perf_counter()
    path = OUTPUT/f'TRAJECTOIRES_{species}_R357.json'
    checkpoint = OUTPUT/f'ETAT_FINAL_{species}_R357.npz'
    try:
        y = prepare_motion(model)
        dy,diag = model.evaluate(y)
        energy0 = model.energy(model.rest())
        excitation = diag['H_native']-energy0
        if excitation<=0.:
            raise ArithmeticError('The prepared motion does not have positive energy above rest')
        work,covariance = work_and_covariance(model,y,dy,diag)
        local = model.size_power_controls(y)
        if max(q['kinetic_work_defect'] for row in local for q in row['tests'])>2e-7:
            raise ArithmeticError('The size work control failed')
        reverse_dy,_ = model.evaluate(model.reverse(y))
        a,b = model.unpack(dy),model.unpack(reverse_dy)
        rev = max(float(np.max(abs(x-s*z))) for x,z,s in zip(a,b,(-1,1,-1,-1,1),strict=True))
        if rev>GATES['instantaneous_reversal_rhs_inf']:
            raise ArithmeticError('The Euler-Lagrange flow did not pass time inversion')
        record = {'revision':357,'species':species,'grid':CORE,'full':False,'physical_validation':False,
                  'scope':'all nodal interior S3 fields and independent mobile SE3 frames, fixed outer faces',
                  'case':case,'common_observation_times_native':COMMON_SAMPLES,
                  'prescribed_steps_before_replay':True,
                  'kinetic_R352_replay_defect_on_two_sites':replay_kinetic(model,y),
                  'flow_potential_and_H_work_checks':work,'global_SE3_covariance_defects':covariance,
                  'kinetic_size_derivative_checks':local,
                  'instantaneous_reversal_RHS_inf':rev,
                  'rest_energy_native':energy0,'initial_excitation_above_rest_native':excitation,
                  'input_sha256':hashes(model,SOURCE),'runs':[]}
        dump(path,record)
        phases,returns = [],[]
        for dt in dt_values:
            endpoints = {}
            def retain(y,derivative,d,k):
                t = d['time_native']
                if t in COMMON_SAMPLES and t!=0.:
                    endpoints[t] = y
            end,forward = timeline(model,y,duration,dt,retain,log=f'{species} forward dt={dt}')
            back,reverse = timeline(model,model.reverse(end),duration,dt,log=f'{species} return dt={dt}')
            reverse_error = errors(model,y,model.reverse(back))
            phases.append(endpoints)
            returns.append({'dt':dt,'error':reverse_error})
            record['runs'].append({
                'dt':dt,'steps_forward_and_return':len(forward)-1,
                'forward':forward,'return':reverse,
                'forward_bounds':history_bounds(forward,excitation),
                'return_bounds':history_bounds(reverse,excitation),
                'reverse_state_errors':reverse_error,
                'maximum_changes_between_initial_and_forward_endpoint':errors(model,y,end)})
            dump(path,record)
            print(species,'dt',dt,'reverse error',reverse_error,flush=True)
        record.update({'full':True,'step_and_return_comparison':compare_steps(model,phases,returns,dt_values),
                       'final_state_file':checkpoint.name,'elapsed_seconds':perf_counter()-start})
        np.savez_compressed(checkpoint,state=end,returned=model.reverse(back))
        record['final_state_sha256'] = digest(checkpoint)
        dump(path,record)
    finally:
        model.close()


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('species',choices=CASES)
    run(p.parse_args().species)
