'Short native-time integrations; step refinement and solver round trip.\n\nNo stationary or physical validation is inferred from this local trajectory.\nRK4 is an explicit numerical choice, not an extra law or physical clock.'
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from dynamics_full import Dynamics,ROOT
from fields_and_references import dump,plain
from verify_final import geometry_check

def target_momentum(q,p):
    from geometry_conditional import left_jacobian
    return -np.linalg.solve(left_jacobian(q[84:87]).T,p[84:87])

def evaluate(d,y):
    rhs,H=d.evaluate(y);return rhs,H,d.last['target_potential_torque'].copy()

def rk4(d,y,h):
    k1,_,t1=evaluate(d,y);k2,_,t2=evaluate(d,y+.5*h*k1);k3,_,t3=evaluate(d,y+.5*h*k2);k4,_,t4=evaluate(d,y+h*k3)
    return y+h/6*(k1+2*k2+2*k3+k4),h/6*(t1+2*t2+2*t3+t4)

def snapshot(d,y,t,H0,mu20):
    dy,H=d.evaluate(y,diagnostics=True);record=plain(d.last)
    record.update(time_native=float(t),relative_energy_drift=float((H-H0)/max(1.,abs(H0))),
        spatial_Casimir_relative_drift=float((y[174:177]@y[174:177]-mu20)/max(1.,mu20)),
        full_calibration_geometry_check=geometry_check(record))
    return record

def integrate(d,y0,duration,steps,name):
    y=y0.copy();h=duration/steps;_,H0=d.evaluate(y);mu20=float(y[174:177]@y[174:177]);impulse=np.zeros(3)
    history=[snapshot(d,y,0.,H0,mu20)];states=[y.copy()];start=time.time()
    for i in range(steps):
        y,increment=rk4(d,y,h);impulse+=increment
        history.append(snapshot(d,y,(i+1)*h,H0,mu20));states.append(y.copy())
        record={'revision':343,'completed':False,'name':name,'duration_native':duration,'step_native':h,'accepted_steps':i+1,
                'history':history,'last_state':y,'reference_torque_impulse_native':impulse,'physical_validation':False}
        dump(ROOT/'results'/f'{name}_INTEGRATION_R343.json',record)
        print(name,'step',i+1,'/',steps,'t',round((i+1)*h,6),'energy drift',history[-1]['relative_energy_drift'],
              'angle',history[-1]['geometry']['equatorial_angle_deg'],'seconds',round(time.time()-start,1),flush=True)
    ell0=target_momentum(y0[:87],y0[87:174]);ell=target_momentum(y[:87],y[87:174]);record.update(completed=True,
        target_momentum_change=ell-ell0,reference_torque_impulse_balance_inf_defect=float(np.max(abs(ell-ell0-impulse))),
        maximum_energy_relative_drift=float(max(abs(r['relative_energy_drift']) for r in history)),
        maximum_spatial_Casimir_relative_drift=float(max(abs(r['spatial_Casimir_relative_drift']) for r in history)),
        seconds=time.time()-start)
    dump(ROOT/'results'/f'{name}_INTEGRATION_R343.json',record)
    np.savez_compressed(ROOT/'results'/f'{name}_TRAJECTOIRE_R343.npz',states=np.array(states),times=np.linspace(0,duration,steps+1))
    return y,record

def run(duration,steps):
    start=time.time();d=Dynamics();y0=d.initial();d.evaluate(y0,diagnostics=True);initial=plain(d.last)
    c=d.configuration;B=np.zeros((90,6));B[:3,3:6]=np.eye(3);B[87:90,:3]=-np.eye(3)
    old=np.load(ROOT/'results'/c.row['operator_file']);oldI=old['I'];massdef=float(np.max(abs(B.T@d.M@B-oldI)))
    assert massdef<2e-7
    p=c.row['common_rotation_fit']['candidates'][0]
    forcedef=float(np.max(abs(np.array(initial['Hamiltonian_configuration_gradient'][:72])-np.array(p['native_force_residual']))))
    assert forcedef<2e-6
    dump(ROOT/'resultats/LIMITE_RIGIDE_ET_DEPART_R343.json',{'revision':343,'initial':initial,'rigid_mass_inf_defect_from_R341':massdef,
        'rigid_force_inf_defect_from_R341':forcedef,'initial_state':y0,'old_target_momentum_conservation_not_assumed':True,
        'held_references_can_exchange_target_angular_momentum':True,'physical_validation':False})
    print('R343 rigid limit checked, force defect',forcedef,'mass defect',massdef,'initial target torque',initial['target_potential_torque'],flush=True)
    runs=[];ends=[]
    for n in steps:
        endpoint,record=integrate(d,y0,duration,n,f'N{n}');runs.append(record);ends.append(endpoint)
    back,backrecord=integrate(d,ends[-1],-duration,steps[-1],'RETOUR')
    scale=np.maximum(1.,abs(y0));differences=[]
    for a,b,n,nn in zip(ends[:-1],ends[1:],steps[:-1],steps[1:],strict=True):
        delta=b-a;differences.append({'coarse_steps':n,'fine_steps':nn,'scaled_state_inf_difference':float(np.max(abs(delta)/scale)),
             'configuration_inf_difference':float(np.max(abs(delta[:87]))),'canonical_momenta_inf_difference':float(np.max(abs(delta[87:174]))),
             'spatial_momentum_inf_difference':float(np.max(abs(delta[174:177])))})
    ratio=differences[0]['scaled_state_inf_difference']/max(differences[1]['scaled_state_inf_difference'],1e-30) if len(differences)>1 else None
    summary={'revision':343,'completed':True,'duration_native':duration,'step_counts':steps,'initial':initial,
        'last_fine_snapshot':runs[-1]['history'][-1],'endpoint_step_differences':differences,'observed_difference_ratio':ratio,
        'solver_round_trip_scaled_state_inf_defect':float(np.max(abs(back-y0)/scale)),
        'solver_round_trip_configuration_inf_defect':float(np.max(abs(back[:87]-y0[:87]))),
        'fine_maximum_energy_relative_drift':runs[-1]['maximum_energy_relative_drift'],
        'fine_maximum_spatial_Casimir_relative_drift':runs[-1]['maximum_spatial_Casimir_relative_drift'],
        'fine_target_momentum_change_native':runs[-1]['target_momentum_change'],
        'fine_reference_torque_impulse_native':runs[-1]['reference_torque_impulse_native'],
        'fine_reference_torque_impulse_balance_inf_defect':runs[-1]['reference_torque_impulse_balance_inf_defect'],
        'rigid_limit_mass_inf_defect':massdef,'rigid_limit_force_inf_defect':forcedef,
        'native_time_not_a_physical_clock':True,'same_coefficients_and_no_angle_constraint_added':True,
        'short_trajectory_does_not_establish_equilibrium_or_molecular_stability':True,
        'held_reference_field_reaction_source_not_dynamically_closed':True,
        'full_nodal_continuum_and_other_species_not_integrated':True,'physical_validation':False,'physical_regions_excluded':0,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seconds':time.time()-start}
    dump(ROOT/'resultats/BILAN_INTEGRATIONS_R343.json',summary)
    print('R343 completed; energy',summary['fine_maximum_energy_relative_drift'],'roundtrip',summary['solver_round_trip_scaled_state_inf_defect'],'seconds',round(time.time()-start,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--duration',type=float,default=.4);p.add_argument('--steps',nargs='+',type=int,default=[2,4,8]);a=p.parse_args();run(a.duration,a.steps)
