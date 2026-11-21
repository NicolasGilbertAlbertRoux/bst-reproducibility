'Resumable sampled exploration. Unknown boxes are never removed by samples.'
import os
os.environ["OPENBLAS_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1"
from pathlib import Path
import argparse,hashlib,itertools,json,time
import numpy as np
from scipy.stats import qmc
from engine_local import NativeConsole,ROOT

def dump(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def fingerprint(data):return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def diagnostic(result):
    if result.get('status')!='FINITE_NATIVE_CALCULATION':return {'numerical_status':'UNKNOWN','physical_status':'UNKNOWN','retained':True}
    g=result['geometry'];A=np.array(g['all_pair_angles_deg']);r=np.array(g['radii_native']);N=len(r)
    if N==4:
        target=np.full((4,4),87.8);np.fill_diagonal(target,0);target[0,1]=target[1,0]=173.1;target[2,3]=target[3,2]=101.6
    else:
        target=np.full((6,6),90.);np.fill_diagonal(target,0)
        for i in [0,2,4]:target[i,i+1]=target[i+1,i]=180.
    upper=np.triu_indices(N,1);best=None
    for p in itertools.permutations(range(N)):
        residual=A[np.ix_(p,p)][upper]-target[upper];score=float(np.sqrt(np.mean(residual**2)))
        if best is None or score<best[0]:best=(score,p,residual)
    score,p,residual=best
    out={'numerical_status':'FINITE','physical_status':'UNKNOWN','retained':True,
         'all_angle_RMS_to_retrospective_reference_deg':score,'all_angle_max_abs_residual_deg':float(max(abs(residual))),
         'best_atom_permutation':list(p),'reference_used_for_ranking_not_validation':True,
         'minimum_F_F_over_minimum_bond_radius':result['minimum_F_F_distance_native']/float(min(r)),
         'center_stationarity':result['center_stationarity_qualified'],
         'shape_force_inf_native':result['maximum_internal_shape_gradient_native'],
         'F_profile_force_inf_native':float(np.max(abs(np.array(result['individual_F_conjugate_gradients_native'])))),
         'near_coincident_centers_diagnostic_only':result['minimum_F_F_distance_native']<.01*float(min(r)),
         'continuous_region_excluded':False,'all_BST_constraints_evaluated':False}
    if N==4:
        q=r[list(p)];out.update(equatorial_angle_best_assignment_deg=float(A[p[2],p[3]]),axial_angle_best_assignment_deg=float(A[p[0],p[1]]),
            axial_to_equatorial_mean_radius_ratio=float(np.mean(q[:2])/np.mean(q[2:])),reference_radius_ratio=1.646/1.545)
    else:out['relative_bond_radius_spread']=float(np.ptp(r)/np.mean(r))
    return out

def recipes(engine):
    bases={};seeds={}
    received=json.loads((ROOT/'resultats/CALIBRATION_LEVIER_R335.json').read_text())['calibrated_native_point']
    for label in ['SF4','SF6']:
        m,Z=engine.model(label);bases[label]=m.baseq.copy()
        seeds[label]=received['full_geometry']['center_coordinates'] if label=='SF4' else received['SF6_same_transported_controls']['center_coordinates']
    cases=[]
    for weight in [0.,1.,1000.]:
        cases.append({'id':f'base_{weight:g}','phase':'wide_law_control','weight':weight,'state':{label:{'shape':np.zeros(50),'q':bases[label],'individual':np.tile(bases[label][2:],(4 if label=='SF4' else 6,1))} for label in bases}})
    # Independent states for the two contexts; identical law coefficient.
    design=qmc.Sobol(d=134,scramble=True,seed=336).random_base2(6)
    for i,u in enumerate(design):
        if i<24:phase='calibration_neighbourhood';level=0;weight=received['existing_kangle']*10**(2*u[0]-1);turn_budget=.16+.48*u[1];qwindow=.15
        else:phase='global_expansion';level=(i-24)//14;weight=10**((-6-level)+(6+2*level)*u[0]);turn_budget=[.8,1.6,2.4][min(level,2)];qwindow=[.8,1.6,2.4][min(level,2)]
        states={};cursor=2
        for label in ['SF4','SF6']:
            m,Z=engine.model(label);N=m.N;direction=2*u[cursor:cursor+50]-1;cursor+=50
            # Every column participates; scaling respects the correlated guard.
            shape=direction*turn_budget/max(float(np.max(abs(Z@direction))),1e-30)
            if phase=='calibration_neighbourhood':shape[4]-=.8
            q=2*u[cursor:cursor+4]-1;cursor+=4
            if phase=='calibration_neighbourhood':q=bases[label]+qwindow*q
            else:q=qwindow*q
            individual=2*u[cursor:cursor+2*N].reshape(N,2)-1;cursor+=2*N
            if phase=='calibration_neighbourhood':individual=q[2:]+qwindow*individual
            else:individual=qwindow*individual
            states[label]={'shape':shape,'q':q,'individual':individual}
        cases.append({'id':f'paired_{i:03d}','phase':phase,'level':int(level),'weight':float(weight),'state':states})
    for case in cases:
        for label,state in case['state'].items():
            recipe={'label':label,'shape':state['shape'].tolist(),'preparation':state['q'].tolist(),'individual_F':state['individual'].tolist(),
                'kangle':case['weight'],'maximum_profile_turn':3.,'relax_shape':False,'maxiter':120}
            if case['phase']=='calibration_neighbourhood':recipe['center_start']=seeds[label]
            yield {'case':case['id'],'phase':case['phase'],'level':case.get('level'),'label':label,'recipe':recipe}

def run(output,maximum=None):
    output.mkdir(parents=True,exist_ok=True);points=output/'points';points.mkdir(exist_ok=True)
    engine=NativeConsole();design=list(recipes(engine));started=time.time()
    signature={'engine_sha256':hashlib.sha256((ROOT/'scripts/moteur_local_R336.py').read_bytes()).hexdigest(),
        'geometry_sha256':hashlib.sha256((ROOT/'scripts/angles_internes_R335.py').read_bytes()).hexdigest(),
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'basis_sha256':engine.basis_hash,'windows':engine.windows,
        'design_sha256':fingerprint(design),'numerical_threads':{k:os.environ[k] for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']}}
    dump(output/'PLAN_BRASSAGE_R336.json',{'revision':336,'signature':signature,'recipes':design,'all_continuous_conditions_exhausted':False,'new_experimental_tolerance':None,'independent_context_states':True})
    rows=[];computed=0;reused=0
    for entry in design:
        key=fingerprint({'signature':signature,'recipe':entry['recipe']});path=points/(key+'.json')
        if path.exists():record=json.loads(path.read_text());assert record['key']==key;reused+=1
        else:
            if maximum is not None and computed>=maximum:continue
            try:result=engine.evaluate(entry['recipe'])
            except (ValueError,RuntimeError) as error:result={'status':'UNKNOWN','reason':str(error),'recipe':entry['recipe'],'retained':True,'physical_rejection':False,'physical_BST_validation':False}
            record={'key':key,'entry':entry,'result':result,'diagnostic':diagnostic(result)};dump(path,record);computed+=1
            print(entry['case'],entry['label'],result['status'],record['diagnostic'].get('all_angle_RMS_to_retrospective_reference_deg'),round(time.time()-started,1),flush=True)
        rows.append({'key':key,'file':str(path.relative_to(output)),**entry,'diagnostic':record['diagnostic']})
        finite=[r for r in rows if r['diagnostic']['numerical_status']=='FINITE']
        summary={'revision':336,'full':len(rows)==len(design),'signature':signature,'planned_points':len(design),'computed_this_invocation':computed,'cache_hits':reused,
            'rows':rows,'finite_points':len(finite),'unknown_points':len(rows)-len(finite),'all_regions_retained':True,
            'physical_regions_excluded':0,'physical_intervals_validated':0,'remaining_points':len(design)-len(rows),'seconds':time.time()-started}
        dump(output/'BRASSAGE_R336.json',summary)
    return summary
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'resultats/brassage_R336');ap.add_argument('--max-new-points',type=int);args=ap.parse_args()
    print(json.dumps({k:v for k,v in run(args.output,args.max_new_points).items() if k not in ['rows','signature']},ensure_ascii=False))
