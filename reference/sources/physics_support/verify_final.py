'Replay all new pressure candidates and retain label-independent geometry checks.'
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import itertools,json
from pathlib import Path
import numpy as np
from mix import dump
ROOT=Path(__file__).resolve().parents[1]

def geometry_check(point):
    A=np.array(point['geometry']['all_pair_angles_deg']);target=np.full((4,4),87.8);np.fill_diagonal(target,0.);target[0,1]=target[1,0]=173.1;target[2,3]=target[3,2]=101.6
    ii=np.triu_indices(4,1);rows=[]
    for p in itertools.permutations(range(4)):
        angles=A[np.ix_(p,p)][ii];res=angles-target[ii];rows.append((float(max(abs(res))),p,angles.tolist(),res.tolist()))
    worst,p,angles,res=min(rows)
    return {'assignment':list(p),'labelled_equatorial_angle_deg':point['geometry']['equatorial_angle_deg'],
        'all_six_angles_in_best_assignment_deg':angles,'all_six_residuals_to_received_calibration_deg':res,
        'maximum_absolute_angular_residual_deg':worst,'calibration_values_not_independent_validation':True,
        'all_24_F_assignments_checked':True,'no_experimental_threshold_inferred':True}

def run():
    r=ROOT/'results';coarse=json.loads((r/'BRASSAGE_DEPENDANCES_R340.json').read_text());fine=json.loads((r/'AFFINEMENT_DEPENDANCES_R340.json').read_text());assert coarse['completed'] and fine['completed']
    count=0;maximum=0.
    for row in [*coarse['rows'],*fine['rows']]:
        z=np.load(r/row['operator_file']);I,D,b,C=[z[k] for k in ['I','D','base_gradient','coefficient_gradients']]
        for p in row['common_rotation_fit']['candidates']:
            w=np.array(p['velocity_native']);force=b+C@np.array([p['gamma'],p['kangle']])-.5*np.einsum('i,aij,j->a',w,D,w);mu=I@w;torque=np.r_[np.cross(w[:3],mu[:3]),np.cross(w[3:],mu[3:])]
            error=max(float(max(abs(force-p['native_force_residual']))),float(max(abs(torque-p['native_torque_residual']))));assert error<3e-7;maximum=max(maximum,error);count+=1
    feedback=json.loads((r/'REBOUCLAGE_APRES_FORME_R340.json').read_text());z=np.load(r/'REBOUCLAGE_OPERATEURS_R340.npz');I,D,b,C=[z[k] for k in ['I','D','base_gradient','coefficient_gradients']]
    for candidate in feedback['fit']['candidates']:
        w=np.array(candidate['velocity_native']);f=b+C@np.array([candidate['gamma'],candidate['kangle']])-.5*np.einsum('i,aij,j->a',w,D,w)
        assert max(abs(f-candidate['native_force_residual']))<2e-7
    transfers=[]
    for prefix in ['COARSE_','']:
        p=r/(prefix+'SF6_TRANSFERT_MEMES_LOIS_R340.json');d=json.loads(p.read_text());z=np.load(r/(prefix+'SF6_TRANSFERT_OPERATEURS_R340.npz'));I,D,b,C=[z[k] for k in ['I','D','base_gradient','coefficient_gradients']]
        for candidate in d['fit']['candidates']:
            w=np.array(candidate['velocity_native']);force=b+C@np.array([d['gamma'],d['kangle']])-.5*np.einsum('i,aij,j->a',w,D,w);assert max(abs(force-candidate['native_force_residual']))<2e-7
        transfers.append({'file':p.name,'beta':d['beta'],'best_candidate':d['fit']['candidates'][0]})
    releases=[]
    for name in ['REAJUSTEMENT_CONJOINT_R340.json','REAJUSTEMENT_FIN_R340.json','CONTINUATION_CONJOINTE_R340.json']:
        d=json.loads((r/name).read_text());energies=[d['initial']['effective_energy_native'],*[p['effective_energy_native'] for p in d['history']]];assert max(np.diff(energies))<1e-8
        releases.append({'file':name,'initial_geometry':geometry_check(d['initial']),'final_geometry':geometry_check(d['result']),
            'all_accepted_energies_nonincreasing':True,'state_force_inf_native':d['result']['state_force_inf_native'],
            'center_force_inf_native':d['result']['center_force_inf_native'],'Euler_torque_inf_native':d['result']['Euler_torque_inf_native']})
    parent=json.loads((ROOT/'parents_canon_R339/BST_SUIVI_VALIDATION.json').read_text());new=json.loads((ROOT/'canon/BST_SUIVI_VALIDATION.json').read_text());assert parent['rows']==new['rows'] and parent['counts']==new['counts'] and parent['common_candidate']==new['common_candidate']
    assert new['journal'][:-1]==parent['journal']
    for name in ['BST_MASTER_PROGRESS_LIST_POST_R9_2026-09-21.md','BST_TRANSMISSION_AUTONOME.txt','BST_METHODE_RESOLUTION_CANONIQUE.txt']:
        assert (ROOT/'canon'/name).read_bytes().endswith((ROOT/'parents_canon_R339'/name).read_bytes())
    oldmethod=json.loads((ROOT/'parents_canon_R339/BST_METHODE_RESOLUTION_CANONIQUE.json').read_text());newmethod=json.loads((ROOT/'canon/BST_METHODE_RESOLUTION_CANONIQUE.json').read_text())
    for k,v in oldmethod.items():
        if k!='revision':assert newmethod[k]==v,k
    result={'revision':340,'coarse_and_fine_pressure_candidates_replayed':count,'maximum_native_force_torque_replay_defect':maximum,
        'same_coefficient_SF6_transfers_replayed':transfers,'post_shape_feedback_replayed':True,'free_releases_and_permutation_invariant_geometry_checks':releases,
        'canonical_28_rows_counts_common_candidate_and_histories_preserved':True,'prior_method_rules_preserved':True,
        'directory_numerical_checks_pass':True,'physical_validation':False,'physical_regions_excluded':0}
    dump(r/'VERIFICATION_FINALE_DOSSIER_R340.json',result);print('R340 final exported-equation, geometry and preservation checks passed',count)

if __name__=='__main__':run()
