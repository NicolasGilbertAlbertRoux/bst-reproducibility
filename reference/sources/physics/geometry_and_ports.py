'Confront the six distances and mechanical quantities in their unit and their preparation.'
import json,hashlib
import numpy as np
from pathlib import Path
from readout_ir import I,O,decode,save
def p(name):return json.loads((I/'evidence'/(name+'.json')).read_text())
def main():
    geo=p('GEOMETRY_SF_R571');st=decode('ETAT_SF6_PARENT_R480.json');a=np.array(st['directions']);r=np.exp(geo['params_log_s_g_b'][0])*np.array(st['radii'])
    source=[k for k in geo['comparisons_experimentales_Longueurs'] if k['molecule']=='SF6' and 'Bartell' in k['reference']]
    assert len(source)==6 and all(abs(r[s['site']]-s['distance_A'])<1e-9 for s in source)
    rows=[]
    for i,j in zip(*np.triu_indices(6,k=1)):
        c=a[i]@a[j];d=np.linalg.norm(r[i]*a[i]-r[j]*a[j]);kind='trans' if c<-.8 else 'cis'; octa=1.561*(2 if kind=='trans' else np.sqrt(2))
        rows.append(dict(i=int(i)+1,j=int(j)+1,family=kind,angle_deg=float(np.degrees(np.arccos(np.clip(c,-1,1)))),distance_A=float(d),ratio_at_frame_octa=float(d/octa)))
    control={}
    for name in ('SF4','SF6'):
        pred=p('DYNAMICS_'+name+'_R434')
        control[name]=dict(covariance=max(float(v) for v in pred['defects_covariance'].values()),work_DF=min(float(x['Hamiltonian_defect']) for x in pred['defects_work']),step=pred['step_native'])
    full=json.loads((O/'MOUVEMENT_N72_impedance_dt0.09.json').read_text());rr=sum(full['historique'].values(),[])
    native=dict(max_residual_Euler_Lagrange=float(max(r['EL_inf'] for r in rr)),minimum_Schur=float(min(r['schur_min'] for r in rr)),minimum_puissance_output=float(min(r['loss'] for r in rr)),max_work=float(max(abs(r['work_residual']) for r in rr)))
    assert native['minimum_Schur']>0 and native['max_residual_Euler_Lagrange']<1e-8
    lens=p('LECTURE_MULTIBANDE_R440')['lens_BST']
    assert len(lens['cases'])==12 and all(np.all(np.array(x['frequencies_natives'])>0) for x in lens['cases'])
    save('GEOMETRIE_ET_PORTS_R573.json',dict(distances_SF_A=r.tolist(),interval_diffraction_A=[1.559,1.563],six_in_interval=bool(np.all((r>=1.559)&(r<=1.563))),distances_FF=rows,check_motion_native=native,checks_actions_archived=control,checksum_lens_native=lens['sha_R379'],count_preparations_lens=len(lens['cases']),
      scope='r_g de diffraction = 1,561(2) Å, incertitude publiée 3σ. Les distances F–F de la configuration sont comparées au repère octaédrique à cette distance ; aucune distribution thermique et aucune intensité de diffraction complète ne sont inférées. Les décalages d’orientation des centres ne servent pas de veto.'))
if __name__=='__main__':main()
