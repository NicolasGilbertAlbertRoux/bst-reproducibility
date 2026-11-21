'Carry the fluoride quantification settings in the fifteen collective preparations.'
import json,sys
import numpy as np
from readout_ir import I,O,save
sys.path.insert(0,str(I/'COLLECTIF_R478'))
from quantum import Action,CASES,ROOT,label,covariance,probabilities,frozen_settings
def main():
    units=json.loads((O/'UNITES_ET_RUPTURE_R573.json').read_text());records={}
    archived=json.loads((ROOT/'resultats/QUANTIFICATION_COMMUNE_R478.json').read_text())['cases']
    hbs={k:v['hbar_native'] for k,v in units['actions_SFx'].items()}
    hbs['R524_eau']=units['hbar_native_H2O'];hbs['controle_R478']=.001
    da,db,_=frozen_settings()
    for ct in CASES:
        n,length,weight=ct;key=label(*ct)
        old=next(v for v in archived if v['N']==n and v['length_ratio']==length and v['gauge_weight']==weight)
        z=np.load(ROOT/f'resultats/OPERATEUR_{key}.npz');V,w=covariance(z['M'],z['H']);ch=np.linalg.cholesky(V[:4,:4])
        acts=Action(*ct);c=z['c'];Vc=acts.potential(c);g=acts.gradient(c);K=z['H']
        rows={}
        for name,hb in hbs.items():
            probs=probabilities((hb*V)/hb,np.sqrt(hb)*da/np.sqrt(hb),np.sqrt(hb)*db/np.sqrt(hb))
            error=max(np.max(abs(np.array(x['p'])-y['p'])) for x,y in zip(probs['settings'],old['gaussian_prediction']['settings']))
            defects=[]; chart_losses=0
            for path in [*np.eye(4),np.ones(4)/2,np.array([1,-1,1,-1])/2]:
                for sign in (-1,1):
                    d=sign*3*np.sqrt(hb)*(ch@path);quad=.5*d@K@d
                    try:
                        val=acts.potential(c+d)
                    except AssertionError as exc:
                        import traceback
                        frame=traceback.extract_tb(exc.__traceback__)[-1]
                        if frame.name not in ('polar','links'):raise
                        chart_losses+=1;continue
                    defects.append(abs(val-Vc-g@d-quad)/quad)
            if name=='controle_R478':
                orig=max(r['relative_potential_error'] for r in old['nonlinear_potential_probes'] if r['hbar']==.001)
                assert abs(orig-max(defects))<2e-10
            assert error<1e-9
            rows[name]=dict(hbar=hb,defects_of_12_probes=defects,maximum=max(defects) if defects else None,probes_outside_carte=chart_losses,deviations_levels= (hb*w).tolist(),CHSH=probs['CHSH'],defect_probabilities=error)
        records[key]=rows
    result={k:dict(geometries_under_10pct=[tag for tag,v in records.items() if v[k]['probes_outside_carte']==0 and v[k]['maximum']<=.1],maximum_probes_in_carte=max(v[k]['maximum'] for v in records.values() if v[k]['maximum'] is not None),count_probes_outside_carte=sum(v[k]['probes_outside_carte'] for v in records.values())) for k in hbs}
    save('QUANTIFICATION_ET_FLUORURES_R573.json',dict(preparations=records,summary=result,hypotheses='Interjacent action evaluated on twelve displacements with three standard deviations by geometry and quantification. Exact homogenity of the probabilities of the quadratic model. Finished probes control local approximation; they do not limit all displacements or all quantum states. Same number of actions in the declared native coordinates, cross-sectional physical identification still conditional.'))
    print(result)
if __name__=='__main__':main()
