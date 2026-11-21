'Verify Gauss and the covariance on the same induced fields as the spectrum of the fifteen preparations.'
import json
import numpy as np
from quantifier import Action,CASES,ROOT,label,covariance
from collective import fields,exponential,random_algebra,adj,shift
from readout_ir import O,save
def main():
    act=json.loads((O/'UNITES_ET_RUPTURE_R573.json').read_text())['actions_SFx']
    cases=[]
    for n,length,weight in CASES:
        tag=label(n,length,weight);op=np.load(ROOT/f'resultats/OPERATEUR_{tag}.npz')
        a=Action(n,length,weight);cv,_=covariance(op['M'],op['H']);direction=np.linalg.cholesky(cv[:4,:4])[:,0];pars={}
        G=exponential(random_algebra(np.random.default_rng(573+n),(n,n,n,3,3),.35))
        for family,v in act.items():
            c=op['c']+3*np.sqrt(v['hbar_native'])*direction
            m,checks=a.metric(c);u,p,b=fields(a.x,c);ug=np.array([G@x@adj(shift(G,j)) for j,x in enumerate(u)])
            err=abs(a.energy_fields(c,ug,G@p,b)-a.potential(c))
            pars[family]=dict(residual_Gauss=checks['gauss_residual'],defect_covariance=err,mass_min=float(np.linalg.eigvalsh(m).min()),position=c.tolist())
        cases.append(dict(geometry=tag,trials=pars))
    vmax=max(v['residual_Gauss'] for c in cases for v in c['trials'].values());mx=max(v['defect_covariance'] for c in cases for v in c['trials'].values())
    mn=min(v['mass_min'] for c in cases for v in c['trials'].values())
    save('GAUSS_ET_QUANTIFICATION_R573.json',dict(cases_field=cases,residual_Gauss_max=vmax,defect_covariance_max=mx,mass_min=mn,passe=bool(vmax<1e-8 and mx<1e-10 and mn>0),scope='Even finished dip R453 and even collective operator R478. A positive displacement at three standard deviations per family and by geometry. Space links are induced by the fields. The physical selection of diving and spectrum allocation to strong states remain hypotheses.'))
    print(vmax,mx,mn)
if __name__=='__main__':main()
