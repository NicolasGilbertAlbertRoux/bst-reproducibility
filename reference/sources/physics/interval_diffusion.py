'Interval of a single calibration on Kr and Xe and on all transport windows.'
import json
import numpy as np
from scipy.optimize import brentq
from readout_ir import I,O,decode,save
def main():
    sim=json.loads((O/'TRANSPORT_COMMUN_R573.json').read_text())
    old=decode('TRANSPORT_R485/DIFFUSION_DYNAMIQUE_R484.json')
    hist=decode('preuves/INTERVALLES_GAZ_R485.json')['after_common_a_equals_b']
    windows=decode('TRANSPORT_R485/CONTRAT_R485.json')['diffusion_windows'];out={}
    for m in (0,1):
        samples=[]
        for r in sim['trajectoires']:
            if r['branch']!=m:continue
            p=next(a for a in old['rows'] if a['gas']==r['gas']);tau=p['tau_s']/r['unit_time_s']
            for sample in [r['integer'],*r['halves']]:
                for i,(lo,hi) in enumerate(windows):
                    t=np.arange(lo,hi+1,dtype=float);weights=(t-t.mean())/np.sum((t-t.mean())**2)
                    samples.append((p['D_infinite_closure_m2_s']/sample['diffusion_native_m2_s'][i],tau,t,weights))
        def quotients(a):return np.array([a*d*np.dot(w,t+a*tau*np.expm1(-t/(a*tau))) for d,tau,t,w in samples])
        low=brentq(lambda a:min(quotients(a))-.90,.4,2.);high=brentq(lambda a:max(quotients(a))-1.10,.4,2.)
        lower=max(low,hist[0]);upper=min(high,hist[1]);ok=lower<=upper
        a=(1. if lower<=1.<=upper else .5*(lower+upper)) if ok else float(.5*(low+high))
        out[str(m)]=dict(count_comparisons=len(samples),intervals_new_trajectoires=[low,high],interval_parent=hist,intersection_common=[lower,upper] if ok else None,control=a,maximum_pct=100*float(max(abs(quotients(a)-1))),passe=bool(ok))
    save('INTERVALLES_DIFFUSION_R573.json',dict(families=out,method='D and the relaxation time are multiplied by the same number. Their ratio, therefore the initial variance of the velocities, is retained. The 18 whole windows and 36 half-windows are imposed jointly for each branch. The intersection with the historic bench of 216 comparisons retains its original qualification.'))
    print(out)
if __name__=='__main__':main()
