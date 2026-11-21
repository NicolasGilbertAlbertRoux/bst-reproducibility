'Apply the size, duration, window and estimator thresholds entered prior to the re-calibration.'
import json
import numpy as np
from readout_ir import I,O,decode,save
def main():
    r=json.loads((O/'CISAILLEMENT_COMPLET_R573.json').read_text())
    co=decode('TRANSPORT_R485/CRITERES_CISAILLEMENT_R444.json')
    results=[];lower=0.;upper=float('inf');size_lo=0.;size_hi=float('inf');time_methods_ok=True
    for g in ('Kr','Xe'):
        def group(n,t):return next(a for a in r['groups'] if a['gas']==g and a['N']==n and a['duration']==t)
        s=group(256,800);b=group(500,800);d=group(500,960)
        ctrl={}
        for k in ('Green_Kubo','Helfand'):
            y=1+np.array(d[k]['deviations_windows_pct'])/100
            lo=float(np.max((1-co['cutoff_error'])/y));hi=float(np.min((1+co['cutoff_error'])/y))
            lower=max(lower,lo);upper=min(upper,hi)
            size_ratio=s[k]['plateau']/b[k]['plateau']
            size_lo=max(size_lo,(1-co['size_error'])/size_ratio);size_hi=min(size_hi,(1+co['size_error'])/size_ratio)
            time_methods_ok=time_methods_ok and abs(b[k]['plateau']/d[k]['plateau']-1)<=co['duration_error']
            ctrl[k]=dict(duration=abs(b[k]['plateau']/d[k]['plateau']-1),size_field=abs(s[k]['plateau']/b[k]['plateau']-1),worst_window=float(np.max(abs(y-1))),interval_gain=[lo,hi],IC95=d[k]['IC95'])
        diff=abs(d['Green_Kubo']['plateau']/d['Helfand']['plateau']-1)
        time_methods_ok=time_methods_ok and diff<=co['estimators_error']
        size=all(z['duration']<=co['duration_error'] and z['size']<=co['size_error'] for z in ctrl.values()) and diff<=co['estimators_error']
        results.append(dict(gas=g,checks=ctrl,deviation_estimators=diff,size_duration_estimators_admitted=size,tout_brut_admitted=size and all(z['worst_window']<=co['cutoff_error'] for z in ctrl.values())))
    interval=[lower,upper] if lower<=upper else None
    gain=(1. if lower<=1<=upper else .5*(lower+upper)) if interval else None
    # g is f(256)/f(500) for a single coefficient b in f(N)-1.pi.(1/3).
    # The amplitude is normalized to 500 to preserve comparison with the two gases.
    factor_range=[size_lo,size_hi] if size_lo<=size_hi else None
    witness=1. if factor_range and size_lo<=1.<=size_hi else ((size_lo+size_hi)/2 if factor_range else None)
    b=(witness-1)/(256**(-1/3)-witness*500**(-1/3)) if witness is not None and witness<((500/256)**(1/3)) else None
    finite=dict(interval_ratio_256_over_500=factor_range,ratio_control=witness,coefficient_of_size=b,others_ratios_admitted=bool(time_methods_ok),passe=bool(factor_range and b is not None and interval and time_methods_ok),scope='Common reading correction g over both available sizes. The long amplitude and the size ratio apply to both gases and both estimators. The two-size shape is a retrospective calibration, without confirmation of a thermodynamic extrapolation.')
    save('BILAN_CISAILLEMENT_R573.json',dict(branch=0,trajectoires=24,seeds=6,groups=8,results_field=results,thresholds={k:co[k] for k in ('duration_error','size_error','cutoff_error','estimators_error')},interval_gain_viscosite=interval,gain_control=gain,correction_of_size=finite,brut_admitted=all(x['tout_brut_admitted'] for x in results),admitted_with_gain_of_readout=bool(interval and all(x['size_duration_estimators_admitted'] for x in results)),note='A multiplicative gain on the viscosity compares the transport of the reader with common constraints. It preserves the ratios of duration, size and estimators. It is recorded as an effective calibration if used. The coefficients of the potential are unchanged. No conclusion on all mixtures derives from the two pure gases.'))
    print(results,interval)
if __name__=='__main__':main()
