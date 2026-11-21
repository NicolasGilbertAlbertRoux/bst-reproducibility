'Read fresh native VACFs and MSDs without fitting the R484 relaxation law.'
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.fft import rfft,irfft,next_fast_len
from scipy.integrate import cumulative_trapezoid
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'inputs'))
from transport import msd
def acf(v,lag):
 n=len(v);s=next_fast_len(2*n-1,real=True);out=np.zeros(lag+1)
 for start in range(0,v.shape[1],24):
  f=rfft(v[:,start:start+24],n=s,axis=0)
  out+=irfft(np.sum(abs(f)**2,axis=(1,2)),n=s)[:lag+1]
 return out/((n-np.arange(lag+1))*v.shape[1])
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=ROOT/'trajectoires');a=p.parse_args()
 jobs=json.loads((a.cache/'SIMULATIONS_R485.json').read_text());states=json.loads((ROOT/'entrees/ETATS_GAZ_R485.json').read_text());pred=json.loads((ROOT/'entrees/DIFFUSION_DYNAMIQUE_R484.json').read_text())
 out=[];arrays={};win=[(80,160),(160,320),(320,640)]
 for j in jobs:
  gas,N,duration=j['gas'],j['N'],j['duration'];raw=np.memmap(a.cache/(j['file']+'.pv'),mode='r',dtype=np.float64,shape=(duration+1,2,N,3));tab=np.loadtxt(a.cache/j['file'],delimiter=',',skiprows=1)
  d=states[gas];c=d['constants'];theory=next(x for x in pred['rows'] if x['gas']==gas);tau=theory['tau_native'];tu=c['unit_time_s'];conversion=c['sigma_m']**2/tu
  lag=1280;t=np.arange(lag+1);v=acf(raw[:,1],lag);m=msd(raw[:,0],lag)
  amp=3*c['T_reduced'];cv=amp*np.exp(-t/tau);ms=6*theory['D_infinite_closure_m2_s']/conversion*(t+tau*np.expm1(-t/tau))
  areas=6*cumulative_trapezoid(cumulative_trapezoid(v/3,t,initial=0),t,initial=0)
  estimates=[]
  for lo,hi in win:
   obs=np.polyfit(t[lo:hi+1],m[lo:hi+1],1)[0]*conversion/6
   calc=np.polyfit(t[lo:hi+1],ms[lo:hi+1],1)[0]*conversion/6
   estimates.append(dict(window=[lo,hi],MD_D=obs,closure_D=calc,relative_residual=calc/obs-1))
  halves=[]
  for start,end in [(0,duration//2),(duration//2,duration)]:
   mh=msd(raw[start:end+1,0],640)
   halves.append([float(e['closure_D']/(np.polyfit(t[lo:hi+1],mh[lo:hi+1],1)[0]*conversion/6)-1) for e,(lo,hi) in zip(estimates,win,strict=True)])
  # Direct all-origin sums control the FFT implementations independently.
  direct={}
  for ell in (1,37,640):
   vv=float(np.mean(np.sum(raw[:-ell,1]*raw[ell:,1],axis=-1)));mm=float(np.mean(np.sum((raw[ell:,0]-raw[:-ell,0])**2,axis=-1)))
   direct[str(ell)]=dict(VACF_defect=abs(vv-v[ell]),MSD_defect=abs(mm-m[ell]))
   np.testing.assert_allclose(vv,v[ell],rtol=1e-10,atol=1e-12);np.testing.assert_allclose(mm,m[ell],rtol=1e-10,atol=1e-8)
  key=f'{gas}_N{N}_dt{j["dt"]}';arrays[key+'_vacf']=v;arrays[key+'_msd']=m;arrays[key+'_vacf_closure']=cv;arrays[key+'_msd_closure']=ms;arrays[key+'_msd_via_vacf']=areas
  error=(v[:641]-cv[:641])/amp
  row=dict(**j,tau_native=tau,estimates=estimates,half_relative_residuals=halves,VACF_max_absolute_normalized_error=float(abs(error).max()),VACF_RMS_normalized_error=float(np.sqrt(np.mean(error**2))),VACF_zero_ratio=float(v[0]/amp),
    energy_relative_range=float(np.ptp(tab[:,4])/abs(tab[0,4])),T_mean_K=float(298.15*tab[:,1].mean()/c['T_reduced']),P_mean_Pa=float(tab[:,2].mean()*c['energie_J']/c['sigma_m']**3),max_com_velocity=float(abs(tab[:,-1]).max()),
    direct_checks=direct,stationary_VACF_MSD_discrepancy_640=float(areas[640]/m[640]-1))
  out.append(row);print(key,'diffusion',np.round([100*x['relative_residual'] for x in estimates],3),'VACF RMS',row['VACF_RMS_normalized_error'],flush=True)
 np.savez_compressed(ROOT/'resultats/COURBES_GAZ_R485.npz',time_native=t,**arrays)
 whole=[x['relative_residual'] for r in out for x in r['estimates']];half=[x for r in out for seg in r['half_relative_residuals'] for x in seg]
 report=dict(revision=485,new_simulations=len(out),new_seed=48511,parameter_fits=0,results=out,
   max_full_relative_error=float(max(abs(x) for x in whole)),max_half_relative_error=float(max(abs(x) for x in half)),whole_in_10pct=int(sum(abs(x)<=.1 for x in whole)),whole_count=len(whole),halves_in_10pct=int(sum(abs(x)<=.1 for x in half)),halves_count=len(half),
   preserved_R484_conditions=True,physical_validation=False,interpretation='New numerical prediction control on six native NVE runs, with fixed R484 closure. VACF shape tested against native velocities; neither a microscopic derivation of exact exponential relaxation nor finite-density convergence. All windows and failures retained.')
 (ROOT/'resultats/CONTROLE_GAZ_R485.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print('SUMMARY',report['max_full_relative_error'],report['max_half_relative_error'],report['whole_in_10pct'],report['halves_in_10pct'],flush=True)
if __name__=='__main__':main()
