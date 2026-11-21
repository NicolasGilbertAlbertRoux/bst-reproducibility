'NVE dynamics with pairing forces and four short-range corrections of the R569.'
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import json,subprocess,sys
import numpy as np
from scipy.constants import Avogadro,Boltzmann
from scipy.optimize import brentq
from scipy.constants import R
from readout_ir import I,O,decode,save
sys.path.insert(0,str(I/'TRANSPORT_R485'))
from transport import msd
from analyze_gas import acf

def main():
    d=I/'TRANSPORT_R485'
    states=json.loads((d/'ETATS_GAZ_R485.json').read_text())
    contract=json.loads((d/'CONTRAT_R485.json').read_text())
    previous=json.loads((d/'DIFFUSION_DYNAMIQUE_R484.json').read_text())
    geo=decode('RAYONS_R569.json');pot=I/'VIRIELS_R569/entrees/R453'
    dispersion=json.loads((pot/'resultats/DISPERSION_R453.json').read_text())['espèces']
    vir=decode('THERMO_R569.json')['trials'];cal=json.loads((O/'AJUSTEMENT_ENERGIE_LIBRE_R573.json').read_text())['branches']
    table=json.loads((O/'VIRIELS_ETENDUS_R573.json').read_text())['calculs'];T=contract['nominal_T'];windows=contract['diffusion_windows']
    with TemporaryDirectory() as tmpdir:
        tmp=Path(tmpdir);binary=tmp/'transport'
        subprocess.run(['c++','-O3','-std=c++17',str(Path(__file__).with_name('transport_R573.cpp')),'-o',str(binary)],check=True)
        def run(job):
            m,g,plan=job;n=4*plan['cells']**3;dt=plan['dt'];length=plan['duration']
            ident=f'{m}_{g}_{n}_{dt}';tab=tmp/(ident+'.csv');par=tmp/(ident+'.dat')
            st=states[g]['constants'];sigma=geo['lengths_calibrees_m'][g];s=sigma/st['sigma_m']
            e=st['energie_J']/s**6;tu=sigma*np.sqrt(st['mass_kg']/e);Tr=T*Boltzmann/e
            vals=np.loadtxt(pot/'entrees/viriels'/f'POTENTIEL_{g}_R444.dat');vals[10]=dispersion[g]['C9']/s**3
            b=np.array(vir[m]['coefficients']);ell=np.log(sigma/3.63e-10)/.05
            amp=.09*np.r_[b[:2]+ell*b[2:4],b[4:6]];amp[0]+=.09*(T-350.)/100*b[-1] if m==1 else 0.
            np.savetxt(par,np.r_[vals,2.45,amp][None])
            _,B,C=next(r for r in table[g][str(m)] if abs(r[0]-T)<1e-8);a,b=cal[str(m)][0]['coefficients']
            B+=(a+b*np.log(geo['radii_recalcules'][g]))*(400./T-1)**2*1e-6
            density=brentq(lambda x:x*R*T*(1+B*x+C*x*x)-contract['nominal_P'],1,300);rho=density*Avogadro*sigma**3
            args=[binary,tab,plan['cells'],rho,Tr,par,dt,4.5,contract['new_seed'],round(length/dt),round(40/dt),1]
            sub=subprocess.run(list(map(str,args)),check=True,capture_output=True,text=True)
            raw=np.memmap(str(tab)+'.pv',mode='r',dtype='float64',shape=(length+1,2,n,3));hist=np.loadtxt(tab,delimiter=',',skiprows=1);t=np.arange(641)
            base=next(z for z in previous['rows'] if z['gas']==g);D=base['D_infinite_closure_m2_s'];tau=base['tau_s']/tu;unit=sigma**2/tu
            MS=6*D/unit*(t+tau*np.expm1(-t/tau))
            def compare(arr):
                pred=[];obs=[]
                for lo,hi in windows:
                    idx=slice(lo,hi+1);pred.append(float(np.polyfit(t[idx],MS[idx],1)[0]*unit/6))
                    obs.append(float(np.polyfit(t[idx],arr[idx],1)[0]*unit/6))
                return dict(diffusion_native_m2_s=obs,closure_m2_s=pred,deviations=(np.array(pred)/obs-1).tolist())
            whole=compare(msd(raw[:,0],640));half=[compare(msd(raw[first:last,0],640)) for first,last in [(0,length//2+1),(length//2,length+1)]]
            cc=acf(raw[:,1],640);v=cc/(3*Tr)-np.exp(-t/tau)
            out=dict(gas=g,branch=m,N=n,dt=dt,duration=length,force_short=amp.tolist(),radius_m=sigma,energy_J=e,mass_kg=st['mass_kg'],unit_time_s=tu,density_molar=density,
                temperature_mean_K=hist[:,1].mean()*T/Tr,pressure_mean_Pa=hist[:,2].mean()*e/sigma**3,check_forces=sub.stdout.strip(),
                integer=whole,halves=half,VACF_deviation_max_normalized=float(abs(v).max()),energy_derivation_relative=float(np.ptp(hist[:,4])/abs(hist[0,4])))
            del raw;tab.unlink();Path(str(tab)+'.pv').unlink()
            print(ident,whole['deviations'],flush=True);return out
        jobs=[(m,g,p) for m in range(2) for g in ('Kr','Xe') for p in contract['plans']]
        with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(run,jobs))
    res={}
    for m in (0,1):
        zz=[x for x in records if x['branch']==m];w=np.array([x['integer']['deviations'] for x in zz]);h=np.array([s['deviations'] for x in zz for s in x['halves']])
        res[str(m)]=dict(integers_in_10pct=int(np.sum(abs(w)<=.1)),integers=int(w.size),max_integers_pct=100*float(abs(w).max()),halves_in_10pct=int(np.sum(abs(h)<=.1)),halves=int(h.size),max_halves_pct=100*float(abs(h).max()),VACF_max=max(x['VACF_difference_max_normalized'] for x in zz))
    save('TRANSPORT_COMMUN_R573.json',dict(branches=res,trajectoires=records,conditions='Kr/Xe à 298,15 K et 303 975 Pa nominaux. Deux branches, une graine archivée, deux tailles, deux pas, douze trajectoires NVE. D et temps de relaxation en unités SI repris de R484 ; potentiel R569, coupures dynamiques R485. La pression de préparation utilise la correction de B2. Ce terme uniforme à densité fixée ne produit pas de force tangentielle. Les sorties quantifient la diffusion et la corrélation de vitesse. La viscosité en mélange ou aux états denses de R444 ne découle pas de cet essai.'))
    print(res,flush=True)
if __name__=='__main__':main()
