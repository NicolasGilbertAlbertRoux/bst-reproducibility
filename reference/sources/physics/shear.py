'Sowing both potentials at the same dense state as the viscosity bank R444.'
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import numpy as np
from scipy.constants import Boltzmann,Avogadro
from scipy.signal import correlate
from scipy.integrate import cumulative_trapezoid
from viscosites_reference import reference
from readout_ir import ROOT,I,decode,save

def main(full=False,reprise=None):
    gas_data=decode('VIRIELS_R569/entrees/R453/entrees/viriels/POTENTIELS_ET_ETATS_R442.json')['configs']
    disp=decode('VIRIELS_R569/entrees/R453/resultats/DISPERSION_R453.json')['espèces']
    geo=decode('RAYONS_R569.json');branches=decode('THERMO_R569.json')['trials'];windows=np.array([2,4,6,8,10,12,16,24,32,48])
    if reprise:Path(reprise).mkdir(parents=True,exist_ok=True)
    with (nullcontext(reprise) if reprise else TemporaryDirectory()) as d:
        temp=Path(d);source=(ROOT/'code/transport_R573.cpp').read_text()
        # Playage of the voltage at the same pitch 0.02 as in R444; the binary trajectory is not necessary for shearing.
        source=source.replace('std::lround(.2/dt)','std::lround(.02/dt)')
        source=source.replace('if (i%stride_pv==0)', 'if (false)')
        cpp=temp/'forces.cpp';cpp.write_text(source);exe=temp/'forces';subprocess.run(['c++','-O3','-std=c++17',str(cpp),'-o',str(exe)],check=True)
        def run(job):
            g,m,n,seed,duration=job;ident=f'{g}_{m}_{n}_{seed}_{duration}';mat=gas_data[g];sigma=geo['lengths_calibrees_m'][g];s=sigma/mat['sigma_m'];e=mat['energie_J']/s**6
            T=400.;Tred=Boltzmann*T/e;rho=.08/(Avogadro*mat['sigma_m']**3);rs=rho*Avogadro*sigma**3
            pair=np.loadtxt(I/'VIRIELS_R569/entrees/R453/entrees/viriels'/f'POTENTIEL_{g}_R444.dat');pair[10]=disp[g]['C9']/s**3
            ell=np.log(sigma/3.63e-10)/.05;b=np.array(branches[m]['coefficients']);amp=.09*np.r_[b[:2]+ell*b[2:4],b[4:6]];amp[0]+=.09*(T-350)/100*b[-1] if m==1 else 0.
            pf=temp/(ident+'.dat');of=temp/(ident+'.csv');np.savetxt(pf,np.r_[pair,2.45,amp][None])
            dt=.005;cells=4 if n==256 else 5
            finished=False
            if reprise and of.exists():
                with of.open('rb') as file:
                    file.seek(max(0,of.stat().st_size-768))
                    end=file.read().splitlines()
                try:finished=(float(end[-1].split(b',')[0])==duration and end[-1].count(b',')==11)
                except (ValueError,IndexError):pass
            if not finished:
                trial=of.with_suffix('.part')
                subprocess.run(list(map(str,[exe,trial,cells,rs,Tred,pf,dt,4.5,seed,round(duration/dt),round(100./dt),1])),capture_output=True,text=True,check=True)
                trial.replace(of)
            tab=np.loadtxt(of,delimiter=',',skiprows=1)
            assert len(tab)==1+round(duration/.02) and abs(tab[-1,0]-duration)<1e-6
            digest=hashlib.sha256(of.read_bytes()).hexdigest()
            def analysis(tab):
                delta=tab[1,0];shear=tab[:,5:10];center=shear-shear.mean(axis=0)
                steps=len(tab);lags=np.arange(round(max(windows)/delta)+1);ct=np.mean([correlate(x,x,method='fft')[steps-1+lags]/(steps-lags) for x in center.T],axis=0)
                fac=np.sqrt(mat['mass_kg']*e)/sigma**2*1e6/(tab[:,1].mean()*n/rs);integ=cumulative_trapezoid(ct,dx=delta,initial=0)
                idx=np.rint(windows/delta).astype(int);gk=fac*integ[idx];sums=cumulative_trapezoid(center,dx=delta,axis=0,initial=0)
                def moment(t):
                    k=round(t/delta);return float(np.mean((sums[k:]-sums[:-k])**2))
                he=np.array([(moment(w+.5)-moment(w-.5))*fac/2 for w in windows]);Tactual=tab[:,1].mean()*T/Tred
                ref=reference(g,Tactual,rho);plat=np.isin(windows,[6,8,10,12])
                out=dict(gas=g,branch=m,N=n,seed=seed,sha256_trajectory=digest,duration_field=float(tab[-1,0]),T_moy_K=Tactual,rho_mol_m3=rho,reference_uPa_s=ref,
                  windows_field=windows.tolist(),Green_Kubo=gk.tolist(),Helfand=he.tolist(),GK_plateau=float(np.mean(gk[plat])),Helfand_plateau=float(np.mean(he[plat])),
                  deviations_plateaux_pct=(100*(np.array([gk[plat].mean(),he[plat].mean()])/ref-1)).tolist(),
                  relaxation_ps=(integ[idx]/ct[0]*sigma*np.sqrt(mat['mass_kg']/e)*1e12).tolist(),
                  check_energy=float(np.ptp(tab[:,4])/abs(tab[0,4])))
                return out
            horizons=[640,800,960] if full and n==500 else [duration]
            rows=[analysis(tab[:round(t/.02)+1]) for t in horizons]
            print(ident,rows[-1]['deviations_plateaux_pct'],flush=True)
            return rows
        if full:
            jobs=[(g,0,n,seed,duration) for g in ('Kr','Xe') for seed in [44301,44309,44311,44317,44321,44327] for n,duration in [(256,800),(500,960)]]
        else:
            jobs=[(g,m,256,44301,320) for m in (0,1) for g in ('Kr','Xe')]
        with ThreadPoolExecutor(max_workers=8 if full else 4) as pool:
            records=[r for batch in pool.map(run,jobs) for r in batch]
    if full:
        from scipy.stats import t as student
        groups=[]
        for g in ('Kr','Xe'):
            for n,dur in [(256,800),(500,640),(500,800),(500,960)]:
                vec=[r for r in records if r['gas']==g and r['N']==n and r['duration']==dur]
                assert len(vec)==6,(g,n,dur,len(vec))
                ref=np.mean([r['reference_uPa_s'] for r in vec]);row=dict(gas=g,N=n,duration_field=dur,reference=ref)
                for key in ('Green_Kubo','Helfand'):
                    data=np.array([r[key] for r in vec])[:,[2,3,4,5]]
                    curve=data.mean(axis=0);centers=data.mean(axis=1);mean=centers.mean();ci=student.ppf(.975,len(vec)-1)*centers.std(ddof=1)/np.sqrt(len(vec))
                    row[key]=dict(plateau=mean,deviations_windows_pct=(100*(curve/ref-1)).tolist(),IC95=[mean-ci,mean+ci],deviation_center_pct=100*(mean/ref-1))
                groups.append(row)
        save('CISAILLEMENT_COMPLET_R573.json',dict(groups=groups,trajectoires=records,scope='Thermal branch 0, same as optical rays, 24 trajectories R444 to 400 K, six seeds and two sizes, two estimators and durations 640–960. The pressure and the preparation density, the windows and the seed are prescribed independently of the calculated tensile measurements. The 4 tray windows already defined are retained.'))
    else:
        save('CISAILLEMENT_R573.json',dict(trajectoires=records,scope='Four trajectories of 256 centers out of 320 units, one seed, same physical densities and plateau windows as R444. Reference values of the fluids at the T achieved. The complete historical bench also employs 500 centres and six seeds over extended periods.'))
if __name__=='__main__':
    import sys
    main('--complet' in sys.argv,sys.argv[sys.argv.index('--reprise')+1] if '--reprise' in sys.argv else None)
