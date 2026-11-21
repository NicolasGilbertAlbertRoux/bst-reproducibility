'Cross the unretuned CMB-normalized transfer with archived eBOSS covariance.\nScores are kept separate from DESI because catalog volumes can overlap.'
from pathlib import Path
import json
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import chi2
from scipy.integrate import quad
from cosmology_R479 import ROOT,DATA,background,read,save,C_LIGHT

def main():
    obs=[];cov=[]
    for tag in ['LRG','QSO']:
        stem=f'sdss_DR16_BAOplus_{tag}_FSBAO_DMDHfs8'
        obs.append(np.genfromtxt(DATA/(stem+'.dat'),dtype=None,encoding='utf-8'))
        cov.append(np.loadtxt(DATA/(stem+'_covtot.txt')))
    observations=np.concatenate(obs);covariance=block_diag(*cov);assert covariance.shape==(12,12)
    is_growth=observations['f2']=='f_sigma8';is_distance=~is_growth
    covariance_cond=covariance[np.ix_(is_growth,is_growth)]-covariance[np.ix_(is_growth,is_distance)]@np.linalg.solve(covariance[np.ix_(is_distance,is_distance)],covariance[np.ix_(is_distance,is_growth)])
    rows=[]
    for index,p in enumerate(read(ROOT/'entrees/CANDIDATS_COSMOLOGIQUES_R479.json')):
        archive=read(ROOT/f'resultats/COSMO_C{index:02d}.json')
        inv,dist,*_=background(p)
        rd=archive['derived']['rdrag'];prediction=[]
        zarr=np.array(archive['redshifts'])
        for z,_,kind in observations:
            if kind=='f_sigma8':
                j=int(np.argmin(abs(zarr-z)));assert abs(zarr[j]-z)<1e-8
                prediction.append(archive['fsigma8'][j])
            else:
                dm=dist(z)*C_LIGHT/(100*p['h']);dh=C_LIGHT/(100*p['h']*(1+z)**2/inv(1/(1+z)))
                prediction.append(dict(DM_over_rs=dm,DH_over_rs=dh,DV_over_rs=(z*dm*dm*dh)**(1/3))[kind]/rd)
        prediction=np.array(prediction);residual=prediction-observations['f1']
        score=float(residual@np.linalg.solve(covariance,residual))
        e=residual[is_distance]
        conditional_res=residual[is_growth]-covariance[np.ix_(is_growth,is_distance)]@np.linalg.solve(covariance[np.ix_(is_distance,is_distance)],e)
        ds=float(e@np.linalg.solve(covariance[np.ix_(is_distance,is_distance)],e))
        gs=float(conditional_res@np.linalg.solve(covariance_cond,conditional_res))
        assert abs(ds+gs-score)<1e-10
        rows.append(dict(index=index,fraction_flux=p['fraction_flux'],sigma0=p['sigma0'],prediction=prediction.tolist(),
            joint_chi2=score,distance_chi2=ds,conditional_growth_chi2=gs,
            diagnostic_pass=bool(score<chi2.ppf(.95,12)),amplitude_refitted=False))
    result=dict(revision=479,rows=rows,observations=[dict(z=float(z),value=float(v),observable=str(k)) for z,v,k in observations],
        covariance=covariance.tolist(),threshold_inherited_from_R453=float(chi2.ppf(.95,12)),
        distinct_from_DESI_score=True,source='Alam et al., PRD 103,083533 (2021), official DR16 BAO+FS products archived in R453.',
        physical_validation=False)
    save(ROOT/'resultats/CROISSANCE_RESERVEE_R479.json',result)
    print([(x['index'],round(x['joint_chi2'],5),x['diagnostic_pass']) for x in rows],flush=True)

if __name__=='__main__':main()
