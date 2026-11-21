'Switch the same actual mass deviation from the oscillation bench to the cosmological transfer.'
import sys
import math
from pathlib import Path
import numpy as np
from readout_ir import I,O,decode,save
sys.path.insert(0,str(I/'COSMO_R479'))
import camb
from cosmology_R479 import read,DATA,background,make_params,bin_spectrum,bao_scores,C_LIGHT
def compute(p,tag,mass,growth):
    ref=read(DATA/'REPERES_R449.json')['readout_baryon_photon']
    _,_,_,a,_,w,_=background(p)
    pars=make_params(p,a,w)
    matter=p['omega_m']*p['h']**2; baryon=ref['omega_b_h2']
    # Keep Omegaim h2 fixed by switching a fraction of the cold material to massive modes.
    if mass:
        pars.set_cosmology(H0=100*p['h'],ombh2=baryon,omch2=matter-baryon,mnu=mass,num_massive_neutrinos=1,nnu=ref['N_eff'],tau=.0544,TCMB=2.7255)
        pars.omch2=matter-baryon-pars.omnuh2
    assert abs(pars.omch2+pars.ombh2+pars.omnuh2-matter)<1e-12
    result=camb.get_results(pars)
    specs=result.get_cmb_power_spectra(CMB_unit='muK',lmax=2500)['total'];ell=np.arange(len(specs))
    cmb={name:bin_spectrum(ell,specs[:,n],name)['chi2_per_bin'] for name,n in [('TT',0),('TE',3),('EE',1)]}
    bao=bao_scores(result)['chi2'];rd=result.get_derived_params()['rdrag']
    fs8=result.get_fsigma8();red=np.array([1.48,.698,.61,.51,.38,0.])
    pred=[]
    for row in growth['observations']:
        z=row['z'];kind=row['observable']
        if kind=='f_sigma8':pred.append(float(fs8[np.flatnonzero(abs(z-red)<1e-8)[0]]))
        else:
            dm=result.comoving_radial_distance(z);dh=C_LIGHT/result.hubble_parameter(z)
            pred.append(dict(DM_over_rs=dm,DH_over_rs=dh,DV_over_rs=(z*dm*dm*dh)**(1/3))[kind]/rd)
    cov=np.asarray(growth['covariance']);delta=np.array(pred)-[r['value'] for r in growth['observations']]
    e=float(delta@np.linalg.solve(cov,delta));pass_=max(cmb.values())<=2 and max(bao.values())<=12.591587243743977 and e<=growth['threshold_inherited_from_R453'] and 62<=p['h']*100<=82
    return dict(family=tag,mass_eV=mass,omega_nu_h2=pars.omnuh2,omega_cdm_h2=pars.omch2,
        H0=p['h']*100,rd_Mpc=rd,chi2_cmb_par_band=cmb,chi2_DESI=bao,chi2_eBOSS=e,
        predictions_eBOSS=pred,compatible=bool(pass_))
def main():
    q=decode('RACCORD_COMMUN_Q3_R501.json');dm=q['mass_square_span_eV2'];mass=math.sqrt(dm)
    data=I/'COSMO_R479';growth=read(data/'resultats/CROISSANCE_RESERVEE_R479.json')
    cos=read(data/'resultats/COSMOLOGIE_COMMUNE_R479.json')['cases']
    parents=read(data/'entrees/CANDIDATS_COSMOLOGIQUES_R479.json')
    out=[]
    frozen=compute(parents[0],'C00',0.,growth)
    maxref=max(abs(frozen['chi2_cmb_par_band'][k]-cos[0]['CMB'][k]['chi2_per_bin']) for k in ['TT','TE','EE'])
    assert maxref<.025,(camb.__version__,maxref)
    for i,p in enumerate(parents):
        ans=compute(p,f'C{i:02d}',mass,growth);out.append(ans)
        print(ans['family'],ans['compatible'],ans['chi2_cmb_par_band'],ans['chi2_eBOSS'],flush=True)
    save('COSMOLOGIE_ET_OSCILLATIONS_R573.json',dict(camb=camb.__version__,delta_m2_eV2=dm,mass_eV=mass,amplitude=q['shared_amplitude'],control_sans_mass=frozen,deviation_max_chi2_par_band_R479=maxref,
       families=out,compatibles=sum(v['compatible'] for v in out),thresholds_earlier=dict(CMB_par_band=2,DESI_par_bloc=12.591587243743977,eBOSS=growth['threshold_inherited_from_R453']),
       scope='Minimal effective complement to a massive species and zero reference for light mode; other relativistic contributions are kept in N-eff. The mass parameter is identical to the deviation retained by the reader R501 under this assumption. Background R453, PPF closure, initial conditions, h and total substance fixed; calculation of the remaining cold density. No native identification of the content of the substrate to this relativistic model, nor of the measurement of the sum of all the masses.'))
if __name__=='__main__': main()
