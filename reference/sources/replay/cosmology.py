'R453 backgrounds passed unchanged to an explicit external PPF/Boltzmann completion.\nPPF is a phenomenological closure, NOT a derived substrate action.'
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parent
if (ROOT.parent/'runtime').is_dir():sys.path.insert(0,str(ROOT.parent/'runtime'))
import numpy as np
import camb
from camb import model,dark_energy
from scipy.interpolate import CubicSpline
from scipy.integrate import quad,simpson
from scipy.linalg import block_diag
from background_archive import mixture
DATA=ROOT/'entrees/cosmologie'
C_LIGHT=299792.458

def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def radiation(ref):
    return ref['omega_gamma_h2']*(1+ref['N_eff']*(7/8)*(4/11)**(4/3))

def background(p,resolution=1):
    ref=read(DATA/'REPERES_R449.json')['readout_baryon_photon']
    inv,dist,rho=mixture(p['omega_m'],radiation(ref)/p['h']**2,p['sigma0'] or 1.35,p['fraction_flux'],2e-11)
    a=np.geomspace(1e-7,1,1800*resolution)
    r=np.array([rho(1/x-1) for x in a]);assert r.min()>0
    log_r=CubicSpline(np.log(a),np.log(r));w=-1-log_r(np.log(a),1)/3
    # rho+p, not just background agreement, diagnoses a canonical scalar reconstruction.
    canonical_kinetic_over_rho=(1+w)/2
    delta=2e-5;test=np.linspace(np.log(.02),np.log(.99),90)
    finite=(log_r(test+delta)-log_r(test-delta))/(2*delta)
    defect=float(np.max(abs(finite-log_r(test,1))))
    return inv,dist,rho,a,r,w,dict(continuity_finite_difference_defect=defect,
        w_range=[float(w.min()),float(w.max())],
        canonical_scalar_kinetic_over_rho_min=float(canonical_kinetic_over_rho.min()),
        canonical_single_scalar_reconstruction_allowed=bool(w.min()>=-1-1e-8))

def bin_spectrum(ell,cl,channel):
    f=next(DATA.glob(f'COM_PowerSpect_CMB-{channel}-binned_*.txt'))
    obs=np.loadtxt(f);selected=obs[:,0]<=2000;obs=obs[selected]
    rows=[]
    for center,value,minus,plus,official in obs:
        # Published diagnostic tables use width 30 here. Their centers exactly
        # match the l(l+1)-weighted centers, checked rather than guessed.
        first=30*int(center//30);last=min(first+29,1996 if channel in ['TE','EE'] else 2508)
        x=np.arange(first,last+1);weights=x*(x+1)
        center_field=weights@x/weights.sum();assert abs(center_field-center)<1e-5
        values=np.interp(x,ell,cl)
        pred=float(weights@values/weights.sum())
        point=float(np.interp(center,ell,cl))
        sigma=plus if pred>=value else minus
        rows.append(dict(ell=float(center),ell_first=first,ell_last=last,
            observed=float(value),sigma=float(sigma),predicted=pred,point_prediction=point,
            official_binned_bestfit=float(official)))
    score=sum(((r['predicted']-r['observed'])/r['sigma'])**2 for r in rows)
    return dict(rows=rows,chi2_diagonal=float(score),n=len(rows),chi2_per_bin=float(score/len(rows)),
        binning='Diagnostic width-30 l(l+1)-weighted Dl; sensitivity to point reader recorded; official likelihood/windows/covariance not reconstructed.',
        max_point_vs_bin_in_sigma=max(abs(r['predicted']-r['point_prediction'])/r['sigma'] for r in rows))

def bao_scores(results):
    obs=np.genfromtxt(DATA/'DESI_DR2_mean.txt',dtype=None,encoding='utf-8');cov=np.loadtxt(DATA/'DESI_DR2_cov.txt')
    rd=results.get_derived_params()['rdrag'];pred=[]
    for z,_,kind in obs:
        dm=results.comoving_radial_distance(z);dh=C_LIGHT/results.hubble_parameter(z)
        pred.append(dict(DM_over_rs=dm,DH_over_rs=dh,DV_over_rs=(z*dm*dm*dh)**(1/3))[kind]/rd)
    pred=np.array(pred);res=pred-obs['f1'];scores={}
    for name,mask in [('near',obs['f0']<1),('far',obs['f0']>=1)]:
        q=res[mask];scores[name]=float(q@np.linalg.solve(cov[np.ix_(mask,mask)],q))
    return dict(predicted=pred.tolist(),observed=obs['f1'].tolist(),chi2=scores,
        note='Same archived DESI covariance, no readjusted acoustic ruler.')

def make_params(p,a,w,accuracy=1.):
    ref=read(DATA/'REPERES_R449.json')['readout_baryon_photon']
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=p['h']*100,ombh2=ref['omega_b_h2'],omch2=p['omega_m']*p['h']**2-ref['omega_b_h2'],
        mnu=0,num_massive_neutrinos=0,nnu=ref['N_eff'],tau=.0544,TCMB=2.7255)
    pars.InitPower.set_params(As=2.1e-9,ns=.9649)
    of=dark_energy.DarkEnergyPPF();of.set_params(cs2=1.);of.set_w_a_table(a,w);pars.DarkEnergy=of
    pars.set_for_lmax(2500,lens_potential_accuracy=0)
    pars.set_accuracy(AccuracyBoost=accuracy,lAccuracyBoost=accuracy,lSampleBoost=accuracy)
    zs=[1.48,.698,.61,.51,.38,0.]
    pars.set_matter_power(redshifts=zs,kmax=2.,nonlinear=False,silent=True)
    pars.NonLinear=model.NonLinear_none
    return pars

def run(index,accuracy=1.,table_resolution=1):
    p=read(ROOT/'entrees/CANDIDATS_COSMOLOGIQUES_R479.json')[index]
    tag=f'C{index:02d}'+('' if accuracy==1 else '_fin')
    out=ROOT/f'resultats/COSMO_{tag}.json'
    if out.exists():return read(out)
    ref=read(DATA/'REPERES_R449.json')['readout_baryon_photon']
    inv,dist,rho,a,r,w,checks=background(p,table_resolution)
    pars=make_params(p,a,w,accuracy);zs=[1.48,.698,.61,.51,.38,0.]
    results=camb.get_results(pars);derived=results.get_derived_params()
    spectra=results.get_cmb_power_spectra(CMB_unit='muK',lmax=2500)['total']
    ell=np.arange(len(spectra));assert np.isfinite(spectra).all()
    assert np.min(spectra[2:,0]*spectra[2:,1]-spectra[2:,3]**2)>-1e-7
    channels={name:bin_spectrum(ell,spectra[:,col],name) for name,col in [('TT',0),('TE',3),('EE',1)]}
    zcheck=np.array([0.,.1,.5,1.,3.,10.,100.,1000.,1e5]);H=np.array(results.hubble_parameter(zcheck))
    original_H=np.array([100*p['h']*(1+z)**2/inv(1/(1+z)) for z in zcheck])
    background_error=float(np.max(abs(H/original_H-1)))
    rt,wt=results.get_dark_energy_rho_w(a)
    density_error=float(np.max(abs(rt/r-1)))
    # Geometrical projection and horizon use the same clock, with recombination computed.
    zstar=derived['zstar'];astar=1/(1+zstar)
    load=.75*ref['omega_b_h2']/ref['omega_gamma_h2']
    rs=quad(lambda aa:inv(aa)/np.sqrt(3*(1+load*aa)),0.,astar,epsabs=1e-12)[0]*C_LIGHT/(100*p['h'])
    eta=quad(inv,0.,astar,epsabs=1e-12)[0]*C_LIGHT/(100*p['h'])
    age=quad(lambda aa:aa*inv(aa),0.,1.,epsabs=1e-11)[0]
    theta=100*rs/(dist(zstar)*C_LIGHT/(100*p['h']))
    transfer=results.get_matter_transfer_data();q=transfer.q
    # Independent reconstruction of P(k) from curvature and matter transfer.
    T=transfer.transfer_data[6,:,0].astype(float);P=2*np.pi**2/q**3*(2.1e-9*(q/.05)**(.9649-1))*(T*q*q)**2
    k,zouts,pk=results.get_linear_matter_power_spectrum(hubble_units=False,k_hunit=False,nonlinear=False)
    reference_pk=pk[-1] # Highest z: CAMB returns redshifts in increasing order.
    power_error=float(np.max(abs(P/np.interp(q,k,reference_pk)-1)))
    xx=q*8/p['h'];window=3*(np.sin(xx)-xx*np.cos(xx))/xx**3
    sigma_rebuilt=float(np.sqrt(simpson(q**3*P/(2*np.pi**2)*window**2,x=np.log(q))))
    sigma8=results.get_sigma8();fsigma8=results.get_fsigma8()
    # Original approximate matter/radiation constants differ slightly from CAMB's constants.
    assert background_error<2e-4 and density_error<1e-5
    assert abs(rs/derived['rstar']-1)<3e-4
    assert abs(eta/results.conformal_time(zstar)-1)<3e-4
    assert power_error<2e-5 and abs(sigma_rebuilt/sigma8[0]-1)<.003
    checks.update(background_H_relative_error=background_error,density_table_relative_error=density_error,
        independent_sound_horizon_Mpc=rs,independent_conformal_time_at_star_Mpc=eta,
        sound_horizon_relative_error=rs/derived['rstar']-1,independent_age_H0=age,
        theta100_from_independent_integrals=theta,power_from_transfer_relative_error=power_error,
        sigma8_from_integral_at_zmax=sigma_rebuilt,sigma8_integral_relative_error=sigma_rebuilt/sigma8[0]-1)
    bao=bao_scores(results)
    candidate_pass=all(s['chi2_per_bin']<=2 for s in channels.values()) and all(v<=12.591587243743977 for v in bao['chi2'].values())
    doc=dict(revision=479,id=tag,parameters=p,camb_version=camb.__version__,accuracy=accuracy,
        background_checks=checks,derived=derived,CMB=channels,BAO=bao,redshifts=zs,
        sigma8=sigma8.tolist(),fsigma8=fsigma8.tolist(),diagonal_joint_screen_pass=bool(candidate_pass),
        background_law_preserved=True,numerical_checks_pass=True,PPF_closure_derived_from_BST=False,
        physical_validation=False,calibration_per_channel=False,neutrinos_massless_approximation=True)
    np.savez_compressed(ROOT/f'resultats/COSMO_{tag}.npz',ell=ell,spectra=spectra,a=a,rho_relative=r,w=w,
        k_Mpc=q,matter_transfer=transfer.transfer_data,power_spectrum_zmax=P)
    save(out,doc)
    print(tag,'CMB',{k:round(v['chi2_per_bin'],3) for k,v in channels.items()},'BAO',bao['chi2'],'pass',candidate_pass,flush=True)
    return doc

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--index',type=int);cli.add_argument('--accuracy',type=float,default=1.);args=cli.parse_args()
    cases=range(len(read(ROOT/'entrees/CANDIDATS_COSMOLOGIQUES_R479.json'))) if args.index is None else [args.index]
    rows=[run(i,args.accuracy,2 if args.accuracy>1 else 1) for i in cases]
    if args.index is None:save(ROOT/'resultats/COSMOLOGIE_COMMUNE_R479.json',dict(revision=479,cases=rows))
