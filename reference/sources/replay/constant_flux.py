'Compare a constant to a mixed term, with the same setting and the same reserves.'
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import least_squares, root

from intervals_R452 import (INPUT, ROOT, acoustic, bao_vector, densities, of_data, profiled_reader,
                             read, read_bao, sha, standard, whiten, write)
from flux_reserves_R450 import expansion


def mixture(matter, radiation, sigma, fraction, tolerance=2e-9, initial_z=99.):
    if fraction==0.:
        f,d=standard(matter,radiation)
        return f,d,lambda z:1.
    if fraction==1.:
        f,d,_,state=expansion(matter,radiation,sigma,tolerance,initial_z)
        return f,d,lambda z:state(z)["rho_DE_sur_rho_crit0"]/state(0.)["rho_DE_sur_rho_crit0"]

    full=1.-matter-radiation
    constant=(1.-fraction)*full
    dark_flux=fraction*full
    ast=1/(1+initial_z)
    n0=np.log(ast)
    finit=1.5*matter*ast/(radiation+1.5*matter*ast)
    nu0=1.686/sigma

    def terms(n,y,normalization):
        a=np.exp(n)
        m,r=matter/a**3,radiation/a**4
        log_d,f=y[:2]
        nu=nu0*np.exp(-log_d)
        g=normalization*nu*np.exp(-.5*nu**2)
        hub2=m+r+constant+g*f
        df=(1.5*m/hub2-f*f-2*f-f*(-3*m-4*r+g*(nu*nu-1)*f*f)/(2*hub2))/(1+g*f/(2*hub2))
        return hub2,df,constant+g*f

    def calculate(x,dense):
        normalization=np.exp(x[1])

        def rhs(n,y):
            hub2,df,_=terms(n,y,normalization)
            return y[1],df,np.exp(-n)/np.sqrt(hub2)

        solution=solve_ivp(rhs,(n0,0.),[x[0],finit,0.],method="DOP853",rtol=tolerance,atol=.06*tolerance,
                           max_step=.13,dense_output=dense)
        assert solution.success
        return solution,normalization

    def mismatch(x):
        solve,amp=calculate(x,False)
        y=solve.y[:,-1]
        _,_,of=terms(0.,y,amp)
        return y[0],(of-full)/dark_flux

    starting=dark_flux/(nu0*np.exp(-.5*nu0**2)*matter**.55)
    x=root(mismatch,[n0+.27,np.log(starting)],tol=2e-9)
    np.testing.assert_allclose(mismatch(x.x),0.,atol=3e-9)
    solve,amp=calculate(x.x,True)

    def early(a):
        return 1/np.sqrt(radiation+matter*a+constant*a**4)

    def inverse(a):
        if a<=ast:
            return early(a)
        n=np.log(a)
        hub2,_,_=terms(n,solve.sol(n),amp)
        return 1/(a*a*np.sqrt(hub2))

    def distance(z):
        a=1/(1+z)
        if a<ast:
            return solve.y[2,-1]+quad(early,a,ast,epsabs=1e-12,epsrel=1e-10)[0]
        return solve.y[2,-1]-solve.sol(np.log(a))[2]

    def relative(z):
        n=-np.log1p(z)
        return (constant if z>initial_z else terms(n,solve.sol(n),amp)[2])/full

    np.testing.assert_allclose(inverse(1.),1.,atol=1e-9)
    return inverse,distance,relative


def main():
    cli=ArgumentParser(description=__doc__)
    cli.add_argument("--sortie",type=Path,default=ROOT/'results')
    a=cli.parse_args()
    ref=read(INPUT/"REPERES_R449.json")['readout_baryon_photon']
    radiation=densities(ref)[0]
    bao,cov=read_bao(INPUT/"DESI_DR2_mean.txt",INPUT/"DESI_DR2_cov.txt")
    use=bao["f0"]<1.
    near,far=bao[use],bao[~use]
    ca,cb=cov[np.ix_(use,use)],cov[np.ix_(~use,~use)]
    assert not np.any(cov[np.ix_(use,~use)])
    light,mcov,removed=of_data()
    assert len(light)==1475
    of, _, _=profiled_reader(light,mcov)
    pantheon=read(INPUT/"RESERVE_SUPERNOVAE_R448.json")
    scov=np.array(pantheon["covariance"])

    @lru_cache(maxsize=5)
    def predict(om,h,fraction,sigma,tolerance=3e-9):
        inv,d,relative=mixture(om,radiation/h**2,sigma,fraction,tolerance)
        return inv,d,relative,acoustic(inv,d,ref)

    def read_near(pars,fraction,sigma,tolerance=3e-9):
        inv,d,_,g=predict(*pars,fraction,sigma,tolerance)
        delta=bao_vector(near,inv,d)/g["rd_H0_sur_c"]-near["f1"]
        theta=(g["theta100"]-ref["theta100"])/ref["theta100_marge"]
        return np.r_[whiten(ca,delta),theta]

    def measures(pars,fraction,sigma,tolerance=3e-9):
        residual=read_near(pars,fraction,sigma,tolerance)
        inv,d,relative,geo=predict(*pars,fraction,sigma,tolerance)
        distant=whiten(cb,bao_vector(far,inv,d)/geo["rd_H0_sur_c"]-far["f1"])
        wave=of(d)
        mct=whiten(scov,np.array([inv(1/(1+z))/(1+z)**2 for z in pantheon["redshifts"]])-pantheon["E_inverse"])
        of=[float(relative(z)) for z in (0.,.4,.8,1.48,2.)]
        return dict(fraction_flux=fraction,sigma0=sigma if fraction else None,omega_m=float(pars[0]),h=float(pars[1]),
                    chi2_fit=float(residual@residual),chi2_DESI_reserve=float(distant@distant),chi2_Panthéon_MCT=float(mct@mct),
                    chi2_Dovekie=wave[0],zéro_mag=wave[1],magnitudes_relatives=wave[2],rho_OF_relative=of)

    rows=[]
    for fraction in (0.,.125,.25,.5,.75,1.):
        for sigma in ((1.35,) if fraction==0. else (1.2,1.35,1.6,1.9)):
            fit=least_squares(read_near,[.3,.68],args=(fraction,sigma),bounds=([.20,.54],[.43,.83]),
                              max_nfev=50,diff_step=3e-5,ftol=2e-8,xtol=2e-8,gtol=2e-8)
            assert fit.success
            r=measures(fit.x,fraction,sigma,2e-11)
            np.testing.assert_allclose(r['chi2_fit'],2*fit.cost,atol=5e-4,rtol=2e-6)
            r["delta_chi2_Dovekie_vs_constante_recalée"]=r["chi2_Dovekie"]-(rows[0]["chi2_Dovekie"] if rows else r["chi2_Dovekie"])
            rows.append(r)
            print(fraction,sigma,[r[k] for k in ('chi2_fit',"chi2_DESI_reserve","chi2_Panthéon_MCT",
                                                "delta_chi2_Dovekie_vs_constante_recalée")],flush=True)

    # Independent differential and early-time controls of the nested model.
    f,d,r=mixture(.31,9e-5,1.35,.4,3e-12)
    f2,d2,r2=mixture(.31,9e-5,1.35,.4,3e-12,199.)
    for z in (.1,.5,1.,2.):
        a_z=1/(1+z)
        np.testing.assert_allclose((d(z+1e-5)-d(z-1e-5))/(2e-5),f(a_z)*a_z*a_z,rtol=1e-9)
        np.testing.assert_allclose([f2(a_z),d2(z),r2(z)],[f(a_z),d(z),r(z)],rtol=1e-6)
    np.testing.assert_allclose(rows[0]["rho_DE_relative"],1.,atol=0.,rtol=0.)
    keys=["DESI_DR2_mean.txt","DESI_DR2_cov.txt","REPERES_R449.json","dovekie/DES-Dovekie_HD.csv","dovekie/STAT+SYS.npz",
          "CID_PANTHEON_PLUS.txt","RESERVE_SUPERNOVAE_R448.json"]
    write(a.output/"CONSTANTE_ET_FLUX_R453.json",dict(revision=453,observations_OF=len(light),duplicates_OF=removed,
          z_density=[0.,.4,.8,1.48,2.],cases=rows,checksums={k:sha(INPUT/k) for k in keys},
          scope=('At zero fraction, the effective density is exactly constant. The fraction is that of the flux in the density '
                  "tardive à z=0. L’interpolation est un banc phénoménologique et la croissance froide lisse est intégrée de manière "
                  'coherent on each fund. m and h are reset on the seven DESI measurements z-1 and .pi. tablet with the same '
                  'terminals for all points. DESI z-1, Panthéon-MCT and 1475 DES-Dovekie modules are read separately, '
                  'without summing these witnesses or selecting a model solely on their difference of .pi. 2. The Conditional Grid '
                  'is neither an overall significance test nor a derivation of substrate disturbances.')))


if __name__=="__main__":
    main()
