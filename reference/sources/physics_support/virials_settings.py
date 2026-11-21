'Calibrate a common three-body correction by holding the pair potential.'
from argparse import ArgumentParser
from functools import lru_cache
from itertools import product
from pathlib import Path
import sys

import numpy as np
from scipy.constants import Avogadro, Boltzmann
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parent
INPUT=ROOT/'inputs'
VIRIAL=INPUT/"viriels"
sys.path[:0]=[str(INPUT),str(VIRIAL)]
from cosmology_R449 import read,sha,write
from virial_virials import nodes


TEMPERATURES=np.array([300.,350.,400.,450.,500.])
SITES=(1.04,1.34)
WIDTHS=(.23,.41)
AMPLITUDE_FRACTION=.09


def grid(species,params,c9,order,limit,short_extent):
    data=np.loadtxt(VIRIAL/f"POTENTIEL_{species}_R444.dat")
    length,energy,a0,a1,a2,a3,a4,a5,bor,hartree_scale,_,alpha,_,_,*poly=data
    pot=lambda x: (energy*params["energie_J"]/Boltzmann)*(a0*np.exp(a1*x)/x+a2*np.exp(a3*x)+a4)/(1.+a5*x**6)
    theta_r=hartree_scale*params["energie_J"]/Boltzmann
    cuts=(.60,.72,.85,.96,1.15,1.35,1.8,2.5,4.,8.,limit)
    first,weights=nodes(order,cuts)
    last,last_w=nodes(order,(0.,.25,.55,1.))
    pref=16*np.pi**2*(Avogadro*params["sigma_m"]**3)**2*1e12
    coords=[]
    tails=np.zeros(len(TEMPERATURES))
    for r,wr in zip(first,weights,strict=True):
        second,second_w=nodes(order,[r,*[k for k in cuts if k>r]])
        s=second[:,None]
        u=s+r*last
        cos1=(r*r+s*s-u*u)/(2*r*s)
        cos2=(r*r+u*u-s*s)/(2*r*u)
        cos3=(s*s+u*u-r*r)/(2*s*u)
        rg=np.cbrt(r*s*u)
        angle=1.+3*cos1*cos2*cos3
        sum_pair=pot(r/length)+pot(s/length)+pot(u/length)
        weights=pref*wr*r*r*s*u*second_w[:,None]*last_w
        e3=theta_r*angle*(c9/(rg*bor)**9+np.polynomial.polynomial.polyval((rg*bor)**2,poly)*
                          np.exp(-alpha*(r+s+u)*bor))
        active=(u<short_extent)&(sum_pair<7000.)
        inactive=~active
        for i,temp in enumerate(TEMPERATURES):
            tails[i]-=np.sum(weights[inactive]*np.exp(np.maximum(-sum_pair[inactive]/temp,-745.))*
                             np.expm1(np.clip(-e3[inactive]/temp,-745.,680.)))
        average=(r+s+u)/3
        edge=np.prod([(1-np.minimum(np.broadcast_to(x,u.shape)/short_extent,1.)**18)**3
                      for x in (r,s,u)],axis=0)
        response=[(params["energie_J"]/Boltzmann)*AMPLITUDE_FRACTION*edge*np.exp(-((average-p)/w)**2)
                  for p,w in zip(SITES,WIDTHS,strict=True)]
        coords.append(np.column_stack([z[active] for z in (weights,sum_pair,e3,*response)]))
    grid_pts=np.concatenate(coords)
    w,ep,e3,*modes=grid_pts.T
    scale=np.log(params["sigma_m"]/3.63e-10)/.05
    basis=np.vstack((modes[0],modes[1],scale*modes[0],scale*modes[1])).T
    return tails, w, ep,e3,basis


def prediction(model,pars):
    tails,w,e2,e3,terms=model
    delta=terms@pars
    time=TEMPERATURES[:,None]
    weight=w*np.exp(-e2/time)
    exp=np.exp(np.clip(-(e3+delta)/time,-745,500.))
    return tails+(weight*(1.-exp)).sum(axis=1),(weight*exp/time)@terms


def prepare(data,parameters,long,yield_order,limit,extent):
    gases=list(parameters)
    models={g:grid(g,parameters[g],long[g]["C9"],yield_order,limit,extent) for g in gases}
    totals={gas:[x for x in data if x['gas']==gas] for gas in gases}
    paired={gas:np.array([x['B3_pair']*1e12 for x in total]) for gas,total in totals.items()}
    observed={gas:np.array([x["B3_ref"]*1e12 for x in total]) for gas,total in totals.items()}
    return models,paired,observed


def run(prepared,pars):
    data=[]
    models,pairs,refs=prepared
    for g,model in models.items():
        signs,der=prediction(model,pars)
        final=pairs[g]+signs
        data.append((g,final,final/refs[g]-1.,der/refs[g][:,None]))
    return data


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--sortie",type=Path,default=ROOT/'results')
    cfg=parser.parse_args()
    out=cfg.output
    dpath=out/"DISPERSION_R453.json"
    vpath=out/"VIRIELS_R453.json"
    geometries=read(VIRIAL/"POTENTIELS_ET_ETATS_R442.json")["configs"]
    sizes={g:geometries[g] for g in ("Kr","Xe")}
    dispersed=read(dpath)["espèces"]
    virial=read(vpath)['cases']
    # Alternate low and high temperatures. Reserve 350 and 450 K for both gases.
    train=[0,2,4]
    hold=[1,3]
    solutions=[]
    half_box=.0003
    for cutoff in (2.15,2.45):
        setup=prepare(virial,sizes,dispersed,16,16.,cutoff)
        baseline=run(setup,np.zeros(4))
        np.testing.assert_allclose([d[1] for d in baseline],[[x["B3_EATM"]*1e12 for x in virial if x['gas']==g]
                                   for g in sizes],rtol=6e-4)
        @lru_cache(maxsize=5)
        def evaluate(*pars):
            stage=run(setup,np.array(pars))
            return np.concatenate([x[2][train] for x in stage]),np.concatenate([x[3][train] for x in stage],axis=0)

        def constraint(x):
            res,_=evaluate(*x[:-1])
            return np.r_[x[-1]-res,x[-1]+res]

        def gradient(x):
            _,jac=evaluate(*x[:-1])
            return np.column_stack((np.r_[-jac,jac],np.ones(12)))

        setup_fine=prepare(virial,sizes,dispersed,24,24.,cutoff)
        for window in (1.5,2.25,3.):
            limits=[(-window,window)]*4+[(0.,.5)]
            optimum=minimize(lambda x:x[-1]+.0003*(x[:-1]@x[:-1]),np.r_[np.zeros(4),.3],
                             jac=lambda x:np.r_[.0006*x[:-1],1.],method="SLSQP",bounds=limits,
                             constraints=[dict(type="ineq",fun=constraint,jac=gradient)],
                             options=dict(ftol=2e-10,maxiter=50))
            assert optimum.success, optimum.message
            pars=optimum.x[:-1]
            verify=run(setup_fine,pars)
            test=run(setup,pars)
            errors=[float(np.max(abs(fine[1]-coarse[1]))) for fine,coarse in zip(verify,test,strict=True)]
            assert max(errors)<4.
            controlled=[]
            for gas,values,res,_ in verify:
                for i,(t,value,relative) in enumerate(zip(TEMPERATURES,values,res,strict=True)):
                    controlled.append(dict(gas_field=gas,temperature_K=t,B3_cm6_mol2=float(value),deviation_relative=float(relative),
                                           reserved=i in hold))
            checked=[]
            for corner in product((-half_box,half_box),repeat=4):
                errs=np.concatenate([result[2] for result in run(setup_fine,pars+corner)])
                checked.append(float(np.max(abs(errs))))
            # For each isotherm B3 is increasing in its effective amplitudes.
            # Componentwise extremes after projecting the common four-cube to
            # each gas certify the contained joint box even between corners.
            certificate=[]
            models,pair_base,refs=setup_fine
            for gas,model in models.items():
                scale=np.log(sizes[gas]["sigma_m"]/3.63e-10)/.05
                pm=pars[:2]+scale*pars[2:]
                width=half_box*(1+abs(scale))
                margin=errors[list(models).index(gas)]/refs[gas]
                lo,hi=[(pair_base[gas]+prediction(model,np.r_[pm+sign*width,0.,0.])[0])/refs[gas]-1.+sign*margin
                       for sign in (-1.,1.)]
                certificate.append(dict(gas_field=gas,deviations=np.column_stack((lo,hi)).tolist()))
            sol=dict(cutoff_reduced=cutoff,half_domaine_coefficients=window,parameters_field=pars.tolist(),
                     max_fit=max(abs(r['écart_relative']) for r in controlled if not r['reserved']),
                     max_reserve=max(abs(r['écart_relative']) for r in controlled if r['reserved']),cases=controlled,
                     error_quadrature_cm6_mol2=errors,max_16_vertices=max(checked),
                     box_around_solution=np.column_stack((pars-half_box,pars+half_box)).tolist(),
                     bound_par_monotonicity=certificate)
            print(cutoff, window, pars, sol['max_fit'],sol["max_reserve"],flush=True)
            solutions.append(sol)
    write(out/"VIRIELS_REGLAGE_R453.json",dict(revision=453,trials=solutions,
          checksums={p.name:sha(p) for p in (dpath,vpath)},
          scope=('Symmetrical triatomic correction, reduced by native size with two common radial profiles and '
                  'one log slope per profile. A triatomic term leaves B2 and the isolated diffusion of two bodies '
                  'identical. The long-range limit retains the optical C9. The six scores at 300, 400 and 500 K are used for the '
                  'and 350 and 450 K at retrospective control. The references remain correlations. The forces of '
                  'finite density transport and the character derived from the correction remain to be tested.')))


if __name__=="__main__":
    main()
