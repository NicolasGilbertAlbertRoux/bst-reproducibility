'Test a common multiplier of the short term EATM with the published C9.'
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import sys

import numpy as np
from scipy.optimize import brentq

CUTS = (.60,.72,.85,.96,1.15,1.35,1.8,2.5,4.,8.)
ORDERS = ((20,16.),(28,24.),(36,32.))
MULTIPLIER = (0.,4.)
CONTROLS_PVT_KR = ((298.15,-50.41,.9,2241.,340.),(348.15,-33.58,.4,1753.,230.))


def shifted_coefficients(args):
    gas, temp, reference, b_pair, factor, coeff, order, limit = args
    from virial_virials import nodes

    length, energy, a0, a1, a2, a3, a4, a5, bohr_scale, h_scale, c9, alpha, _, _, *a = coeff

    def pair(r):
        x = r/length
        return energy*(a0*np.exp(a1*x)/x+a2*np.exp(a3*x)+a4)/(1.+a5*x**6)

    radii, r_weights = nodes(order,(*CUTS,limit))
    upsilon, upsilon_weights = nodes(order,(0.,.25,.55,1.))
    weights, long, short = [],[],[]
    for r,wr in zip(radii,r_weights,strict=True):
        radial, sw = nodes(order,[r,*[stop for stop in CUTS if stop>r],limit])
        s = radial[:,None]
        u = s+r*upsilon
        cos1 = (r*r+s*s-u*u)/(2.*r*s)
        cos2 = (r*r+u*u-s*s)/(2.*r*u)
        cos3 = (s*s+u*u-r*r)/(2.*s*u)
        rg = np.cbrt(r*s*u)*bohr_scale
        angle = 1.+3.*cos1*cos2*cos3
        exponent = -pair(r)-pair(s)-pair(u)
        w = np.exp(np.maximum(exponent,-745.))*wr*sw[:,None]*upsilon_weights*r*r*s*u
        weights.append(w.ravel())
        long.append((h_scale*angle*c9/rg**9).ravel())
        short.append((h_scale*angle*np.polynomial.polynomial.polyval(rg**2,a)*
                      np.exp(-alpha*(r+s+u)*bohr_scale)).ravel())
    w, v0, v1 = [np.concatenate(x) for x in (weights,long,short)]
    # Some regions at very short range have exactly zero numerical
    # Boltzmann weight. Enforce the representable exponent before integrating.
    w = w/factor
    keep = w>0.
    w, v0, v1 = w[keep],v0[keep],v1[keep]
    assert np.max(abs(v0))+MULTIPLIER[-1]*np.max(abs(v1)) < 640.
    pref = 16.*np.pi**2

    def value(x, kind):
        extra = v0+x*v1 if kind=="court" else x*(v0+v1)
        return b_pair - pref*np.sum(w*np.expm1(-extra))

    def slope(x, kind):
        base = v1 if kind=="court" else v0+v1
        exp = -v0-x*v1 if kind=="court" else -x*base
        return pref*np.sum(w*base*np.exp(exp))

    def intervals(target, tolerance, kind):
        lower,upper = MULTIPLIER
        sizes = [lower,upper]
        d1,d2 = slope(lower,kind),slope(upper,kind)
        # B3 is concave in a scalar coefficient since weights are positive.
        if d1>0 and d2<0:
            sizes.insert(1,brentq(lambda x:slope(x,kind),lower,upper))
        roots = set(sizes)
        bounds = [target*(1.-tolerance),target*(1.+tolerance)]
        for a,b in zip(sizes[:-1],sizes[1:],strict=True):
            for target_value in bounds:
                if (value(a,kind)-target_value)*(value(b,kind)-target_value) < 0:
                    roots.add(brentq(lambda x:value(x,kind)-target_value,a,b))
        sections = sorted(roots)
        return [[float(a),float(b)] for a,b in zip(sections[:-1],sections[1:],strict=True)
                if bounds[0]<=value((a+b)/2.,kind)<=bounds[1]]

    return dict(gas_field=gas, T_K=temp, order=order, cutoff=limit,
                B3_ref=reference, B3_pair=b_pair, B3_EATM=float(value(1.,"court")),
                B3_ATM=float(value(0.,"court")), B3_court4=float(value(4.,"court")),
                slope_court_aux_bounds=[float(slope(x,"court")) for x in MULTIPLIER],
                tolerance={str(u):{k:intervals(reference,u,k) for k in ("court","global")} for u in (.05,.1)})


def pair_from_scratch(temp, basis, p, order, limit):
    from scipy.constants import Avogadro, Boltzmann
    from scipy.integrate import quad, cumulative_simpson
    from scipy.interpolate import CubicSpline
    from virial import viriel3

    length, energy, a0,a1,a2,a3,a4,a5 = p[:8]
    scale = energy*basis["energie_J"]/(Boltzmann*temp)
    def mayer(r):
        x = np.maximum(r/length,1e-14)
        return np.expm1(-scale*(a0*np.exp(a1*x)/x+a2*np.exp(a3*x)+a4)/(1.+a5*x**6))

    f = Avogadro*basis["sigma_m"]**3
    b2 = -2.*np.pi*f*quad(lambda x:x*x*mayer(x),0.,np.inf,epsabs=1e-9)[0]
    x = np.linspace(0.,2.*limit,round(2400.*limit)+1)
    primitive = CubicSpline(x,cumulative_simpson(x*mayer(x),x=x,initial=0.))
    b3 = f**2*viriel3(mayer,primitive,2*order,limit,(0.,*CUTS))
    return b2,b3


def intersect(given):
    candidates = [list(MULTIPLIER)]
    for intervals in given:
        candidates = [[max(a,c),min(b,d)] for a,b in candidates for c,d in intervals
                      if max(a,c) <= min(b,d)]
    return candidates


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--parent",type=Path,required=True,help="Archive R445 extraite")
    cli.add_argument("--r444",type=Path,required=True,help='BST-R444 directory of the R444 archive')
    cli.add_argument("--sortie",type=Path,required=True)
    a = cli.parse_args()
    root = a.parent.resolve()
    sys.path[:0] = [str(root / r) for r in ("BST_R432","BST_R435","BST_R442")]
    from hashlib import file_digest
    from scipy.constants import Avogadro, Boltzmann
    from virial_virials import BOHR, HARTREE

    p445 = root/"BST_R445/resultats/VIRIELS_QUEUES_R445.json"
    table = json.loads(p445.read_text())['cases']
    models = json.loads((root/"BST_R442/entrees/POTENTIELS_ET_ETATS_R442.json").read_text())["configs"]
    coeffs = {g:np.loadtxt(a.r444/'inputs'/f"POTENTIEL_{g}_R444.dat") for g in ("Kr","Xe")}
    calls = []
    for row in table:
        g,T = row['gas'],row["T_K"]
        for q,(order,limit) in zip(row["quadratures"],ORDERS,strict=True):
            # Convert reduced pair and trimer energy to kBT and use the
            # absolute molar virial factor of this same gas.
            p = coeffs[g].copy()
            model = models[g]
            p[[1,9]] *= model["energie_J"]/(Boltzmann*T)
            np.testing.assert_allclose(p[8:10],
                                       [model["sigma_m"]/BOHR,HARTREE/(Boltzmann*T)],rtol=4e-10)
            calls.append((g,T,row["B3_ref"],q['B3_pair'],1./(Avogadro*model["sigma_m"]**3)**2,
                          p,order,limit))
    for T,_,_,ref,_ in CONTROLS_PVT_KR:
        for order,limit in ORDERS[-2:]:
            model = models["Kr"]
            b2,b3 = pair_from_scratch(T,model,coeffs["Kr"],order,limit)
            p = coeffs["Kr"].copy()
            p[[1,9]] *= model["energie_J"]/(Boltzmann*T)
            calls.append(("Kr",T,ref*1e-12,b3,1./(Avogadro*model["sigma_m"]**3)**2,p,order,limit))

    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(shifted_coefficients,calls,chunksize=1))
    final = [r for r in rows if r['order']==ORDERS[-1][0]]
    for old,new in zip(table,final[:len(table)],strict=True):
        np.testing.assert_allclose(old["quadratures"][-1]["B3_total"],new["B3_EATM"],rtol=2e-10)
        lo,hi = old["quadratures"][-1]["multiplicateurs_EATM"][1]
        lo_new,hi_new = new["tolerance"]["0.1"]["global"][0]
        np.testing.assert_allclose([lo_new,min(hi_new,2.)],[lo,hi],atol=2e-8,rtol=0.)
    results = {}
    for tolerance in ("0.05","0.1"):
        results[tolerance] = {name:intersect(r["tolerance"][tolerance][name] for r in final[:len(table)])
                              for name in ("court","global")}
    pvt = []
    for setup in CONTROLS_PVT_KR:
        t,ref_b2,u_b2,ref_b3,u_b3 = setup
        b2,_ = pair_from_scratch(t,models["Kr"],coeffs["Kr"],*ORDERS[-1])
        r = next(row for row in final if row["T_K"]==t)
        pvt.append(dict(T_K=t,B2_calc_cm3_mol=b2*1e6,B2_ref=ref_b2,B2_marge=u_b2,
                        B3_calc_cm6_mol2=r["B3_EATM"]*1e12,B3_ref=ref_b3,B3_marge=u_b3,
                        B3_in_band=bool(abs(r["B3_EATM"]*1e12-ref_b3)<=u_b3),
                        B2_in_band=bool(abs(b2*1e6-ref_b2)<=u_b2)))
    with p445.open("rb") as stream:
        digest = file_digest(stream,"sha256").hexdigest()
    a.output.mkdir(parents=True,exist_ok=True)
    result = dict(revision=446,data_parent_sha256=digest,multiplier=MULTIPLIER,intersections=results,
                  point_reference='CoolProp correlations archived in R445',cases=rows,
                  pVT_Kr=dict(source="https://doi.org/10.1063/1.4943959",cases=pvt),
                  scope='Frozen SAAP pair and B3 archived pair pair. The C9 Kr and Xe are frozen in the short variant. '
                         'The intervals are quadrature calculations on a declared box. '
                         'The pVT control of two insulated Kr uses the published margin, without adjustment.')
    (a.output/"VIRIELS_COURT_ET_LONG_R446.json").write_text(
        json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf-8")
    print(results)
    print(pvt,flush=True)


if __name__ == "__main__":
    main()
