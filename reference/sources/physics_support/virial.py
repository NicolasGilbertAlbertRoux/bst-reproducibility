'Freeze predictions of finite density potential R428.'
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import sys

import numpy as np
from scipy.constants import Avogadro, Boltzmann, R, physical_constants
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import roots_legendre


TEMPERATURES = (400.,500.)
PRESSURES_BAR = (1.,11.,21.,31.,41.)
SPECIES = ("Ar","Kr","Xe")


def viriel3(mayer, integrale, n, limit, cutoffs):
    t,w = roots_legendre(n)
    intervals = list(zip(cutoffs, (*cutoffs[1:],limit)))
    x = np.concatenate([.5*((b-a)*t+b+a) for a,b in intervals])
    p = np.concatenate([.5*(b-a)*w for a,b in intervals])*x*mayer(x)
    u,v = x[:,None],x[None,:]
    return -8*np.pi**2/3*float(p @ ((integrale(u+v)-integrale(abs(u-v))) @ p))


def B3(t,c6,sigma,n):
    tau = Boltzmann*t*sigma**6/c6

    def f(x):
        xx = np.maximum(x,1e-5)
        log_x = np.log(xx)
        v = np.exp(-6.*log_x)*np.expm1(np.minimum((6.-n)*log_x,620.))
        return np.expm1(np.clip(-v/tau,-740.,700.))

    values = []
    for nd,lim,n_int in ((56,16.,1024),(84,24.,2048),(110,32.,3072)):
        x = np.linspace(0.,2*lim,int(2*lim*n_int)+1)
        g = CubicSpline(x,cumulative_simpson(x*f(x),x=x,initial=0.))
        b = viriel3(f,g,nd,lim,(0.,.7,.92,1.,1.12,1.35,1.8,2.5,4.,8.))
        values.append(b*(Avogadro*sigma**3)**2)
    return values


def run(a):
    sys.path.insert(0,str(a.r425))
    from gas_mie import virial

    sphere = viriel3(lambda x: -np.ones_like(x),lambda x: -.5*np.minimum(x,1.)**2,
                     200,1.,(0.,))
    np.testing.assert_allclose(sphere,5*np.pi**2/18,rtol=1e-5)
    j = json.loads(a.fit.read_text())
    atlas = json.loads((a.r425/"parents/BST_R420/donnees/REPONSE_GAZ_NOBLES.json").read_text())["rows"]
    c6s = {x["species"]:x["C6_R187"] for x in atlas}
    conversion = physical_constants["Hartree energy"][0]*physical_constants["Bohr radius"][0]**6
    output = []
    for branch in j["raffinements"]:
        for name in SPECIES:
            anchors = [x for x in branch['rows'] if x['gas']==name and x["role"]=="ancrage"]
            p = anchors[0]
            assert all((x["sigma_A"],x["n"])==(p["sigma_A"],p["n"]) for x in anchors)
            n,s,c6 = p["n"],p["sigma_A"]*1e-10,c6s[name]*conversion
            for t in TEMPERATURES:
                b2 = virial(t,c6,s,n)*1e-6
                bs = B3(t,c6,s,n)
                b3 = bs[-1]
                deviation = abs(b3-bs[-2])
                assert deviation<max(abs(b3)*2e-4,1e-13)
                d = dict(gas=name,T=t,p_size=branch["p"],n=n,sigma_A=p["sigma_A"],C6_R187=c6s[name],
                         B2_m3_mol=b2,B3_m6_mol2=b3,error_quadrature_B3_m6_mol2=deviation,
                         convergence_B3_m6_mol2=bs,points=[])
                for bar in PRESSURES_BAR:
                    rho_ideal = 1e5*bar/(R*t)
                    def model(rho):
                        return rho*(1.+b2*rho+b3*rho**2)
                    rho = brentq(lambda z:model(z)-rho_ideal,.75*rho_ideal,1.35*rho_ideal)
                    d["points"].append(dict(pressure_bar=bar,rho_ideale_mol_m3=rho_ideal,
                                            rho_B2_B3_mol_m3=rho))
                output.append(d)
    with a.fit.open("rb") as f:
        sha = file_digest(f,"sha256").hexdigest()
    d = dict(scope='Conventional gas with pair potential R428. Truncated pressure cell after B3.',
             selection="Deux paramètres de taille R428, Ar/Kr/Xe, 400 et 500 K, 1 à 41 bar. "
                       'Parameters and fixed points before reading the comparison isotherms.',
             source_fit_sha256=sha,sphere_hard_B3_reduced=sphere,predictions=output)
    a.output.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    with a.output.open("rb") as f:
        print("Gel",len(output)*len(PRESSURES_BAR),"points",file_digest(f,"sha256").hexdigest())


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("r425",'fit','output'):
        p.add_argument("--"+n,type=Path,required=True)
    run(p.parse_args())
