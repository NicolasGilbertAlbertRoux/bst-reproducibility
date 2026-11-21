'Add an independently fixed triple-dipole contribution to the R432 gas test.'

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.constants import Avogadro, Boltzmann, R, physical_constants
from scipy.optimize import brentq
from scipy.special import gammainc, roots_legendre


# Tang et al., Table V. Its angular factor has no extra factor of three.
Z111 = {'Ar': 518.3, 'Kr': 1572., 'Xe': 5573.}
BOHR = physical_constants['Bohr radius'][0]
HARTREE = physical_constants['Hartree energy'][0]


def nodes(order, cuts):
    x, w = roots_legendre(order)
    left = np.array(cuts[:-1])[:,None]
    right = np.array(cuts[1:])[:,None]
    return (.5*((right-left)*x+right+left)).ravel(), (.5*(right-left)*w).ravel()


def integral(row, z111, damping, order, limit):
    # Order the sides a<=b<=c and integrate the six equivalent sectors.
    # Use c=b+t*a with 0<=t<=1 to impose the triangle inequalities exactly.
    grid = (0., .72, .9, 1., 1.15, 1.35, 1.8, 2.5, 4., 8., limit)
    a, wa = nodes(order, grid)
    # Split the b quadrature at a rather than integrating across a diagonal step.
    t, wt = nodes(order, (0., .25, .55, 1.))
    sigma = row['sigma_A']*1e-10
    temp = row['T']
    n = row['n']
    pair_scale = row['C6_R187']*HARTREE*(BOHR/sigma)**6/(Boltzmann*temp)
    triple_scale = z111*HARTREE*(BOHR/sigma)**9/(Boltzmann*temp)

    def pair(r):
        v = np.log(np.maximum(r, 1e-8))
        return pair_scale*np.exp(-6*v)*np.expm1(np.minimum((6.-n)*v, 615.))

    def short_range(r):
        return gammainc(7., damping*r)

    total = 0.
    for r, wr in zip(a,wa,strict=True):
        left = [r, *(v for v in grid[1:] if v>r)]
        s, ws = nodes(order, left)
        s = s[:,None]
        u = s+r*t
        c1 = (r*r+s*s-u*u)/(2*r*s)
        c2 = (r*r+u*u-s*s)/(2*r*u)
        c3 = (s*s+u*u-r*r)/(2*s*u)
        atm = triple_scale*(1.+3*c1*c2*c3)*short_range(r)*short_range(s)*short_range(u)/(r*s*u)**3
        boltzmann = np.exp(np.clip(-pair(r)-pair(s)-pair(u), -745., 690.))
        integrand = r*r*s*u*boltzmann*np.expm1(-atm)
        total += wr*float(ws@(integrand@wt))
    return -16*np.pi**2*(Avogadro*sigma**3)**2*total


def density(temp, pressure, b2, b3):
    ideal = 1e5*pressure/(R*temp)
    return brentq(lambda x:x*(1.+x*(b2+x*b3))-ideal,.75*ideal,1.35*ideal)


def run(args):
    frozen = json.loads((args.r432/'resultats/gaz/GEL_GAZ_R432.json').read_text())
    ref = json.loads((args.r432/'resultats/gaz/CONFRONTATION_GAZ_R432.json').read_text())
    measured = {(x['gas'],x['T'],x['p_size']):x for x in ref['comparisons']}
    out = []
    # The size exponent and both temperatures were fixed in R432. The three
    # dimensionless damping scales are sensitivity tests with no fit to B3.
    p = frozen['predictions'][-1]['p_size']
    for row in frozen['predictions']:
        if row['p_size'] != p:
            continue
        gas,temp = row['gas'],row['T']
        target = measured[gas,temp,p]
        b2,b3 = (row[k] for k in ('B2_m3_mol','B3_m6_mol2'))
        np.testing.assert_allclose(integral(row,0.,8.,8,12.),0.,atol=1e-30)
        np.testing.assert_allclose([density(temp,x['pressure_bar'],b2,b3) for x in row['points']],
                                   [x['rho_B2_B3_mol_m3'] for x in row['points']],rtol=1e-13)
        variants = []
        for damp in (6.,8.,10.):
            values = [integral(row,Z111[gas],damp,o,lim) for o,lim in ((14,16.),(21,24.),(26,32.))]
            e = abs(values[-1]-values[-2])
            assert e<max(1e-14,abs(values[-1])*.01),(gas,temp,damp,values)
            b3_corrected = b3+values[-1]
            pts = []
            for x in target['points']:
                rho = density(temp,x['pressure_bar'],b2,b3_corrected)
                pts.append(dict(pressure_bar=x['pressure_bar'],rho_corrected=rho,
                                error_corrected=rho/x['rho_ref_mol_m3']-1.,
                                error_R432=x['deviation_density'],error_gas_ideal=x['deviation_density_ideale']))
            variants.append(dict(beta_sigma=damp,delta_B3_m6_mol2=values[-1],quadrature_m6_mol2=values,
                                 B3_cm6_mol2=b3_corrected*1e12,
                                 error_B3=b3_corrected*1e12/target['B3_ref_cm6']-1.,points=pts))
        out.append(dict(gas_field=gas,T_K=temp,source_C9_atoms_u=Z111[gas],B2_R432_cm3_mol=b2*1e6,
                        B3_R432_cm6_mol2=b3*1e12,B3_EOS_cm6_mol2=target['B3_ref_cm6'],variantes=variants))
        print(gas,temp,'B3',round(b3*1e12,2),'corrigé',round(variants[1]['B3_cm6_mol2'],2),
              'EOS',round(target['B3_ref_cm6'],2),flush=True)
    doc = dict(revision=435,scope='Classical viriel up to B3 and imported triple dipole ATM, damped by '
               'un produit de trois Tang–Toennies d’ordre 6. Sensibilité βσ=6,8,10 sans ajustement au viriel. '
               'Two EOS temperatures and references already examined in R432. The three-body term does not change B2.',
               sources={'C9':'https://doi.org/10.1063/1.3691891',
                        'amortissement_court_illustration':'https://www.sciencedirect.com/science/article/abs/pii/S0009261499014426'},
               p_size=p,cases=out)
    args.output.write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--r432',type=Path,required=True)
    parser.add_argument('--sortie',type=Path,required=True)
    run(parser.parse_args())
