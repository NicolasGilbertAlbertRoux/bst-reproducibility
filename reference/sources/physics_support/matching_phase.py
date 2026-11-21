'Connecting BST Virons to Argonal Energy and experience co-existence.'
from argparse import ArgumentParser
from functools import lru_cache
from hashlib import file_digest
from pathlib import Path
import sys

import CoolProp as CP
from CoolProp.CoolProp import AbstractState, PropsSI
import numpy as np
from scipy.constants import R
from scipy.optimize import brentq, root

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
DATA = ROOT / 'inputs'
sys.path.insert(0,str(ROOT.parent/'BST_R438'))

from criticality import CRITICAL
from gas_joint import gas_input, phase_reference, read, write, PRESSURES

FROZEN = OUT/'RACCORD_PHASE_GELE_R441.json'
AR = AbstractState('HEOS','Argon')
AR.specify_phase(CP.iphase_gas)
AR_T, AR_D, _ = CRITICAL['Ar']
VIRIAL_WIDTH = .06
PROPERTIES = ('pressure','density_vapeur','density_liquide','chaleur_latente')


def raw(t,y):
    AR.update(CP.DmolarT_INPUTS, y*AR_D, t*AR_T)
    return np.array([AR.alphar(),AR.dalphar_dDelta(),AR.d2alphar_dDelta2()])


@lru_cache(maxsize=512)
def endpoints(t):
    t_si = t*AR_T
    g,l = (PropsSI('Dmolar','T',t_si,'Q',q,'Argon')/AR_D for q in (1.,0.))
    a,ay,ayy = raw(t,g)
    p = g*(1.+g*ay)
    mu = np.log(g)+a+g*ay
    ag = (1.+2.*g*ay+g*g*ayy)/(2.*g*(g-l)**2)
    a,ay,ayy = raw(t,l)
    al = (1.+2.*l*ay+l*l*ayy)/(2.*l*(g-l)**2)
    np.testing.assert_allclose([l*(1.+l*ay),np.log(l)+a+l*ay], [p,mu], atol=5e-8,rtol=0.)
    assert min(ag,al) > 0.
    return g,l,p,mu,ag,al


def argon(t,y):
    # A Hermite polynomial joins the two reference branches inside the
    # argon binodal. It preserves f, f' and f'' at both ends and keeps
    # the free energy above the common tangent within that interval.
    if t >= 1.:
        return raw(t,y)
    g,l,p,mu,kg,kl = endpoints(t)
    if y <= g or y >= l:
        return raw(t,y)
    q = (y-g)*(y-l)
    dq = 2.*y-g-l
    d = (kl-kg)/(l-g)
    k = kg+d*(y-g)
    f = mu*y-p + k*q*q
    df = mu+d*q*q + 2.*k*q*dq
    ddf = 4.*d*q*dq + 2.*k*(dq*dq+2.*q)
    a = f/y
    ap = (df-a)/y
    app = (ddf-2.*ap)/y
    return np.array([a-np.log(y)+1., ap-1./y,app+1./y**2])


@lru_cache(maxsize=512)
def argon_virials(t):
    AR.update(CP.DmolarT_INPUTS, .01*AR_D, t*AR_T)
    return AR.Bvirial()*AR_D,AR.Cvirial()*AR_D**2


def values(gas,jet,y):
    T,B,C,_ = jet
    tc,rc,_ = CRITICAL[gas]
    t = T/tc
    b,c = argon_virials(t)
    db,dc = B*rc-b, C*rc**2-c
    p = db*y+.5*dc*y*y
    dp = db+dc*y
    exponent = -(y/VIRIAL_WIDTH)**4
    w = np.exp(exponent)
    ew = 4.*exponent/y
    eww = 12.*exponent/(y*y)
    a = argon(t,y)
    return a + w*np.array([p,dp+ew*p,dc+2.*ew*dp+(ew*ew+eww)*p])


def pressure_mu_slope(gas,jet,y):
    a,ap,app = values(gas,jet,y)
    return y*(1.+y*ap),np.log(y)+a+y*ap,1.+2.*y*ap+y*y*app


def saturation(gas,jet):
    tc,rc,_ = CRITICAL[gas]
    t = jet[0]/tc
    g,l,*_ = endpoints(t)

    def equality(z):
        v,w = np.exp(z)
        p1,m1,_ = pressure_mu_slope(gas,jet,v)
        p2,m2,_ = pressure_mu_slope(gas,jet,w)
        return np.array([p1-p2,m1-m2])
    sol = root(equality, np.log([g,l]), tol=5e-10)
    vg,vl = np.exp(sol.x)
    pd,mu,dp_g = pressure_mu_slope(gas,jet,vg)
    _,_,dp_l = pressure_mu_slope(gas,jet,vl)
    assert np.max(abs(equality(sol.x)))<1e-9 and min(dp_g,dp_l)>0. and 0.<vg<vl
    assert vl-vg>.1
    # Check each stationary value of the grand potential in the tested
    # density interval, in addition to the two coexistence endpoints.
    y_grid = np.unique(np.r_[np.geomspace(1e-8,.1,120),np.linspace(.1,2.8,801),vg,vl])
    gaps = [pressure_mu_slope(gas,jet,y)[1]-mu for y in y_grid]
    roots = [brentq(lambda y:pressure_mu_slope(gas,jet,y)[1]-mu,a,b,xtol=1e-13)
             for a,b,ma,mb in zip(y_grid[:-1],y_grid[1:],gaps[:-1],gaps[1:],strict=True) if ma*mb<0.]
    def grand(y):
        return y*(np.log(y)-1.+values(gas,jet,y)[0]-mu)+pd
    minim = min(grand(y) for y in [y_grid[0],y_grid[-1],vg,vl,*roots])
    assert minim > -1e-8, (gas,t,minim)
    return np.array([pd*R*jet[0]*rc,vg*rc,vl*rc]),dict(
        error_coexistence=float(np.max(abs(equality(sol.x)))), slopes=[dp_g,dp_l],
        grand_potential_min=minim, interval_density_reduced=[float(y_grid[0]),float(y_grid[-1])])


def phase(row):
    gas = row['gas']
    tc,rc,_ = CRITICAL[gas]
    jets = row['jet']
    state, controls = saturation(gas,jets[2])
    p,g,l = state
    t = jets[2][0]
    h = jets[3][0]-t
    differences = np.array([values(gas,j,g/rc)[0]-values(gas,j,l/rc)[0] for j in jets])
    derivative = (differences[0]-8.*differences[1]+8.*differences[3]-differences[4])/(12.*h)
    dv = 1./g-1./l
    latent = p*dv-R*t*t*derivative
    # Solve coexistence at a finer spacing for the independent Clapeyron control.
    pressures = [saturation(gas,jets[i])[0][0] for i in (1,3)]
    clap = t*dv*(pressures[1]-pressures[0])/(2.*h)
    assert latent > 0.
    return dict(gas_field=gas,T_K=t,tau=t/tc,computed=np.r_[state,latent].tolist(),
                error_clapeyron=float(clap/latent-1.), checks=controls)


def gas(row, pressures=PRESSURES):
    species, T, jets = row['gas'],row['T_K'],row['jet']
    rc = CRITICAL[species][1]
    mass = row['mass']
    h = jets[3][0]-T
    result = []
    for bar in pressures:
        p = bar*1e5
        y_ideal = p/(R*T*rc)
        y = brentq(lambda d:pressure_mu_slope(species,jets[2],d)[0]-y_ideal,.50*y_ideal,2.20*y_ideal)
        e = np.array([values(species,j,y) for j in jets])
        dA = (e[0]-8.*e[1]+8.*e[3]-e[4])/(12.*h)
        d2A = (-e[0]+16.*e[1]-30.*e[2]+16.*e[3]-e[4])/(12.*h*h)
        dp = pressure_mu_slope(species,jets[2],y)[2]*R*T
        dt = R*rc*y*(1.+y*(e[2,1]+T*dA[1]))
        cv = 1.5*R-R*T*(2.*dA[0]+T*d2A[0])
        cp = cv+T*dt*dt/((rc*y)**2*dp)
        assert min(dp,cv,cp)>0.
        result.append([rc*y,cv,cp,1./(rc*y*dp),np.sqrt(cp*dp/(cv*mass))])
    return result


def virial_check(rows):
    checks = []
    for row in rows:
        gas = row['gas']
        rc = CRITICAL[gas][1]
        T,B,C,_ = row['jet'][2]
        e = []
        for y in (1e-4,5e-5):
            a,da,dda = values(gas,row['jet'][2],y)
            e.append(np.array([da-dda*y,dda]))
        est = 2.*e[-1]-e[0]
        ref = [B*rc,C*rc**2]
        checks.append(dict(gas_field=gas,T_K=T,reference=ref,derivative_field=est.tolist(),
                           error_absolute_max=float(np.max(abs(est-ref)))))
    return checks


def freeze():
    initial = read(ROOT.parent/'BST_R438/entrees/COEFFICIENTS_PHASE_R438.json')['cases']
    reserve = read(DATA/'COEFFICIENTS_RESERVE_PHASE_R441.json')['cases']
    phase_rows = [dict(**phase(r),role=tag) for tag,rows in [('historique',initial),('reserve',reserve)] for r in rows]
    gas_rows = [dict(gas_field=r['gas'],T_K=r['T_K'],values_field=gas(r),reference=r['reference'].tolist()) for r in gas_input()]
    transition = read(DATA/'COEFFICIENTS_RACCORD_GAZ_R441.json')['cases']
    from thermo_R437 import FLUIDS
    pressures = (1.,3.,7.,11.,21.,31.,41.)
    transition_rows = []
    for row in transition:
        item = dict(row,mass_field=PropsSI('molar_mass',FLUIDS[row['gas']]))
        try:
            calculated = gas(item,pressures)
            reason = None
        except (AssertionError,ValueError) as exc:
            calculated = None
            reason = str(exc) or 'Defficient mechanical or thermal stability'
        transition_rows.append(dict(gas_field=row['gas'],T_K=row['T_K'],tau=row['tau'],
                                    values_field=calculated,failure=reason))
    checks = virial_check(reserve+transition)
    assert max(r['error_absolute_max'] for r in checks) < .0001
    write(FROZEN,dict(revision=441, model='Reference helmholtz Ar at reduced temperature and density, connector to R435',
         width_density_reduced=VIRIAL_WIDTH,
         source_argon='https://doi.org/10.1063/1.556037',
         standard='Référence Ar, constantes critiques des trois gaz. Aucun calage sur les propriétés de saturation Kr/Xe.',
         choice_correspondence='Correction of attenuated viral values around 0.06 density reduced to any temperature. '
                       'The intermediate zone is tested under pressure and temperature in a reserved bench.',
         order_observables=PROPERTIES,phase=phase_rows,gas_field=gas_rows,check_viriels=checks,
         gas_transition=transition_rows,pressures_transition_bar=list(pressures)))
    print('frozen phase connection',len(phase_rows),len(gas_rows),flush=True)


def compare():
    doc = read(FROZEN)
    with FROZEN.open('rb') as stream:
        doc['sha256_predictions'] = file_digest(stream,'sha256').hexdigest()
    for r in doc['phase']:
        ref = phase_reference(r['gas'],r['T_K'])
        r['reference'] = ref.tolist()
        r['errors'] = (np.array(r['computed'])/ref-1.).tolist()
    rows = [r for r in doc['phase'] if r['role']=='reserve' and r['gas'] in ('Kr','Xe')]
    maxima = np.max(abs(np.array([r['errors'] for r in rows])),axis=0)
    gases = np.max(abs(np.array([np.array(r['values'])/r['reference']-1. for r in doc['gas']])),axis=(0,1))
    from thermo_R437 import FLUIDS
    bridge_errors = []
    for row in doc['gas_transition']:
        T = row['T_K']
        props = ('Dmolar','Cvmolar','Cpmolar','isothermal_compressibility','speed_of_sound')
        ref = [[PropsSI(p,'T',T,'P',bar*1e5,FLUIDS[row['gas']]) for p in props]
               for bar in doc['pressures_transition_bar']]
        row['reference'] = ref
        if row['values'] is not None:
            err = np.array(row['values'])/ref-1.
            row['errors'] = err.tolist()
            bridge_errors.append(err)
    doc['comparison'] = dict(reference_Kr_Xe='https://doi.org/10.1021/je050186n',
            type='Comparison with reference equations, not raw data.',
            reserve_Kr_Xe=len(rows), max_reserve_Kr_Xe=maxima.tolist(),max_gas=gases.tolist(),
            transition_states_with_calculation=sum(len(x) for x in bridge_errors),
            transition_states_total=len(doc['gas_transition'])*len(doc['pressures_transition_bar']),
            transition_max=np.max(abs(np.array(bridge_errors)),axis=(0,1)).tolist(),
            max_clapeyron=max(abs(r['error_clapeyron']) for r in doc['phase']))
    write(OUT/'RACCORD_PHASE_R441.json',doc)
    print(doc['comparison'],flush=True)


if __name__=='__main__':
    p = ArgumentParser(description=__doc__)
    p.add_argument('--comparer',action='store_true')
    compare() if p.parse_args().compare else freeze()
