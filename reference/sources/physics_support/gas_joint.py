'Seek the same free energy for gas, coexistence and latent heat.'
import argparse
import json
from pathlib import Path
from hashlib import file_digest
import sys

import numpy as np
from scipy.constants import R
from scipy.optimize import brentq, differential_evolution, minimize
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
OUT = ROOT/'results'
sys.path.insert(0,str(BASE/'BST_R437'))
from coexistence_reprise_R437 import volume_bh
from thermo_R437 import FLUIDS, PRESSURES, TIME

BOUNDS = np.array([[.83,1.18],[0.,24.],[-25.,25.]])
SPINODAL_GRID = np.linspace(0.,.9995,161)
TOL = (.05,.10)
IDEAL = 1.5*R
P_NAMES = ('pressure','density_vapeur','density_liquide','chaleur_latente')
G_NAMES = ('rho','Cv','Cp','kappa_T','son')


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p,data):
    tmp=p.with_suffix('.part')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(p)


def coefficients(raw,params):
    T,B,C,v=np.asarray(raw).T
    scale,screen,d=params
    v=v*scale**3
    return T,v,B/v-4.,C/v**2-10.,screen,d


def excess(eta,b,c,s,d):
    w=1.+s*eta
    p=(c/2.+d*eta)*eta**2
    p1=(c+3.*d*eta)*eta
    p2=c+6.*d*eta
    v=1.-eta
    a=eta*(4.-3.*eta)/v**2+b*eta+p/w
    d1=(4.-2.*eta)/v**3+b+p1/w-s*p/w**2
    d2=(10.-4.*eta)/v**4+p2/w-2.*s*p1/w**2+2.*s*s*p/w**3
    return a,d1,d2


def pressure_eta(eta,b,c,s,d):
    return eta*(1.+eta*excess(eta,b,c,s,d)[1])


def slope_eta(eta,b,c,s,d):
    _,a1,a2=excess(eta,b,c,s,d)
    return 1.+2.*eta*a1+eta*eta*a2


def stable_branches(T,v,energy):
    def pressure(eta):
        return eta*(1.+eta*energy(eta)[1])
    def slope(eta):
        _,a1,a2=energy(eta)
        return 1.+2.*eta*a1+eta*eta*a2
    signs=np.sign(slope(SPINODAL_GRID))
    ix=np.flatnonzero(signs[:-1]*signs[1:]<0.)
    if len(ix)!=2:
        return None
    g,l=[brentq(slope,SPINODAL_GRID[i],SPINODAL_GRID[i+1]) for i in ix]
    p_min=max(0.,pressure(l))
    p_max=pressure(g)
    if p_max<=p_min:
        return None
    def dens(p):
        return [brentq(lambda y:pressure(y)-p,lo,hi,xtol=1e-19)
                for lo,hi in ((0.,g),(l,.999999))]
    def difference(g0,l0):
        def mu(x):
            a,a1,_=energy(x)
            return np.log(x)+a+x*a1
        return mu(l0)-mu(g0)
    def gap(logp):
        return difference(*dens(np.exp(logp)))
    pad=3e-12*max(1.,abs(p_max),abs(p_min))
    if p_max-p_min<2.*pad:
        return None
    p0=p_min+pad if p_min>0 else p_max*1e-17
    p1=p_max-pad
    low,high=np.log([p0,p1])
    try:
        if gap(low)*gap(high)>=0.:
            return None
        logp=brentq(gap,low,high,xtol=1e-11)
        p=np.exp(logp)
        g,l=dens(p)
        if g<=0. or abs(difference(g,l))>=2e-7:
            return None
    except ValueError:
        return None
    return np.array([p*R*T/v,g/v,l/v])


def branches(T,v,b,c,s,d):
    return stable_branches(T,v,lambda eta:excess(eta,b,c,s,d))


def phase_state(raw,params,verify=False):
    T,v,b,c,s,d=coefficients(raw,params)
    j=len(T)//2
    state=branches(T[j],v[j],b[j],c[j],s,d)
    if state is None:
        return None, None
    p,g,l=state
    a=excess(v[:,None]*np.array([g,l]),b[:,None],c[:,None],s,d)[0]
    derivative=(a[0]-8.*a[1]+8.*a[3]-a[4])/(12.*(T[3]-T[2]))
    delta_v=1./g-1./l
    latent=p*delta_v-R*T[j]**2*(derivative[0]-derivative[1])
    result=np.r_[state,latent]
    check=None
    if verify:
        side=[branches(T[i],v[i],b[i],c[i],s,d) for i in (1,3)]
        if all(x is not None for x in side):
            clap=T[j]*delta_v*(side[1][0]-side[0][0])/(T[3]-T[1])
            check=float(clap/latent-1.)
    return result,check


def gas_states(row,params):
    T,v,b,c,s,d=coefficients(row['jet'],params)
    j=len(T)//2
    p=np.asarray(PRESSURES)*1e5
    rho_ideal=p/(R*T[j])
    eta=np.array([brentq(lambda e:pressure_eta(e,b[j],c[j],s,d)-x,.65*x,1.4*x,xtol=1e-15)
                  for x in rho_ideal*v[j]])
    rho=eta/v[j]
    a,a1,a2=excess(v[:,None]*rho,b[:,None],c[:,None],s,d)
    step=T[3]-T[j]
    dt=(a[0]-8.*a[1]+8.*a[3]-a[4])/(12.*step)
    dtt=(-a[0]+16.*a[1]-30.*a[2]+16.*a[3]-a[4])/(12.*step**2)
    res=-R*T[j]*(2.*dt+T[j]*dtt)
    cv=IDEAL+res
    z=1.+v[:,None]*rho*a1
    zt=(z[0]-8.*z[1]+8.*z[3]-z[4])/(12.*step)
    p_t=R*rho*(z[j]+T[j]*zt)
    p_r=R*T[j]*(1.+2.*eta*a1[j]+eta*eta*a2[j])
    cp=cv+T[j]*p_t*p_t/(rho*rho*p_r)
    if min(cv.min(),p_r.min(),cp.min())<=0.:
        raise ValueError('gas stability')
    sound=np.sqrt(cp*p_r/(cv*row['mass']))
    return np.column_stack((rho,cv,cp,1./(rho*p_r),sound)),res


def gas_input():
    base=read(BASE/'BST_R437/resultats/THERMODYNAMIQUE_R437.json')['cases']
    gel=read(BASE/'BST_R432/resultats/gaz/GEL_GAZ_R432.json')['predictions']
    p=read(BASE/'BST_R435/resultats/VIRIEL_R435.json')['p_size']
    ref={k:next(r for r in gel if r['gas']==k and r['p_size']==p) for k in FLUIDS}
    from CoolProp.CoolProp import PropsSI
    out=[]
    for r in base:
        T=r['T_K']; species=r['gas']
        steps=np.arange(-2,3)*.5
        B,der,curv=np.asarray(r['jet_B2_B3_ATM'])
        Tvalues=T+steps
        jets=B+steps[:,None]*der+.5*steps[:,None]**2*curv
        jet=np.column_stack((Tvalues,jets[:,0],jets[:,1]+jets[:,2],
                             [volume_bh(ref[species],t) for t in Tvalues]))
        # Reference observables were already present in R437. Only the
        # first and second derivative of the local virial jet are used.
        values=[[x['rho_reference'],*[x['reference'][k] for k in G_NAMES[1:]]] for x in r['cases']]
        out.append(dict(gas_field=species,T_K=T,jet=jet,mass=PropsSI('molar_mass',FLUIDS[species]),
                        reference=np.asarray(values),residual=[x['reference']['Cv_residual'] for x in r['cases']]))
    return out


def phase_reference(gas,T):
    from CoolProp.CoolProp import PropsSI
    fluid=FLUIDS[gas]
    def f(key,quality):
        return PropsSI(key,'T',T,'Q',quality,fluid)
    return np.array([f('P',0.),f('Dmolar',1.),f('Dmolar',0.),f('Hmolar',1.)-f('Hmolar',0.)])


def worst(values):
    return float(np.max(np.abs(values)))


def box_search(evaluate,bounds,base,seed):
    # Sobol revisits the entire window before local subdivision near the
    # sampled minima. No box is excluded from its centre or sample.
    active=np.flatnonzero(bounds[:,1]>bounds[:,0])
    def full(v):
        a=bounds[:,0].copy()
        a[active]=v
        return a
    low,high=bounds[active].T
    sample=qmc.scale(qmc.Sobol(len(active),scramble=True,seed=seed).random_base2(8),low,high)
    histories=[]
    best=[]
    for t in sample:
        v=full(t)
        score=evaluate(v)
        histories.append([*v,score])
        best.append((score,t))
    best.sort(key=lambda x:x[0])
    evaluations=len(histories)
    def f(t):
        nonlocal evaluations
        evaluations+=1
        return evaluate(full(t))
    if len(active)>1:
        # Search throughout the declared window, carrying the best global samples.
        of=differential_evolution(f,list(zip(low,high)),x0=best[0][1],popsize=8,maxiter=32,
                                  tol=2e-4,polish=False,rng=seed)
        best.append((float(of.fun),of.x))
    best.append((f(np.asarray(base)[active]),np.asarray(base)[active]))
    best.sort(key=lambda x:x[0])
    for _,start in best[:3]:
        v=minimize(f,start,method='Powell',bounds=list(zip(low,high)),options={'xtol':2e-5,'ftol':2e-6})
        best.append((float(v.fun),v.x))
    value,p=min(best,key=lambda x:x[0])
    return full(p),dict(evaluations=evaluations,maximum_fit=value,
                         sweep_integer=histories,box=bounds.tolist(),
                         certification_continue=False)


def common_search():
    cfg=read(ROOT/'entrees/COEFFICIENTS_PHASE_R438.json')['cases']
    old=[r for r in cfg if r['tau'] in (.8,.9,.97)]
    train=[r for r in old if r['gas']!='Kr' and r['tau'] in (.8,.97)]
    gas=gas_input()
    for row in old:
        row['ref']=phase_reference(row['gas'],row['T_K'])
    def objective(params):
        violations=[]
        for row in train:
            calc,_=phase_state(row['jet'],params)
            if calc is None:
                return 15.
            violations.append(worst(calc/row['ref']-1.))
        try:
            for r in gas:
                if r['T_K']==450.:
                    continue
                state,_=gas_states(r,params)
                violations.append(worst(state/r['reference']-1.))
        except ValueError:
            return 20.
        return max(violations)

    def all_gas(p):
        out=[]
        for r in gas:
            state,res=gas_states(r,p)
            out.append(dict(gas_field=r['gas'],T_K=r['T_K'],reference=r['reference'].tolist(),computed=state.tolist(),
                            deviations=(state/r['reference']-1.).tolist(),
                            deviations_Cv_residual=(res/np.asarray(r['residual'])-1.).tolist()))
        return out

    models=[('R437_diametre_nominal',np.array([1.,0.,0.]),{})]
    for name,n in (('diametre_common',1),('diametre_and_ecran',2),('diametre_ecran_et_B4',3)):
        bounds=BOUNDS.copy()
        bounds[n:]=0.
        par,history=box_search(objective,bounds,models[-1][1],438+n)
        models.append((name,par,history))
        print(name,par,history['maximum_fit'],history['evaluations'],flush=True)
    records=[]
    for name,p,history in models:
        pg=all_gas(p)
        phase=[]
        for row in cfg:
            new=row['tau'] in (.86,.94)
            value,clap=phase_state(row['jet'],p,verify=True)
            ref=row.get('ref')
            phase.append(dict(gas_field=row['gas'],tau=row['tau'],T_K=row['T_K'],
                role='reserve_new' if new else 'fit' if row in train else 'reserve_retrospective',
                calculation=None if value is None else value.tolist(), check_Clapeyron=clap,
                reference=None if ref is None else ref.tolist(),
                deviations=None if ref is None or value is None else (value/ref-1.).tolist()))
        records.append(dict(model=name,parameters=p.tolist(),recherche=history,gas=pg,phase=phase))
    # B2/B3 preservation, gas and coexistence calls all share the same a.
    check=phase_state(cfg[0]['jet'],models[0][1])[0]
    old_check=read(BASE/'BST_R437/resultats/COEXISTENCE_RESOMMEE_R437.json')['cases'][0]['variantes'][1]
    np.testing.assert_allclose(check[:3],list(old_check[k] for k in ('pressure_Pa','vapeur','liquide')),rtol=2e-8)
    for p in (models[-1][1],np.array([.94,3.,3.])):
        T,v,b,c,s,d=coefficients(cfg[0]['jet'][2],p)
        eta=1e-6
        fitted=(pressure_eta(eta,b,c,s,d)/eta-1.-(b+4.)*eta)/(eta*eta)
        np.testing.assert_allclose(fitted,c+10.,atol=5e-4)
    out=dict(revision=438,phase_observables=P_NAMES,gas_observables=G_NAMES,
             temperatures_gas=TIME,pressures_bar=PRESSURES,
             model="a_ex/RT = CS(η) + (B₂/v−4)η + [(B₃/v²−10)η²/2+dη³]/(1+sη)",
             hypothese='Same diameter scale, screen and correction of order four in the three gases. B2 and B3 remain R435.',
             codes_non_evaluated={"15.0":'unresolved co-existence', "20.0":'unqualified gas branch at point'},
             references_reserve_new_lues_before_gel=False,models=records)
    write(OUT/'PREDICTIONS_GAZ_GELEES_R438.json',out)


def compare():
    path=OUT/'PREDICTIONS_GAZ_GELEES_R438.json'
    frozen=read(path)
    with path.open('rb') as f:
        digest=file_digest(f,'sha256').hexdigest()
    assert all(x['reference'] is None for model in frozen['models']
               for x in model['phase'] if x['role']=='reserve_new')
    for m in frozen['models']:
        for x in m['phase']:
            if x['role']!='reserve_new':
                continue
            x['reference']=phase_reference(x['gas'],x['T_K']).tolist()
            if x['calculation'] is not None:
                x['deviations']=(np.asarray(x['calculation'])/x['reference']-1.).tolist()
    write(OUT/'CROISEMENT_GAZ_R438.json',dict(**frozen,sha256_predictions=digest,
                                            step='Comparison after freezing of parameters and values'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('step',choices=('recherche','compare'))
    args=p.parse_args()
    if args.step=='recherche':
        common_search()
    else:
        compare()
