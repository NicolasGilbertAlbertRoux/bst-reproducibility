'Test whether a common critical anchorage improves the gas-liquid connector.'
import argparse
from functools import lru_cache
from hashlib import file_digest

import numpy as np
from scipy.constants import R
from scipy.optimize import brentq

from gas_joint import (ROOT, OUT, gas_input, phase_reference, read, write,
                              excess, stable_branches, box_search, worst, PRESSURES, IDEAL)

CRITICAL = {
    'Ar': (150.6870000000106, 13407.429658563326, 4863000.544876666),
    'Kr': (209.4795517973577, 10847.443770773472, 5525432.47699568),
    'Xe': (289.7325683307145, 8400.185085764742, 5841913.640364157),
}
LIMITS=np.array([[.90,1.10],[-15.,15.],[-15.,15.],[-15.,15.]])
ORDERS=np.array([3.,4.,5.])
CRIT_MATRIX=np.array([ORDERS,ORDERS*(ORDERS+1.),ORDERS**2*(ORDERS+1.)])


def context(config,gas,params):
    temp,B,C,V=np.asarray(config).T
    scale,*trend=params
    tc,rc,pc=CRITICAL[gas]
    volume=V*scale**3
    ratios=(tc/temp-1.)
    core=anchor(gas,float(scale))
    a=core[None,:]+ratios[:,None]*trend
    return temp,volume,B/volume-4.,C/volume**2-10.,rc,a


@lru_cache(maxsize=4096)
def anchor(gas,scale):
    tc,rho,pc=CRITICAL[gas]
    _,B,C,V=next(row['jet'][2] for row in load_critical() if row['gas']==gas)
    vol=V*scale**3
    eta=rho*vol
    b=B/vol-4.
    c=C/vol**2-10.
    _,d1,d2=excess(eta,b,c,0.,0.)
    d3=(36.-12.*eta)/(1.-eta)**5
    target=np.array([pc/(R*tc*rho)-(1.+eta*d1),
                    -(1.+2.*eta*d1+eta**2*d2),
                    -(2.*eta*d1+4.*eta**2*d2+eta**3*d3)])
    a=np.linalg.solve(CRIT_MATRIX,target)
    np.testing.assert_allclose(CRIT_MATRIX@a,target,atol=1e-10,rtol=1e-10)
    return a


@lru_cache(maxsize=1)
def load_critical():
    return read(ROOT/'entrees/COEFFICIENTS_CRITIQUES_R438.json')['cases']


def evaluate(eta,b,c,e0,co):
    a,a1,a2=excess(eta,b,c,0.,0.)
    z=eta/e0
    c3,c4,c5=co
    return (
        a+z**3*(c3+z*(c4+z*c5)),
        a1+z*z*(3.*c3+z*(4.*c4+z*5.*c5))/e0,
        a2+z*(6.*c3+z*(12.*c4+z*20.*c5))/e0**2,
    )


def saturation(row,params,check=False):
    T,v,b,c,rc,a=context(row['jet'],row['gas'],params)
    def coexist(i):
        return stable_branches(T[i],v[i],lambda eta:evaluate(eta,b[i],c[i],rc*v[i],a[i]))
    result=coexist(2)
    if result is None:
        return None,None
    p,g,l=result
    values=np.array([evaluate(v[i]*np.array([g,l]),b[i],c[i],rc*v[i],a[i])[0] for i in range(5)])
    deriv=(values[0]-8.*values[1]+8.*values[3]-values[4])/(12.*(T[3]-T[2]))
    delta_v=1./g-1./l
    latent=p*delta_v-R*T[2]**2*(deriv[0]-deriv[1])
    residual=None
    if check:
        lo,hi=coexist(1),coexist(3)
        if lo is not None and hi is not None and latent!=0.:
            residual=float(T[2]*delta_v*(hi[0]-lo[0])/(T[3]-T[1])/latent-1.)
    return np.r_[result,latent],residual


def gas(row,params):
    T,v,b,c,rc,a=context(row['jet'],row['gas'],params)
    ideal=np.asarray(PRESSURES)*1e5*v[2]/(R*T[2])
    def pressure(eta):
        return eta*(1.+eta*evaluate(eta,b[2],c[2],rc*v[2],a[2])[1])
    eta=np.array([brentq(lambda z:pressure(z)-x,.65*x,1.4*x,xtol=1e-15) for x in ideal])
    rho=eta/v[2]
    vals=np.array([evaluate(v[i]*rho,b[i],c[i],rc*v[i],a[i]) for i in range(5)])
    A,A1,A2=np.moveaxis(vals,1,0)
    step=T[3]-T[2]
    def derivative(A):
        return (A[0]-8.*A[1]+8.*A[3]-A[4])/(12.*step)
    att=(-A[0]+16.*A[1]-30.*A[2]+16.*A[3]-A[4])/(12.*step**2)
    Cv=IDEAL-R*T[2]*(2.*derivative(A)+T[2]*att)
    Z=1.+v[:,None]*rho*A1
    pt=R*rho*(Z[2]+T[2]*derivative(Z))
    pr=R*T[2]*(1.+2.*eta*A1[2]+eta**2*A2[2])
    if min(pr.min(),Cv.min())<=0.:
        raise ValueError('nonphysical gas branch')
    Cp=Cv+T[2]*pt**2/(rho**2*pr)
    return np.column_stack((rho,Cv,Cp,1./(rho*pr),np.sqrt(Cp*pr/(Cv*row['mass']))))


def main(compare=False):
    input_path=OUT/'PREDICTIONS_CRITIQUES_GELEES_R438.json'
    if compare:
        result=read(input_path)
        with input_path.open('rb') as f:
            result['sha256_predictions']=file_digest(f,'sha256').hexdigest()
        for r in result['phase']:
            if r['role']=='fit':
                continue
            ref=phase_reference(r['gas'],r['T_K'])
            assert r['reference'] is None
            r['reference']=ref.tolist()
            if r['prediction'] is not None:
                r['deviations']=(r['prediction']/ref-1.).tolist()
        write(OUT/'CROISEMENT_CRITIQUE_R438.json',result)
        return
    cfg=read(ROOT/'entrees/COEFFICIENTS_PHASE_R438.json')['cases']
    # This refinement follows the first R438 comparison. The new reference
    # temperatures for this model are disjoint from that previous challenge.
    train=[r for r in cfg if r['gas']!='Kr' and r['tau'] in (.80,.90,.97)]
    for r in train:
        r['ref']=phase_reference(r['gas'],r['T_K'])
    test_gas=gas_input()
    def objective(p):
        errors=[]
        for r in train:
            val,_=saturation(r,p)
            if val is None:
                return 15.
            errors.append(worst(val/r['ref']-1.))
        try:
            for r in test_gas:
                if r['T_K']!=450.:
                    errors.append(worst(gas(r,p)/r['reference']-1.))
        except ValueError:
            return 20.
        return max(errors)
    params,history=box_search(objective,LIMITS,np.array([1.,0.,0.,0.]),4438)
    print('critical',params,history['maximum_fit'],history['evaluations'],flush=True)
    new=read(ROOT/'entrees/COEFFICIENTS_PHASE_RESERVE_CRITIQUE_R438.json')['cases']
    rows=[]
    for r in (*cfg,*new):
        val,check=saturation(r,params,True)
        ref=r.get('ref')
        rows.append(dict(gas_field=r['gas'],tau=r['tau'],T_K=r['T_K'],role='fit' if ref is not None else
                         'reserve' if r in new else 'retrospective',prediction=None if val is None else val.tolist(),
                         reference=None if ref is None else ref.tolist(),Clapeyron=check,
                         deviations=None if val is None or ref is None else (val/ref-1.).tolist()))
    out=dict(revision=438,critical_anchors=CRITICAL,parameters=params.tolist(),recherche=history,
             scope='B2 and B3 of R435. The three coordinates of the critical point of each gas are anchors. '
                    'Les trois amplitudes en température sont communes et calibrées sur six saturations Ar/Xe et la phase gazeuse. '
                    'The conditions for the reservation shall be laid down before reading their references. Possible non-level branches '
                    'isolation remains indeterminate. This construction is a conditional thermodynamic interpolation.',
             gas=[dict(gas_field=r['gas'],T_K=r['T_K'],deviations=(gas(r,params)/r['reference']-1.).tolist()) for r in test_gas],
             phase=rows)
    write(input_path,out)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--comparer',action='store_true')
    main(ap.parse_args().compare)
