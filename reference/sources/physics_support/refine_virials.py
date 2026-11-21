'Extends the short part of the potential, retaining the same targets Kr and Xe.'
import numpy as np
from scipy.optimize import linprog
from functools import lru_cache
from viriels_common import (I,O,rd,dump,as_csv,GASES,TIME,controls,prepare,scaled,size_law,
                           second,compute,v,disp)
def add_modes(prep,scales,order,limit,cut):
    old=(v.SITES,v.WIDTHS)
    try:
        v.SITES=(1.14,1.60);v.WIDTHS=(.12,.28)
        for gas in GASES:
            s=scales[gas];model,pair=prep[gas]
            new=v.grid(gas,scaled(gas,s),disp[gas]['C9']/s**3,order,limit,cut)
            for a,b in zip(model[:4],new[:4]):
                np.testing.assert_allclose(a,b,rtol=1e-13,atol=1e-13)
            prep[gas]=((*model[:4],np.c_[model[4],new[4][:,:2]]),pair)
    finally:
        v.SITES,v.WIDTHS=old
    return prep
def prediction(prep,x,thermal):
    if not thermal:return compute(prep,x)
    # An overall slope factor in T in the first mode. The law is then effective.
    res={};jac={}
    for gas,(grid,pair) in prep.items():
        tails,w,e2,e3,basis=grid; vals=[]; derivatives=[]
        for k,temp in enumerate(TIME):
            tau=(temp-350.)/100.;features=np.c_[basis,basis[:,0]*tau]
            exp=np.exp(-e2/temp);e=np.exp(np.clip(-(e3+features@x)/temp,-745,500))
            vals.append(pair[k]+tails[k]+np.sum(w*exp*(1-e)))
            derivatives.append((w*exp*e/temp)@features)
        res[gas]=vals;jac[gas]=derivatives
    return (np.array([res[row['g']][TIME.index(row['T'])] for row in controls]),
            np.array([jac[row['g']][TIME.index(row['T'])] for row in controls]))
def fit(prep,x0,thermal):
    obs=np.array([q['B3'] for q in controls]);d=np.array([q['d3'] for q in controls])
    @lru_cache(maxsize=3)
    def eval_(x):
        pred,J=prediction(prep,x,thermal)
        return (pred-obs)/d,J/d[:,None]
    x=np.array(x0,dtype=float); radius=.5
    value=max(abs(eval_(tuple(x))[0]))
    for _ in range(90):
        residual,J=eval_(tuple(x)); n=len(residual)
        bounds=[(max(-radius,-3.-yi),min(radius,3.-yi)) for yi in x]+[(0.,None)]
        lp=linprog(np.r_[np.zeros(len(x)),1.],A_ub=np.c_[np.r_[J,-J],-np.ones(2*n)],
                   b_ub=np.r_[-residual,residual],bounds=bounds)
        assert lp.success,lp.message
        dx=lp.x[:-1];target=lp.x[-1]
        if value-target<2e-9:break
        candidate=np.clip(x+dx,-3,3);new=max(abs(eval_(tuple(candidate))[0]))
        if new<value-2e-10:
            shrink=(value-new)/(value-target)
            x,value=candidate,new
            if shrink>.8:radius=min(1.,1.4*radius)
            if shrink<.25:radius*=.6
        else:radius*=.45
        if radius<2e-7:break
    return x

def main():
    old=rd(O/'VIRIELS_R569.json'); scales,_=size_law();cut=2.45
    initial=next(q for q in old['trials'] if q['cutoff']==cut and q['scope_calibrated']==12)
    # Quadratures are not additional observations.
    mid=add_modes(prepare(scales,17,20.,cut),scales,17,20.,cut)
    fine=add_modes(prepare(scales,23,26.,cut),scales,23,26.,cut)
    final=add_modes(prepare(scales,29,34.,cut),scales,29,34.,cut)
    check=add_modes(prepare(scales,33,40.,cut),scales,33,40.,cut)
    out=[]
    for thermal in (False,True):
        start=np.r_[initial['coefficients'],0.,0.]
        if thermal:start=np.r_[out[0]['coefficients'],0.]
        a=fit(mid,start,thermal);b=fit(fine,a,thermal);c=fit(final,b,thermal)
        z,_=prediction(check,c,thermal);last,_=prediction(final,c,thermal)
        rows=[]
        for j,q in enumerate(controls):
            b2=second(q['g'],q['T'],scales[q['g']]); dev=(z[j]-q['B3'])/q['d3']; u=abs(z[j]-last[j])
            rows.append(dict(**q,B2_calc=b2,B3_calc=z[j],B2_units_of_marge=(b2-q['B2'])/q['d2'],B3_units_of_marge=dev,
                  B2_deviation_pct=100*(b2/q['B2']-1),B3_deviation_pct=100*(z[j]/q['B3']-1),variation_quadrature_B3=u))
        bound=max((abs(q['B3_calc']-q['B3'])+q['variation_quadrature_B3'])/q['d3'] for q in rows)
        out.append(dict(thermique=thermal,amplitude_max_allowed=3.,coefficients=c.tolist(),rows_field=rows,worst_B2_margins=max(abs(q['B2_units_de_marge']) for q in rows),
                        worst_B3_margins=max(abs(q['B3_units_de_marge']) for q in rows),with_variation_quadrature=bound,
                        worst_B3_correlation_pct=max(abs(q['B3_difference_pct']) for q in rows if q['source']=='corrélation')))
        print('extension',thermal,round(bound,5),np.round(c,3),flush=True)
    dump(O/'EXTENSION_TRIMERE_R569.json',dict(order_check=33,cutoff=2.45,law_size=old['law_size'],trials=out,
          interpretation='The margins are those of the twelve inputs, correlations and pVT reanalysis. Quadrature sensitivity is a numerical check. The six-amplitude material branch is independent of the temperature in the potential. The seven-amplitude branch is an effective potential dependent on T. The two modes added do not change the pair or the long-range coefficient.'))
    as_csv(O/'CONTROLE_THERMO_R569.csv',min(out,key=lambda z:z['with_variation_quadrature'])['rows'])
if __name__=='__main__': main()
