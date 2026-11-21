'Same native scale of size, pairs and three bodies for the twelve Kr/Xe states.'
from pathlib import Path
import csv
from functools import lru_cache
import json
import sys
import numpy as np
from scipy.constants import Avogadro,Boltzmann
from scipy.integrate import quad
from scipy.optimize import minimize,minimize_scalar
ROOT=Path(__file__).resolve().parents[1]
I,O=ROOT/'inputs',ROOT/'results'
P=I/'R453'
sys.path.insert(0,str(P))
import virials_settings as v
from virials import pair_from_scratch,CONTROLS_PVT_KR
GASES=('Kr','Xe')
TIME=[298.15,300.,348.15,350.,400.,450.,500.]
v.TEMPERATURES=np.array(TIME)
def rd(p):return json.loads(p.read_text())
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def as_csv(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        out=csv.DictWriter(f,list(rows[0]));out.writeheader();out.writerows(rows)
geometry=rd(P/'entrees/viriels/POTENTIELS_ET_ETATS_R442.json')['configs']
positions=rd(I/'RAYONS_R568.json')
coef={g:np.loadtxt(P/'entrees/viriels'/f'POTENTIEL_{g}_R444.dat') for g in GASES}
disp=rd(P/'resultats/DISPERSION_R453.json')['espèces']
old=rd(P/'entrees/viriels/VIRIELS_QUEUES_R445.json')['cases']
controls=[]
for row in old:
    controls.append(dict(g=row['gas'],T=row['T_K'],source='corrélation',B2=row['B2_ref']*1e6,B3=row['B3_ref']*1e12,
                         d2=abs(row['B2_ref'])*1e5,d3=abs(row['B3_ref'])*1e11))
for t,b2,d2,b3,d3 in CONTROLS_PVT_KR:
    controls.append(dict(g='Kr',T=t,source='re-analyzed pVT',B2=b2,B3=b3,d2=d2,d3=d3))
def scaled(g,s):
    # The potential of the pair at great distance retains its coefficient at 1/r-6.
    q=geometry[g].copy();q['sigma_m']*=s;q['energie_J']/=s**6
    return q
def second(g,temp,s):
    q=scaled(g,s); length,amp,a,b,c,d,e,f=coef[g][:8]; fact=amp*q['energie_J']/(Boltzmann*temp)
    def fn(r):
        x=r/length;pot=(a*np.exp(b*x)/x+c*np.exp(d*x)+e)/(1+f*x**6)
        return r*r*np.expm1(-fact*pot)
    return -2*np.pi*Avogadro*q['sigma_m']**3*1e6*quad(fn,0,np.inf,epsabs=1e-10)[0]
def size_law():
    scales={};fits={}
    for g in GASES:
        rc=[row for row in controls if row['g']==g]
        def objective(s):
            return max(abs(second(g,row['T'],s)-row['B2'])/row['d2'] for row in rc)
        opt=minimize_scalar(objective,bounds=(.98,1.02),method='bounded',options=dict(xatol=1e-12))
        assert opt.success
        scales[g]=opt.x;fits[g]={'s':opt.x,'max_units_of_marge':opt.fun,'origin':objective(1.)}
    physical=np.array([geometry[g]['sigma_m']*scales[g] for g in GASES])
    logL,exponent=np.linalg.solve(np.c_[np.ones(2),np.log([positions[g] for g in GASES])],np.log(physical))
    lengths={g:np.exp(logL)*positions[g]**exponent for g in ['Ar',*GASES]}
    scales={g:lengths[g]/geometry[g]['sigma_m'] for g in GASES}
    ratios=np.array([geometry[g]['sigma_m']/positions[g] for g in lengths])
    return scales,dict(L_m=np.exp(logL),exponent=exponent,B2= fits,
                       radii_colllision_m=lengths,deviations_aux_lengths_archived_pct={g:100*(s/geometry[g]['sigma_m']-1) for g,s in lengths.items()},
                       minimum_pct_law_linear_un_only_factor=100*np.tanh(np.ptp(np.log(ratios))/2))
def prepare(scales,order,limit,cut):
    models={}
    for g in GASES:
        s=scales[g];props=scaled(g,s)
        # With energy /s 6 and length 's, C9 reduced /s'3 keeps the physical term in 1/r'9.
        grid=v.grid(g,props,disp[g]['C9']/s**3,order,limit,cut)
        pair=np.array([pair_from_scratch(t,props,coef[g],order,limit)[1]*1e12 for t in TIME])
        models[g]=(grid,pair)
    return models
def compute(prep,x):
    y={};jac={}
    for g,(grid,pair) in prep.items():
        correction,der=v.prediction(grid,x);y[g]=pair+correction;jac[g]=der
    data=np.array([y[row['g']][TIME.index(row['T'])] for row in controls]);der=np.array([jac[row['g']][TIME.index(row['T'])] for row in controls])
    return data,der
def opt(prep,x0,indices):
    idx=np.array(indices);target=np.array([q['B3'] for q in controls])[idx]; window=np.array([q['d3'] for q in controls])[idx]
    @lru_cache(maxsize=3)
    def f(x):
        y,der=compute(prep,x)
        return (y[idx]-target)/window,der[idx]/window[:,None]
    def constraint(x):
        y,_=f(tuple(x[:-1]));return np.r_[x[-1]-y,x[-1]+y]
    def deriv(x):
        _,D=f(tuple(x[:-1]));return np.c_[np.r_[-D,D],np.ones(2*len(idx))]
    initial=np.r_[x0,max(abs(f(tuple(x0))[0]))+.01]
    fit=minimize(lambda x:x[-1],initial,jac=lambda x:np.array([0.,0.,0.,0.,1.]),bounds=[(-3,3)]*4+[(0,50)],
                 method='SLSQP',constraints=[dict(type='ineq',fun=constraint,jac=deriv)],options=dict(ftol=2e-10,maxiter=80))
    assert fit.success,fit.message
    return fit.x[:-1]
def main():
    scales,law=size_law(); trials=[]
    for cut in [2.15,2.45]:
        coarse=prepare(scales,14,16.,cut);mid=prepare(scales,20,24.,cut);fine=prepare(scales,26,32.,cut)
        old_witness=next(v for v in rd(P/'resultats/VIRIELS_REGLAGE_R453.json')['trials'] if v['cutoff_reduced']==cut and v['half_domaine_coefficients']==3.)
        x0=np.array(old_witness['parameters'])
        weights={10:'dix_corrélations',12:'douze_entrées'}
        for n,scope in weights.items():
            x1=opt(coarse,x0,range(n));x2=opt(mid,x1,range(n));x3=opt(fine,x2,range(n))
            first=compute(mid,x3)[0];end=compute(fine,x3)[0]
            rows=[]
            for j,row in enumerate(controls):
                b2=second(row['g'],row['T'],scales[row['g']]);b3=end[j]
                rows.append(dict(**row,B2_calc=b2,B3_calc=b3,deviation_B2_pct=100*(b2/row['B2']-1),deviation_B3_pct=100*(b3/row['B3']-1),
                      B2_units_of_marge=(b2-row['B2'])/row['d2'],B3_units_of_marge=(b3-row['B3'])/row['d3'],
                      deviation_quadrature_B3=abs(first[j]-end[j])))
            er=max(abs(row['B3_units_de_marge']) for row in rows[:n]);verified=max((abs(row['B3_calc']-row['B3'])+row['difference_quadrature_B3'])/row['d3'] for row in rows[:n])
            trials.append(dict(cutoff=cut,under_set=scope,coefficients=x3.tolist(),scope_calibrated=n,worst_B3_over_marge=er,with_deviation_quadrature=verified,
                worst_B2_over_marge=max(abs(v['B2_units_de_marge']) for v in rows),worst_B3_pct=100*max(abs(v['B3_calc']/v['B3']-1) for v in rows[:n]),
                rows_field=rows))
            print(cut,n,round(er,4),round(verified,4),'pire B2',round(trials[-1]['worst_B2_sur_marge'],4),flush=True)
    full=min((v for v in trials if v['scope_calibrated']==12),key=lambda v:v['with_deviation_quadrature'])
    short=min((v for v in trials if v['scope_calibrated']==10),key=lambda v:v['with_deviation_quadrature'])
    as_csv(O/'VIRIELS_TOUTES_TEMPERATURES.csv',full['rows'])
    dump(O/'VIRIELS_R569.json',dict(law_size=law,short_and_long_distances='Factors of length - and energy - coupled to conserve C6. The reduced coefficient C9 is divided by s3 to retain the term triple-dipole at great distance. The short-range interaction amplitudes remain common and depend on the size by the law of R453.',
            trials=trials,control_all_inputs=full,control_all_correlations=short,check_before_R569=old_witness['max_reserve'],
            scope_field='B2 en cm³/mol, B3 en cm⁶/mol². Dix états de la corrélation Kr/Xe et deux estimations expérimentales pVT de Kr avec leurs marges publiées. Les deux gaz contraignent la pente de la loi de taille, l’échelle de Ar est vérifiée sur l’estimation physique archivée. Les minima numériques des interactions ne sont pas certifiés globalement.'))
if __name__=='__main__':main()
