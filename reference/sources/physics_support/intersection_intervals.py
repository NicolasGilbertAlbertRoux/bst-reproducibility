'Common retrospective calibration and a continuously enclosed parameter box.\n\nThe box is limited to the declared effective atom/radius/binding/weak subset.\nNo primordial embedding, human data fit, or all-domain closure is claimed.'
from pathlib import Path
import json,csv,hashlib,math,time
import numpy as np
import mpmath as mp
from scipy.optimize import linprog
from scipy.integrate import quad
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parent;I=R/'inputs';O=R/'results';O.mkdir(exist_ok=True)
mp.mp.dps=65;iv=mp.iv;iv.dps=45
def read(n):return json.loads((I/n).read_text())
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def v(x):return iv.mpf(str(x))
def interval(lo,hi):return iv.mpf([str(lo),str(hi)])
def ends(x):return [float(np.nextafter(float(x.a),-np.inf)),float(np.nextafter(float(x.b),np.inf))]
HB=v('6.62607015e-34')/(2*iv.pi)/v('1.602176634e-19')*v('1e-9')
HBC=v('6.62607015e-34')/(2*iv.pi)*v(299792458)/v('1.602176634e-19')*v('1e9')
hb=float(6.62607015e-34/(2*math.pi)/1.602176634e-19*1e-9)
hbc=float(6.62607015e-34/(2*math.pi)*299792458/1.602176634e-19*1e9)

def delta(k,a,ctx):return a*a/(k+ctx.sqrt(k*k-a*a))
def gap(lo,hi,a,ctx):
    nl,kl=lo;nh,kh=hi;dl=nl-delta(kl,a,ctx);dh=nh-delta(kh,a,ctx)
    dif=(nh-nl) if kh==kl else (nh-nl)+delta(kl,a,ctx)-delta(kh,a,ctx)
    sl=ctx.sqrt(1+(a/dl)**2);sh=ctx.sqrt(1+(a/dh)**2)
    return a*a*dif*(dh+dl)/(dl*dl*dh*dh*sl*sh*(sl+sh))
def fine(a,ctx):
    s1=ctx.sqrt(1-a*a);s2=ctx.sqrt(4-a*a)
    # Rationalized difference of the two small deltas: no cancellation.
    dd=(1+3/(s1+s2))/((1+s1)*(2+s2))
    d1=4-a*a/(1+s1);d2=4-a*a/(2+s2)
    r1=ctx.sqrt(1+(a/d1)**2);r2=ctx.sqrt(1+(a/d2)**2)
    return a**4*dd*(d1+d2)/(d1*d1*d2*d2*r1*r2*(r1+r2))

transitions=[((1,1),(2,1)),((1,1),(3,1)),((2,1),(8,3)),((2,1),(4,1)),((2,1),(4,2))]
obs=read('RAIES_H_MESUREES_R502.json');specrows=[obs['calibration']]+obs['held_out_from_scale_fit']
rad=read('CROISEMENT_RAYONS_R502.json');beta=read('DUREES_CROISEES_R503.json')
refs=read('MESURES_ET_REFERENCES_R503.json');angular=read('COURANT_ORIENTE_R503.json')
nuclides=read('NOYAUX_R453.json')['channels_beta']
me=refs['constants']['me_GeV'];me_u=refs['constants']['me_u_GeV'];G=refs['MuLan2011']['G_GeV_m2'];G_u=refs['MuLan2011']['G_u']
lam=beta['fixed_parameters']['lambda_value'];lbox=angular['selected']['lambda_box']
fineobs=read('CROISEMENT_ELECTRIQUE_R502.json')['fine_component_separation']['observed_Hz']

def spec_error(a):
    aa=mp.mpf(str(a));scale=mp.mpf(str(me))/(2*mp.pi*mp.mpf(str(hb)))
    return [float(scale*gap(lo,hi,aa,mp.mp))/r['frequency_Hz']-1 for (lo,hi),r in zip(transitions,specrows)]+[float(scale*fine(aa,mp.mp))/fineobs-1]

def B(t,m,a,z,ctx=math):
    if t==0:return 2*ctx.pi*z*a*m*m
    E=t+m;p=ctx.sqrt(t*(t+2*m));x=2*ctx.pi*z*a*E/p
    return 2*ctx.pi*z*a*E*E/(-ctx.expm1(-x))
def beta_integral(Q,m,a,z):
    return Q**3*quad(lambda u:B(Q*u,m,a,z)*(1-u)**2,0,1,epsabs=1e-30,epsrel=2e-12,limit=160)[0]
def tau(Q,m,a,z,L=lam,g=G,C=1):return 2*math.pi**3*hb/(C*g*g*(1+3*L*L)*beta_integral(Q,m,a,z))

def load_ame():
    rows=[]
    for s in (I/'mass_1.mas20.txt').read_text().splitlines():
        try:
            A,Z=int(s[14:19]),int(s[9:14]);fields=[s[28:42],s[42:54],s[54:67],s[68:78]]
            if any('#' in t or '*' in t for t in fields):continue
            _,_,b,ub=map(float,fields)
        except ValueError:continue
        y=b*A/1000;u=ub*A/1000
        if A>=16 and y>0 and u<=.1:rows.append(dict(A=A,Z=Z,B=y,u=u,split='train' if A%5 in [0,1,2] else 'historical_holdout'))
    return rows
def features(A,Z):
    parity=0 if A%2 else (1 if Z%2==0 else -1)
    return np.array([A,-A**(2/3),-Z*(Z-1)/A**(1/3),-(A-2*Z)**2/A,parity/math.sqrt(A)])
rows=load_ame();F=np.array([features(r['A'],r['Z']) for r in rows]);ys=np.array([r['B'] for r in rows]);us=np.array([r['u'] for r in rows]);train=np.array([r['split']=='train' for r in rows])
assert len(rows)==2355 and sum(train)==1422

def error_bound(pred,observed):
    z=pred/observed-1;b=ends(z);return max(abs(x) for x in b),b

def make_box(center,w):
    box={k:[x*(1-w),x*(1+w)] for k,x in center.items()}
    box.update(lambda_value=lbox,me_GeV=[me-me_u,me+me_u],GF_GeVm2=[G-G_u,G+G_u])
    return box

def bounds(box,strict=False):
    a=interval(*box['alpha']);r0=interval(*box['r0_fm']);m=interval(*box['me_GeV']);L=interval(*lbox);GG=interval(*box['GF_GeVm2']);CC=interval(*box['C_rate'])
    ac=v('.6')*a*HBC/r0;scale=m/(2*iv.pi*HB)
    spectral=[]
    for (lo,hi),row in zip(transitions,specrows):
        e,b=error_bound(scale*gap(lo,hi,a,iv),interval(row['frequency_Hz']-row['uncertainty_Hz'],row['frequency_Hz']+row['uncertainty_Hz']))
        spectral.append(dict(label=row['label'],max_relative_error=e,error_interval=b))
    fe,fb=error_bound(scale*fine(a,iv),interval(fineobs-6000,fineobs+6000))
    radii=[]
    for row in rad['radii']:
        p=iv.sqrt(v('.6'))*r0*v(row['A'])**(v(1)/3)
        e,b=error_bound(p,interval(row['rms_fm']-row['u_fm'],row['rms_fm']+row['u_fm']))
        radii.append(dict(label=row['label'],max_relative_error=e,error_interval=b))
    cs=[interval(*box['av_MeV']),interval(*box['as_MeV']),ac,interval(*box['aa_MeV']),interval(*box['ap_MeV'])]
    if strict:
        bulk=[]
        for row in rows:
            A,Z=row['A'],row['Z'];AA=v(A);parity=0 if A%2 else (1 if Z%2==0 else -1)
            terms=[AA,-AA**(v(2)/3),-v(Z*(Z-1))/AA**(v(1)/3),-v((A-2*Z)**2)/AA,v(parity)/iv.sqrt(AA)]
            prediction=sum(t*c for t,c in zip(terms,cs))
            e,b=error_bound(prediction,interval(row['B']-row['u'],row['B']+row['u']))
            bulk.append(dict(**row,max_relative_error=e,error_interval=b))
        bulkmax=max(x['max_relative_error'] for x in bulk)
    else:
        cbs=np.array([ends(c) for c in cs]);lower=np.sum(np.where(F>=0,F*cbs[:,0],F*cbs[:,1]),axis=1);upper=np.sum(np.where(F>=0,F*cbs[:,1],F*cbs[:,0]),axis=1)
        bulkmax=float(max(np.max(abs(lower/(ys+us)-1)),np.max(abs(upper/(ys-us)-1))))+1e-10;bulk=None
    nrm=1+3*L*L
    angularbounds=[]
    coeffs={'a':(1-L*L)/nrm,'A':-2*L*(L+1)/nrm,'B':2*L*(L-1)/nrm}
    for row in angular['selected']['comparisons']:
        e,b=error_bound(coeffs[row['observable']],interval(row['observed']-row['u'],row['observed']+row['u']))
        angularbounds.append(dict(id=row['id'],max_relative_error=e,error_interval=b))
    lifeb=[];integrals=[]
    for j in [0,1]:
        row=beta['table'][j];qlo=(row['Q_keV']-row['Q_margin_keV'])*1e-6;qhi=(row['Q_keV']+row['Q_margin_keV'])*1e-6;Z=row['Z_daughter_received']
        if strict:
            low=riemann_integral(qlo,box['me_GeV'][0],box['alpha'][0],Z)
            high=riemann_integral(qhi,box['me_GeV'][1],box['alpha'][1],Z)
            integral=interval(low['lower'],high['upper']);integrals.append(dict(isotope=row['isotope'],lower_endpoint=low,upper_endpoint=high))
            value=2*iv.pi**3*HB/(CC*GG**2*nrm*integral)
            tlo,thi=ends(value)
        else:
            tlo=tau(qhi,box['me_GeV'][1],box['alpha'][1],Z,lbox[0],box['GF_GeVm2'][1],box['C_rate'][1])
            thi=tau(qlo,box['me_GeV'][0],box['alpha'][0],Z,lbox[1],box['GF_GeVm2'][0],box['C_rate'][0])
        if j==0:
            for ref in beta['neutron_comparisons']:
                e,b=error_bound(interval(tlo,thi),interval(*ref['reading_box_s']));lifeb.append(dict(label=ref['id'],max_relative_error=e,error_interval=b))
        else:
            u=float(nuclides[1]['marge_source']['u'])*365.25*86400
            e,b=error_bound(interval(tlo,thi)*iv.ln(2),interval(row['observed_half_life_s']-u,row['observed_half_life_s']+u))
            lifeb.append(dict(label='tritium',max_relative_error=e,error_interval=b))
    maxima={'H-frequencies':max(x['max_relative_error'] for x in spectral),'Fine separation H':fe,'Cinq rayons':max(x['max_relative_error'] for x in radii),'2355 liaisons':bulkmax,'Angular correlations':max(x['max_relative_error'] for x in angularbounds),'Neutron/tritium':max(x['max_relative_error'] for x in lifeb)}
    return dict(maxima=maxima,maximum=max(maxima.values()),spectra=spectral,fine_error_interval=fb,radii=radii,angular=angularbounds,lifetimes=lifeb,derived_ac_MeV=ends(ac),integral_bounds=integrals,bulk=bulk)

def riemann_integral(Q,m,a,Z,N=4096):
    'Outward interval lower/upper Darboux product bounds, not quadrature estimates.\n\nB(T) increases with T; (Q-T)^2 decreases. On each cell, product\nlower/upper bounds follow from opposite endpoint values. Parameters are\npositive; B also increases with m and alpha. The integral increases with Q.'
    qq=v(Q);mm=v(m);aa=v(a);of=v(1)/N;low=v(0);high=v(0)
    def bb(u):
        if u==0:return 2*iv.pi*Z*aa*mm*mm
        T=qq*u;E=T+mm;p=iv.sqrt(T*(T+2*mm));x=2*iv.pi*Z*aa*E/p
        return 2*iv.pi*Z*aa*E*E/(-iv.expm1(-x))
    leftB=bb(v(0))
    for k in range(N):
        left=v(k)/N;right=v(k+1)/N;rightB=bb(right)
        low+=leftB*(1-right)**2*of;high+=rightB*(1-left)**2*of;leftB=rightB
    low*=qq**3;high*=qq**3
    lo=ends(low)[0];hi=ends(high)[1];reference=beta_integral(Q,m,a,Z)
    assert lo<=reference<=hi
    # Independent 65-digit quadrature, solely a check of the enclosure.
    qm=mp.mpf(str(Q));mm0=mp.mpf(str(m));am=mp.mpf(str(a))
    precise=qm**3*mp.quad(lambda u:B(qm*u,mm0,am,Z,mp.mp)*(1-u)**2,[0,mp.mpf('.25'),1])
    assert mp.mpf(str(lo))<=precise<=mp.mpf(str(hi))
    return dict(lower=lo,upper=hi,relative_width=(hi-lo)/reference,subintervals=N,quad=reference,mp_quad=float(precise))

def main():
    begin=time.time()
    # Balanced maximum relative residual on H frequencies and separation.
    alo,ahi=.006921564437216338,.0076520612766264135
    for _ in range(65):
        amid=(alo+ahi)/2;e=spec_error(amid)
        if max(e)+min(e)>0:ahi=amid
        else:alo=amid
    alpha=(alo+ahi)/2;r0=rad['r0_fm'];ac=.6*alpha*hbc/r0
    X=F[train]/ys[train,None];A=np.r_[np.c_[X,-np.ones(sum(train))],np.c_[-X,-np.ones(sum(train))]];b=np.r_[np.ones(sum(train)),-np.ones(sum(train))]
    lims=[(0,None)]*6;lims[2]=(ac,ac)
    fit=linprog([0,0,0,0,0,1],A_ub=A,b_ub=b,bounds=lims,method='highs');assert fit.success
    co=fit.x[:5]
    tn=tau(beta['table'][0]['Q_keV']*1e-6,me,alpha,1);th=tau(beta['table'][1]['Q_keV']*1e-6,me,alpha,2)*math.log(2)
    ratios=[tn/r['observed_s'] for r in beta['neutron_comparisons']]+[th/beta['table'][1]['observed_half_life_s']]
    C=(min(ratios)+max(ratios))/2
    center=dict(alpha=alpha,r0_fm=r0,av_MeV=float(co[0]),as_MeV=float(co[1]),aa_MeV=float(co[3]),ap_MeV=float(co[4]),C_rate=C)
    lo,hi=0.,.04
    assert bounds(make_box(center,lo))['maximum']<.094 and bounds(make_box(center,hi))['maximum']>.094
    for _ in range(32):
        mid=(lo+hi)/2
        if bounds(make_box(center,mid))['maximum']<.094:lo=mid
        else:hi=mid
    # Inward rounding and deliberate margin, then independently enclosed check.
    width=math.floor(lo*.95*1e5)/1e5;box=make_box(center,width)
    result=bounds(box,strict=True);assert result['maximum']<.10
    bulk=result.pop('bulk')
    with (O/'LIAISONS_BORNEES_R531.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(bulk[0]));writer.writeheader();writer.writerows(bulk)
    # Random spot checks for four sliders; these are not the continuous enclosure proof.
    rng=np.random.default_rng(531);random_max=0.
    for _ in range(128):
        a0=rng.uniform(*box['alpha']);r00=rng.uniform(*box['r0_fm']);c0=rng.uniform(*box['C_rate']);l0=rng.uniform(*lbox)
        errs=spec_error(a0);errs += [math.sqrt(.6)*r00*x['A']**(1/3)/x['rms_fm']-1 for x in rad['radii']]
        t=tau(beta['table'][0]['Q_keV']*1e-6,me,a0,1,l0,C=c0)
        errs += [t/x['observed_s']-1 for x in beta['neutron_comparisons']]
        errs += [tau(beta['table'][1]['Q_keV']*1e-6,me,a0,2,l0,C=c0)*math.log(2)/beta['table'][1]['observed_half_life_s']-1]
        random_max=max(random_max,max(abs(x) for x in errs))
    assert random_max<=result['maximum']
    allbeta=[]
    for row in beta['table']:
        val=tau(row['Q_keV']*1e-6,me,alpha,row['Z_daughter_received'],C=C)*math.log(2)
        allbeta.append(dict(isotope=row['isotope'],half_life_s=val,relative_error=val/row['observed_half_life_s']-1))
    rmin=min(x['half_life_s']/row['observed_half_life_s']*C for x,row in zip(allbeta,beta['table']))
    rmax=max(x['half_life_s']/row['observed_half_life_s']*C for x,row in zip(allbeta,beta['table']))
    assert rmax/1.1>rmin/.9
    nominal={'H-frequencies':max(abs(e) for e in spec_error(alpha)[:5]),'Fine separation H':abs(spec_error(alpha)[-1]),'Cinq rayons':rad['max_radius_relative_error'],'2355 liaisons':float(max(abs(F@co/ys-1))),'Angular correlations':max(abs(r['relative_error']) for r in angular['selected']['comparisons']),'Neutron/tritium':max(abs(r/C-1) for r in ratios)}
    out=dict(revision=531,parent=530,hypothesis='H531_COMMON_EFFECTIVE_BOX; H530-M retained',
        center=center,derived_charge_Qstar=math.sqrt(4*math.pi*alpha)/.3,derived_ac_MeV=ac,
        independent_relative_half_width=width,box=box,derived_Qstar_interval=ends(iv.sqrt(4*iv.pi*interval(*box['alpha']))/v('.3')),
        lambda_not_same_as_H528_barrier=True,canonical_native_G=.3,canonical_native_parameters_modified=False,
        central_maxima=nominal,central_maximum=max(nominal.values()),enclosure=result,
        full_box_below_10_percent=True,spectral_and_binding_enclosure='45-digit outward interval arithmetic; stable rationalized spectral formulas',
        beta_enclosure='Monotone parameter bounds and outward Darboux product sums, 4096 cells; independent quadrature checks.',
        observations='Reported source bounds propagated; not a simultaneous confidence statement. Model discrepancy excluded from source uncertainties.',
        calibration='alpha from H minimax; r0 inherited radius minimax; four bulk coefficients trained on 1422 bindings with derived ac; C from common neutron/tritium compromise.',
        historical_holdout_933_used_in_box_selection=True,prospective_validation=False,
        all_five_beta=allbeta,all_five_beta_C_interval=[rmax/1.1,rmin/.9],
        original_R530_and_old_ppm_results_preserved_as_records=True,new_qualified_rows=[],global_all_68_compatibility=False,
        random_checks=128,random_subset_max_error=random_max,elapsed_s=time.time()-begin,
        remaining=['native derivation of effective operators and coefficients','nuclear overlaps for failed beta channels','global material/optical/gravity/cosmology mapping','biological support/readout and A6 observations'])
    save('INTERSECTION_COMMUNE_R531.json',out)
    # Published aggregated human result: descriptive uncertainty only, no pooled studies.
    n=40;k=19;z=norm.ppf(.975);phat=k/n;den=1+z*z/n;mid=(phat+z*z/(2*n))/den;half=z*math.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/den
    save('LECTURE_EFFECTIF_HYPNOSE_R531.json',dict(source='Baker & Levitt 1989, doi:10.1080/00207148908410543',resistance_count=k,n=n,proportion=phat,wilson95=[mid-half,mid+half],calculation_is_ours=True,raw_data_analyzed=False,scope='Descriptive binomial interval for this selected sample; not a universal resistance probability or proof of coercion in the other participants.'))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8.8,4.8));names=list(nominal);yy=np.arange(len(names));vals=np.array([nominal[n] for n in names])*100;upper=np.array([result['maxima'][n] for n in names])*100
    ax.barh(yy,upper,color='#b4d8d6',label='Terminal on the whole box and margins received');ax.barh(yy,vals,color='#287f84',label='Centre: central observed values')
    for y,x in zip(yy,upper):ax.text(x+.12,y,f'≈ {x:.2f} %',va='center',fontsize=9)
    ax.set_yticks(yy,names);ax.invert_yaxis();ax.axvline(10,color='#a74043',ls='--');ax.set(xlim=(0,11),xlabel='Erreur relative maximale',title='A common interval controlled on the H531 subset')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),fontsize=8,ncol=1,frameon=False);fig.text(.03,.01,'Retrospective calibration. A parameter box does not validate the unconnected domains.',fontsize=8);fig.tight_layout(rect=(0,.15,1,1));fig.savefig(O/'INTERVALLES_COMMUNS_R531.png',dpi=180);fig.savefig(O/'INTERVALLES_COMMUNS_R531.pdf');plt.close(fig)
    print(json.dumps({'center':center,'half_width_percent':width*100,'maxima_percent':{k:v*100 for k,v in result['maxima'].items()},'maximum_percent':result['maximum']*100,'human_wilson95':[mid-half,mid+half],'seconds':time.time()-begin},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
