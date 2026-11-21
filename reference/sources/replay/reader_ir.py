'Calibrate the same reading vector of the six contacts on the two regions of the PTB.'
import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.linalg import eigh

ROOT=Path(__file__).resolve().parents[1]
I=ROOT/'inputs'
O=ROOT/'results'
WINDOWS={'nu3':(800.,1075.),'nu4':(575.,675.)}
CORE={'nu3':(900.,980.),'nu4':(595.,635.)}
def decode(name):
    return json.loads((I/name).read_text())
def save(name,body):
    (O/name).write_text(json.dumps(body,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def observed():
    raw={}
    with zipfile.ZipFile(I/'PTB_RAW.zip') as z:
        for p in ('225.5','745.7'):
            raw[p]={}
            for band,(left,right) in WINDOWS.items():
                name=next(n for n in z.namelist() if p+'Torr' in n and (('_800.' in n) if band=='nu3' else ('_570.' in n or '_575.' in n)))
                x,y,u=np.loadtxt(io.BytesIO(z.read(name)),skiprows=2,unpack=True)
                def moments(edges):
                    lo=max(edges[0],x[0]); hi=min(edges[1],x[-1])
                    grid=np.r_[lo,x[(x>lo)&(x<hi)],hi]
                    vals=np.interp(grid,x,y)*1e19
                    area=np.trapezoid(vals,grid)
                    return dict(bounds=[lo,hi],count_points=len(grid),area=area,center=np.trapezoid(grid*vals,grid)/area)
                raw[p][band]=dict(source=name,region=moments((left,right)),core=moments(CORE[band]))
    return raw
def operators():
    state=decode('ETAT_SF6_PARENT_R480.json')
    c=state['chart']; directions=np.array(state['directions']);r=np.array(state['radii'])
    bars=np.array(c['contact_frame']);y=np.array(state['coordinates'])
    d0=np.array(c['contact_base'])+np.einsum('ia,iaj->ij',y[48:60].reshape(6,2),bars)
    length=np.linalg.norm(d0,axis=1)
    assert np.max(abs(d0/length[:,None]-directions))<1e-9
    of=(bars-directions[:,None,:]*np.einsum('ij,iaj->ia',directions,bars)[:,:,None])/length[:,None,None]
    rref=float(np.mean(r))  # definition of the contact reference level
    # p(r)--------------------------------------------------------------------------------------------------------
    da=np.zeros((3,66));db=np.zeros_like(da)
    da[:,60:66]=directions.T
    da[:,48:60]=((r-rref)[:,None,None]*of).reshape(12,3).T
    db[:,48:60]=(rref*of).reshape(12,3).T
    with np.load(I/'OPERATEURS_SF6_PARENT_R480.npz') as z:
        v=z['modes']
        assert np.max(abs(v.T@z['M']@v-np.eye(66)))<1e-8
        op=np.stack((da@v,db@v))
        freq=np.sqrt(z['omega_squared'])
    def P(coordinates,a,b):
        raw=np.array(c['contact_base'])+np.einsum('ia,iaj->ij',coordinates[48:60].reshape(6,2),bars)
        return np.sum((a*(coordinates[60:66]-rref)+b*rref)[:,None]*raw/np.linalg.norm(raw,axis=1,keepdims=True),axis=0)
    assert np.max(abs(y[60:66]-r))<1e-8
    probe=np.cos(np.arange(66));eps=2.e-5
    err=np.max(abs((P(y+eps*probe,.71,-.27)-P(y-eps*probe,.71,-.27))/(2*eps)-(.71*da-.27*db)@probe))
    assert err<1e-7
    return freq,op,err
def setup(raw):
    # Compromise minimax for two pressures on the same area; no statistical average.
    ref={}
    for band in WINDOWS:
        vals=[g[band]['region']['area'] for g in raw.values()]
        ref[band]=2/(1/min(vals)+1/max(vals))
    return ref
def fit(gain,w,B,raw):
    x=gain*w
    masks={k:(x>=lo)&(x<=hi) for k,(lo,hi) in WINDOWS.items()}
    matrices={b:np.einsum('aim,bim->ab',B[:,:,m],B[:,:,m]) for b,m in masks.items()}
    ref=setup(raw); ratio=ref['nu3']/ref['nu4']
    limits,axes=eigh(matrices['nu3'],matrices['nu4'])
    desired=float(np.clip(ratio,*limits)); v=(desired-limits[0])/(limits[1]-limits[0])
    fits=[]
    for sign in (-1,1):
        h=axes@np.array([np.sqrt(max(0,1-v)),sign*np.sqrt(max(0,v))])
        s0=np.sum(np.einsum('a,aim->im',h,B)**2,axis=0)
        relative=np.array([s0[masks[k]].sum()/val[k]['region']['area'] for val in raw.values() for k in WINDOWS])
        scale=2/(min(relative)+max(relative));s=scale*s0
        metrics={}
        for band,m in masks.items():
            mc=(x>=CORE[band][0])&(x<=CORE[band][1])
            metrics[band]=dict(count_modes=int(m.sum()),area=float(s[m].sum()),center=float(s[m]@x[m]/sum(s[m])),fraction_core=float(sum(s[mc])/sum(s[m])))
        residuals=[abs(metrics[k]['area']/val[k]['region']['area']-1) for val in raw.values() for k in WINDOWS]
        centroids=[abs(metrics[k]['center']/val[k]['region']['center']-1) for val in raw.values() for k in WINDOWS]
        core_res=[abs(metrics[k]['fraction_core']/(val[k]['core']['area']/val[k]['region']['area'])-1) for val in raw.values() for k in WINDOWS]
        fits.append(dict(a_b_standards=(np.sqrt(scale)*h).tolist(),bands=metrics,deviation_max_4_aires_pct=100*max(residuals),deviation_max_4_centers_pct=100*max(centroids),deviation_max_4_fractions_core_pct=100*max(core_res),intensites=s.tolist()))
    return dict(gain=gain,ratio_measurement_minimax=ratio,ratio_accessible=limits.tolist(),frequencies=x.tolist(),solutions=fits)
def main():
    raw=observed(); w,B,fd=operators()
    old=decode('CONTROLE_SYMETRIES_R572.json')
    keys={'moderne':old['results'][0],'moderne_with_1995':old['moderne_with_comparateurs_1995_branch_4']}
    fits={k:fit(v['gain_control'],w,B,raw) for k,v in keys.items()}
    out=dict(measurements=raw,unit_aires='10^-19 cm/molecule',error_DF_readout=fd,candidates=fits,model='Effective dipole P plus sumi (a) (r-i-ref)-b ref-- i, with the mean geometric radius of the six contacts at equilibrium. The harmonic integrated weight is proportional to the square norm of its derivative along the mass-normalized mode. Both coefficients include the reading scale. All modes, even outside the R572 allocation, contribute according to their frequency. Model of sticks for an area and its barycenter. The rotational structure and the pressure profiles are not resolved. The inner windows are descriptive; the point uncertainties of the PTB are not quadrature.')
    save('AIRES_IR_ET_LECTEUR_R573.json',out)
    for k,v in fits.items(): print(k,v['ratio_accessible'],[(s['deviation_max_4_aires_pct'],s['deviation_max_4_centers_pct'],s['deviation_max_4_fractions_core_pct']) for s in v['solutions']])
if __name__=='__main__': main()
