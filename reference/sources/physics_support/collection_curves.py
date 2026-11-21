'Integrate native kinetic blocks against the two local substrate variables.'

import argparse
import json
from pathlib import Path
from time import perf_counter

from sweep_sf4 import read
from sweep_beta import base_fields
from fields_and_references import dump
from dynamics_nodal_scripts import NodalDynamics
from trials_nodal import digest
from clock_inertia import OUTPUT as R350
from inertia_and_entrainment import BANK as R351_BANK
from quadrature_topological import along,samples
from refit_dependencies import State
from free_rotations import CALIBRATION,ROOT
import numpy as np


OUTPUT = ROOT/'results'/'R352'
EDGES = np.r_[0.,np.geomspace(1e-12,10000.,385),np.inf]
CHI_EDGES = np.linspace(0.,1.,21)
CASE = {
    'SF4_R349_C2':('R349_free',2,3),
    'SF6_R336_C2':('SF6_R336_reserve',2,3),
    'SF4_R349_C3':('R349_free',3,3),
    'SF6_R336_C3':('SF6_R336_reserve',3,3),
    'SF4_R349_L2':('R349_free',2,1),
    'SF4_R350_C2':('R350_precontraint',2,3),
    'SF6_R321_C2':('SF6_R321_reserve',2,3),
}
PAIRS = tuple(zip(*np.triu_indices(6),strict=True))


def integrand(field,h,scheme):
    points,weights,mat,grad,_ = scheme
    coords = h*(points-23.5)
    raw = field.reshape(48,48,48,4)
    z = along(raw,mat,2)
    yz = along(z,mat,1)
    dy = along(z,grad,1)
    dz = along(along(raw,grad,2),mat,1)
    for k in range(0,len(points),7):
        sl = slice(k,k+7)
        value = along(yz,mat[sl],0).reshape(-1,4)
        norm = np.linalg.norm(value,axis=1)
        if min(norm)<.2:
            raise ValueError('Interpolated fields leave the normalized chart')
        n = value/norm[:,None]
        d = np.stack([p.reshape(-1,4) for p in (
            along(yz,grad[sl],0),along(dy,mat[sl],0),along(dz,mat[sl],0))],axis=1)
        d -= n[:,None,:]*np.einsum('ni,nai->na',n,d)[:,:,None]
        d /= (h*norm)[:,None,None]
        pos = np.stack(np.meshgrid(coords[sl],coords,coords,indexing='ij'),axis=-1).reshape(-1,3)
        volume = (h**3*(weights[sl,None,None]*weights[None,:,None]*weights[None,None,:])).ravel()
        rho2 = np.linalg.det(np.concatenate((n[:,None,:],d),axis=1))**2
        trace = np.sum(d*d,axis=(1,2))
        chi = np.divide(27*rho2,trace**3,out=np.zeros_like(trace),where=trace>1e-24)
        if max(chi)>1.+1e-10:
            raise ArithmeticError('The normalized determinant violated the arithmetic-mean bound')
        v = np.empty((len(d),6,4))
        v[:,3:] = -d
        for a,e in enumerate(np.eye(3)):
            v[:,a] = -np.einsum('ni,nic->nc',np.cross(e,pos),d)
        b = np.einsum('nac,nic->nai',v,d)
        yield rho2,np.clip(chi,0.,1.),v,b,trace,volume,float(min(norm))


def matrices(a):
    M = np.zeros(a.shape[:-1]+(6,6))
    for k,(i,j) in enumerate(PAIRS):
        M[...,i,j]=M[...,j,i]=a[...,k]
    return M


def binned(field,h,beta,scheme):
    nx,ny = len(EDGES)-1,len(CHI_EDGES)-1
    B = np.zeros((nx,ny,len(PAIRS)))
    weighted = np.zeros(len(PAIRS))
    xmax,norm = 0.,float('inf')
    for x,y,v,b,tr,w,u in integrand(field,h,scheme):
        xmax = max(xmax,float(max(x)))
        norm = min(norm,u)
        xb = np.searchsorted(EDGES[1:-1],x,side='right')
        yb = np.searchsorted(CHI_EDGES[1:-1],y,side='right')
        index = xb*ny+yb
        for k,(i,j) in enumerate(PAIRS):
            values = w*((1.+beta*tr)*np.einsum('ni,ni->n',v[:,i],v[:,j])-
                         beta*np.einsum('ni,ni->n',b[:,i],b[:,j]))
            B[:,:,k] += np.bincount(index,weights=values,minlength=nx*ny).reshape(nx,ny)
            weighted[k] += x@values
    return B,weighted,{'max_rho_squared':xmax,'minimum_interpolant_norm':norm}


def shape_inputs(model,tag):
    source = next((r for r in read(R351_BANK)['rows'] if r['preparation']==tag and r['count']==48),None)
    candidate = read(CALIBRATION)['candidate']
    if tag.startswith('SF6'):
        state6 = State('SF6')
        if tag=='SF6_R321_reserve':
            sf6 = next(r for r in read(R351_BANK)['rows'] if r['species']=='SF6')
            with np.load(ROOT/'sources'/'R321'/'inputs'/'SF6_ENTREES_NATIVES_R321.npz') as archive:
                n = archive['fields']
            source = {**sf6,'preparation':tag}
        else:
            n = np.array(state6.fields)
        n = n/np.linalg.norm(n,axis=2)[...,None]
    else:
        c = candidate if tag=='R349_free' else read(R350/'CONTACT_PRECONTRAINT_R350.json')['prestressed_seed']['candidate']
        initial = model.unpack(model.project(model.initial())[0])[0]
        n = base_fields(model,c,initial)
    geometry = {k:source[k] for k in ('centers','frames','h_native')}
    geometry['preparation'] = tag
    geometry['source_length_scale'] = source.get('length_scale_native_per_angstrom')
    return n,geometry,candidate['beta']


def run(name):
    tag,gauss,order = CASE[name]
    model = NodalDynamics(48)
    fields,geometry,beta = shape_inputs(model,tag)
    scheme = samples(gauss,order)
    start = perf_counter()
    bins,exact,checks = [],[],[]
    for j,(field,h) in enumerate(zip(fields,geometry['h_native'],strict=True)):
        B,G,test = binned(field,h,beta,scheme)
        bins.append(B)
        exact.append(G)
        checks.append(test)
        print(name,'site',j,'max x',test['max_rho_squared'],'seconds',round(perf_counter()-start,1),flush=True)
    B = np.array(bins)
    summed = matrices(B.sum(axis=(1,2)))
    weights = matrices(np.array(exact))
    if min(np.linalg.eigvalsh(summed).ravel())<=0 or min(np.linalg.eigvalsh(weights).ravel())<=0:
        raise ArithmeticError('The integrated mass is not positive')
    if name in ('SF4_R349_C2','SF4_R349_C3','SF4_R349_L2'):
        p = read(ROOT/'results'/'R351'/'QUADRATURE_DENSITE_R351.json')
        key = {'SF4_R349_C2':'cubic_Gauss2','SF4_R349_C3':'cubic_Gauss3','SF4_R349_L2':'linear_Gauss2'}[name]
        defect = float(np.max(abs(summed+p['lambda_fitted_only_on_cubic_Gauss2_by_isotope_mass']*weights-
                                  np.array(p['results'][key]['native_metric']))))
        if defect>2e-8:
            raise ArithmeticError(f'R351 kinetic quadrature did not replay: {defect}')
    else:
        defect = None
    dest = OUTPUT/f'BANQUE_{name}_R352.npz'
    temp = dest.with_suffix('.tmp.npz')
    np.savez_compressed(temp,bins=B,linear_x_matrices=weights,constant_matrices=summed,edges=EDGES,chi_edges=CHI_EDGES)
    temp.replace(dest)
    dependencies = [R351_BANK,CALIBRATION,ROOT/'scripts'/'quadrature_topologique_R351.py',Path(__file__).resolve()]
    if tag=='R350_precontraint':
        dependencies.append(R350/'CONTACT_PRECONTRAINT_R350.json')
    elif tag.startswith('SF6'):
        dependencies += [ROOT/'sources'/'R321'/'inputs'/'SF6_ENTREES_NATIVES_R321.npz',
                         ROOT/'results'/'INVERSION_PREPARATIONS_R336.json']
    dump(dest.with_suffix('.json'),{'revision':352,'full':True,'physical_validation':False,'name':name,
         'case':CASE[name],'geometry':geometry,'beta':beta,'checks':checks,'spline_derivative_defect':scheme[-1],
         'R351_matrix_replay_inf_defect':defect,'matrix_sha256':digest(dest),
         'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in dependencies}})


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case',choices=CASE)
    run(parser.parse_args().case)
