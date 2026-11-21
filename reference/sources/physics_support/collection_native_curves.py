'Integrate density windows on the received nodal action for a separate native check.'

import argparse
import itertools
from pathlib import Path

from collection_curves import EDGES,OUTPUT,PAIRS,matrices,shape_inputs
from fields_and_references import dump
from dynamics_full import NodalKinetic
from dynamics_nodal_scripts import NodalDynamics
from trials_nodal import digest
from reader_transported import FixedReader
from free_rotations import CALIBRATION,ROOT
from sizes_substrate import nodes,stencils
import numpy as np


NATIVE = {
    'SF4_N48':('R349_free',48),
    'SF6_N48':('SF6_R336_reserve',48),
    'SF4_N64':('R349_free',64),
    'SF6_N64':('SF6_R336_reserve',64),
}


def native_density(field,h,beta):
    n = round(len(field)**(1/3))
    f = field.reshape(n,n,n,4)
    bank = np.zeros((len(EDGES)-1,len(PAIRS)))
    for signs,(d,x) in zip(itertools.product((1,-1),repeat=3),stencils(field,h,nodes(n,h)),strict=True):
        axes = tuple(a for a,s in enumerate(signs) if s<0)
        t = np.flip(f,axis=axes) if axes else f
        t = t[:-1,:-1,:-1].reshape(-1,4)
        rho2 = np.linalg.det(np.concatenate((t[:,None,:],d),axis=1))**2
        ind = np.searchsorted(EDGES[1:-1],rho2,side='right')
        trace = np.sum(d*d,axis=(1,2))
        v = np.empty((len(d),6,4))
        v[:,3:] = -d
        for a,e in enumerate(np.eye(3)):
            v[:,a] = -np.einsum('ni,nic->nc',np.cross(e,x),d)
        b = np.einsum('nac,nic->nai',v,d)
        for j,(a,c) in enumerate(PAIRS):
            w = (h**3/8)*((1.+beta*trace)*np.einsum('ni,ni->n',v[:,a],v[:,c])
                         -beta*np.einsum('ni,ni->n',b[:,a],b[:,c]))
            bank[:,j] += np.bincount(ind,weights=w,minlength=len(bank))
    return bank


def native_fields(model,name):
    tag,n = NATIVE[name]
    fields,g,beta = shape_inputs(model,tag)
    if n==48:
        return fields,g,beta
    g = {**g,'h_native':np.array(g['h_native'])*47/(n-1)}
    return FixedReader(fields,n,3,model).anchor,g,beta


def run(name):
    m = NodalDynamics(48)
    fields,g,beta = native_fields(m,name)
    kernel = NodalKinetic(beta)
    bank = np.array([native_density(a,h,beta) for a,h in zip(fields,g['h_native'],strict=True)])
    ref = np.array([kernel.mass(a,h)[2][3:,3:] for a,h in zip(fields,g['h_native'],strict=True)])
    error = float(np.max(abs(ref-matrices(bank.sum(axis=1)))))
    if error>2e-8:
        raise ArithmeticError('Native curve bank does not reproduce the R343 mass')
    dest = OUTPUT/f'BANQUE_{name}_R352.npz'
    temp = dest.with_suffix('.tmp.npz')
    np.savez_compressed(temp,bins=bank,constant=ref,edges=EDGES)
    temp.replace(dest)
    dump(dest.with_suffix('.json'),{'revision':352,'full':True,'physical_validation':False,'geometry':g,
        'native_case':NATIVE[name],'kernel_baseline_defect':error,'matrix_sha256':digest(dest),
        'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (CALIBRATION,Path(__file__).resolve(),
            ROOT/'scripts'/'banque_courbes_R352.py',ROOT/'sources'/'cinetique_complete_R343.c',
            ROOT/'results'/'INVERSION_PREPARATIONS_R336.json')}})
    print(name,'native bank mass defect',error,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('case',choices=NATIVE)
    run(a.parse_args().case)
