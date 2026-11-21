'Test common positive kinetic-density weights in the actual S and F profiles.'

from pathlib import Path
import itertools

from sweep_sf4 import read
from sweep_beta import base_fields
from fields_and_references import dump
from dynamics_nodal_scripts import NodalDynamics
from trials_nodal import digest
from inertia_and_entrainment import BANK, M_RATIO, blocks, free_local, spin_free_mass
from reader_transported import FixedReader
from free_rotations import CALIBRATION, ROOT
from sizes_substrate import OUTPUT, SHAPES, nodes, stencils
import numpy as np


def weighted_grams(field,h,beta):
    N = round(len(field)**(1/3))
    f = field.reshape((N,)*3+(4,))
    kernels = ('contrast','contrast_square','topological_density_square')
    out = {k:np.zeros((6,6)) for k in ('constant',*kernels)}
    for sign,(d,x) in zip(itertools.product((1,-1),repeat=3),stencils(field,h,nodes(N,h)),strict=True):
        axes = tuple(a for a,s in enumerate(sign) if s<0)
        z = np.flip(f,axis=axes) if axes else f
        core = z[:-1,:-1,:-1].reshape(-1,4)
        contrast = np.maximum(1-core[:,0],0.)
        cube = np.concatenate((core[:,None,:],d),axis=1)
        determinant = np.linalg.det(cube)
        weight = (np.ones(len(x)),contrast,contrast**2,determinant**2)
        v = np.empty((len(d),6,4))
        v[:,3:] = -d
        for j,e in enumerate(np.eye(3)):
            v[:,j] = -np.einsum('ni,nic->nc',np.cross(e,x),d)
        dot = np.einsum('nac,nic->nai',v,d)
        g = np.sum(d*d,axis=(1,2))
        for k,w in zip(out,weight,strict=True):
            left = (v*np.sqrt(w*(1.+beta*g))[:,None,None]).transpose(1,0,2).reshape(6,-1)
            right = (dot*np.sqrt(w*beta)[:,None,None]).transpose(1,0,2).reshape(6,-1)
            out[k] += (h**3/8)*(left@left.T-right@right.T)
    return out


def main():
    rows = read(BANK)['rows']
    c = read(CALIBRATION)['candidate']
    model = NodalDynamics(48)
    seed = model.unpack(model.project(model.initial())[0])[0]
    n = base_fields(model,c,seed)
    results = []
    for count in (48,64):
        anchor = next(r for r in rows if r['preparation']==SHAPES[0] and r['count']==count)
        fields = n if count==48 else FixedReader(n,count,3,model).anchor
        gS,gF = [weighted_grams(a,h,c['beta']) for a,h in zip(fields[:2],anchor['h_native'][:2],strict=True)]
        expected = sum(blocks(anchor))[:2]
        replay = max(np.max(abs(g['constant']-ref)) for g,ref in zip((gS,gF),expected,strict=True))
        if replay>2e-8:
            raise ArithmeticError('Weighted kinetic quadrature does not replay native G')
        # For a sum of nonnegative kinetic forms, minimization over each spin
        # separately is a lower bound on the minimum of the sum. Holding
        # sulfur spin fixed gives its upper bound.
        records = []
        for key in gS:
            lower = spin_free_mass(gS[key])/float(np.trace(gF[key][3:,3:])/3.)
            upper = float(np.trace(gS[key][3:,3:])/np.trace(free_local(gF[key])))
            records.append({'weight':key,'S_free_inertia_trace_over3':spin_free_mass(gS[key]),
                            'F_free_inertia_trace_over3':spin_free_mass(gF[key]),
                            'lower_mass_ratio':lower,'upper_mass_ratio':upper})
        max_ratio = max(r['upper_mass_ratio'] for r in records)
        results.append({'count':count,'mass_block_replay_inf_defect':float(replay),'weights':records,
                        'upper_ratio_for_any_nonnegative_mixture_of_tested_weights':max_ratio,
                        'can_exclude_full_mixture':max_ratio<M_RATIO})
        print('positive density mixtures',count,[(r['weight'],r['upper_mass_ratio']) for r in records],flush=True)
    dump(OUTPUT/'POIDS_DE_DENSITE_R351.json',{'revision':351,'full':True,'rows':results,
         'measured_isotope_ratio':M_RATIO,'physical_validation':False,
         'scope':'local common kinetic weights 1, (1-n0), (1-n0)^2 and squared unnormalised topological density; current S and F1 anchors',
         'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (BANK,CALIBRATION,Path(__file__).resolve())}})


if __name__=='__main__':
    main()
