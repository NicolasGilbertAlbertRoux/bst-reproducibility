'Apply the R352 mass constraints to received SF4 and nodally relaxed SF6.'

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sweep_sf4 import read
from collection_curves import EDGES,matrices
from collection_native_curves import native_density,native_fields
from calibrate_curves import M4,X,isotropic
from fields_and_references import dump
from curves_of_distribution import POSITIONS,WIDTHS,shape,solve,laws
from dynamics_nodal_scripts import NodalDynamics
from trials_nodal import digest
from reader_transported import FixedReader
from read_equilibria import NATIVE,read_states
from relax_nodes import OUTPUT,ROOT
from connect_curves_kernel import F4,F6,exact_blocks,fit_native,load_bank,read_rotor
import numpy as np


SOURCE = Path(__file__).resolve()


def read_bank(n,h,beta):
    with ThreadPoolExecutor(max_workers=4) as pool:
        return np.array(list(pool.map(lambda z:native_density(z[0],z[1],beta),zip(n,h,strict=True))))


def bases(b,catalog):
    a = np.array([shape(X,np.ones_like(X),c,w,0) for c,w,_ in catalog])
    return matrices(np.einsum('axt,kx->kat',b,a))


def exact_template(fields,h,beta,terms):
    geometry = {'h_native':h}
    constant = exact_blocks(fields,geometry,beta,[])
    result = []
    for t in terms:
        unit = {**t,'amplitude':1.}
        result.append(exact_blocks(fields,geometry,beta,[unit])-constant)
    return constant,np.array(result)


def main(a):
    path = OUTPUT/f'ETAT_{a.label}_R354.npz'
    with np.load(path) as z:
        n6,h6,r6,d6,o6 = [z[k] for k in ('fields','h_native','radii_native','directions','frames')]
    inherited = read(NATIVE)
    old_terms = inherited['native_terms']
    beta = read(ROOT/'results'/'R349'/'CALAGE_TORSIONS_R349.json')['candidate']['beta']
    scale = inherited['native_scale_per_angstrom']
    b4,u4,g4,bank4 = load_bank(F4)
    b6 = read_bank(n6,h6,beta)
    u6 = matrices(b6.sum(axis=1))
    bank6 = OUTPUT/f'BANQUE_{a.label}_R354.npz'
    np.savez_compressed(bank6,bins=b6,constant=u6,edges=EDGES)
    print('Nodal density bins ready',a.label,'unweighted masses',isotropic(u4).sum(),isotropic(u6).sum(),flush=True)
    catalogs = [('received_supports',[(q['center'],q['width_log'],0) for q in old_terms]),
                ('received_window_catalog',[(c,w,0) for c in POSITIONS for w in WIDTHS]),
                ('broader_density_catalog',[(c,w,0) for c in np.geomspace(1e-19,100.,103)
                                            for w in (.35,.60,1.0,1.8,2.8)])]
    trials = []
    proposal = None
    for name,catalog in catalogs:
        B4,B6 = bases(b4,catalog),bases(b6,catalog)
        v,proof = solve(u4,u6,B4,B6,np.array([1.+.22/w for _,w,_ in catalog]))
        trial = {'catalog':name,'windows':len(catalog),'proof':proof,'feasible_in_binned_iteration':v is not None}
        if v is not None:
            proposal = laws(catalog,v)
            trial['selected_terms'] = proposal
        trials.append(trial)
        print(name,trial,flush=True)
        if proposal is not None:
            break
    outcome = {'revision':354,'full':True,'physical_validation':False,'relaxed_SF6_label':a.label,
        'SF4_still_received_preparation':True,'isotopic_local_shares_are_hypotheses':True,
        'no_rotation_or_vibration_constraint_in_mass_fit':True,'trials':trials,'exact_candidate':None}
    files = [SOURCE,path,NATIVE,bank4,bank6,ROOT/'scripts'/'relier_courbes_noyau_R352.py',
             ROOT/'scripts'/'courbes_de_repartition_R352.py',ROOT/'scripts'/'banque_native_courbes_R352.py']
    if proposal is not None:
        model = NodalDynamics(48)
        n4,geo4,b = native_fields(model,F4)
        if abs(b-beta)>1e-14:
            raise ValueError('The common beta must be used in both cases')
        h4 = np.array(geo4['h_native'])
        constants,extras = {},{}
        for name,n,h in ((F4,n4,h4),(F6,n6,h6)):
            constants[name],extras[name] = exact_template(n,h,beta,proposal)
        defect = max(float(np.max(abs(constants[k]-v))) for k,v in ((F4,u4),(F6,u6)))
        if defect>2e-8:
            raise ArithmeticError('The density bank did not reproduce the unweighted kernel')
        amps,check = fit_native(constants,extras,np.array([p['amplitude'] for p in proposal]))
        terms = [{**w,'amplitude':float(z)} for w,z in zip(proposal,amps,strict=True) if z>1e-10]
        exact = {k:constants[k]+np.tensordot(amps,extras[k],axes=1) for k in (F4,F6)}
        common = M4/isotropic(exact[F4]).sum()
        outcome['exact_candidate'] = {'proof':check,'terms':terms,'one_mass_unit_kg':common,
                                      'bin_to_exact_unweighted_defect':defect,'readings':{}}
        angles4 = np.array(inherited['native_SF4_torsions_rad'])
        O = np.concatenate((np.eye(3)[None],o6))
        results = outcome['exact_candidate']['readings']
        for count in (48,64):
            for tag,n,h,geometry in ((F4,n4,h4,geo4),(F6,n6,h6,None)):
                fields = n if count==48 else FixedReader(n,64,3,None).anchor
                H = h if count==48 else h*47/63
                G = exact[tag] if count==48 else exact_blocks(fields,{'h_native':H},beta,terms)
                if tag==F4:
                    out = read_rotor(G,geometry,scale,common,False,angles4,None)
                else:
                    out = read_states(G,O,d6,d6,r6,scale,common)
                results[f'{tag[:3]}_N{count}'] = out
                print('common mass candidate',tag,count,out,flush=True)
    outcome['input_sha256'] = {str(q.relative_to(ROOT)):digest(q) for q in files}
    dump(OUTPUT/f'INERTIE_RECALEE_{a.label}_R354.json',outcome)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--label',default='POLI_N48')
    main(p.parse_args())
