'Extension H528 and control with eight pairs of phases. Dependencies of the BST file.'
import argparse
import csv
import importlib.util
import itertools
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np

ICI = Path(__file__).resolve().parent
COGNITION = ICI.parent
DIM_MOT = 8
EPSILON = .12


def write_table(path, names, values):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(names)
        w.writerows(values)


def json_write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def calculation_h528(output, reference=None):
    # The historic programme creates its results file on import.
    # Its temporary copy leaves the distributed file unchanged.
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp)/'R528'
        shutil.copytree(COGNITION/'sources/R528', source)
        spec = importlib.util.spec_from_file_location('h528_historique', source/'tester_persistance_R528.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with (COGNITION/'attendus/BALAYAGE_COGNITIF_R528.csv').open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        cases = [module.make_case(r['tag'],float(r['eta']),float(r['lam']),float(r['mass']),
                                  float(r['impulse']),int(r['sign'])) for r in rows]
        assert len(cases)==144
        rec, drift, minimum = module.integrate(cases,dt=.002,end=24.,sample=.01)
        if reference:
            with np.load(reference) as hist:
                assert list(hist['columns'])==['t','x','q','p_x','p_q','H','Q_loss']
                court = hist['records']
        else:
            court = module.integrate(cases,dt=.002,end=8.,sample=.01)[0]
        prefix = float(np.max(np.abs(court-rec[:801])))
        assert court.shape==(801,144,7) and prefix<1e-11
        time = rec[:,0,0]
        pulse = np.array([c['impulse'] for c in cases])
        signs = np.array([c['sign'] for c in cases])
        q = rec[:,:,2]
        xj = rec[:,:,1]/pulse
        magnitude = q*signs
        assert np.max(drift)<1e-4 and np.min(minimum)>0
        lam = np.array([c['lam'] for c in cases])
        energy = pulse**2/2
        radial = np.sqrt(4*energy/lam)
        low, high = np.sqrt(1-radial), np.sqrt(1+radial)
        assert np.all(magnitude>=low[None,:]-5e-6) and np.all(magnitude<=high[None,:]+5e-6)
        pair = [i for i,c in enumerate(cases) if c['tag']=='m1' and c['eta']==.5 and c['lam']==.5
                 and c['mass']==.5 and c['impulse']==.2]
        assert list(signs[pair])==[-1,1]
        published = np.loadtxt(COGNITION/'figures/Paire_R528.csv',delimiter=',',skiprows=1)
        current = np.column_stack((time,q[:,pair],xj[:,pair]))
        deviation_published = float(np.max(abs(current[:801]-published)))
        assert deviation_published<1e-11
        checks = []
        for i,c in enumerate(cases):
            if c['eta']!=.8 or c['lam']!=.5 or c['mass']!=.5 or c['impulse']!=.2:
                continue
            t, y, ed, neval = module.independent(c,end=24.)
            assert np.allclose(time,t,atol=1e-12,rtol=0)
            err_x = float(np.max(abs(rec[:,i,1]-y[0]))/max(abs(y[0])))
            err_q = float(np.max(abs(rec[:,i,2]-y[1])))
            assert err_x<.002 and err_q<.00015
            checks.append(dict(bath=c['tag'],sign_field=c['sign'],deviation_x_relative=err_x,
                                  deviation_q_absolute=err_q,summary_DOP853=float(ed),evaluations=int(neval)))
        assert len(checks)==4
        records0, _, _ = module.integrate([module.make_case(eta=0.,sign=s) for s in [-1,1]],end=24.)
        blind=float(np.max(abs(records0[:,0,1]-records0[:,1,1])))
        assert blind==0.
    np.savez_compressed(output/'Trajectoires_H528_prolongees.npz',time=time,q=q,x_over_J=xj,
                        sign_field=signs,pair_illustree=np.array(pair))
    columns=['time'];values=[time]
    for sign_field,name in [(-1,'minus'),(1,'plus')]:
        mask=signs==sign_field
        for label, mat in [('q',q),('x_sur_J',xj),('module_q',magnitude)]:
            columns.extend([f'{label}_{name}_min',f'{label}_{name}_max'])
            values.extend([mat[:,mask].min(axis=1),mat[:,mask].max(axis=1)])
    write_table(output/'Intervalles_H528.csv',columns,zip(*values))
    write_table(output/'Preparations_H528.csv',['index','bath','eta','lambda','mass','J','sign','bound_inf_module',
                                                'bound_sup_module','drift_energy_max','min_sign_fois_q'],
                 ([i,c['tag'],c['eta'],c['lam'],c['mass'],c['impulse'],c['sign'],low[i],high[i],drift[i],minimum[i]]
                  for i,c in enumerate(cases)))
    return dict(preparations=len(cases),par_sign=72,samples=len(time),step=.002,t_fin=24,
                t_fin_historique=8,deviation_prefix_144_trajectoires=prefix,deviation_pair_published=deviation_published,
                energy_max=float(max(energy)),barrier_min=float(min(lam)/4),
                bounds_analytic_global=[float(min(low)),float(max(high))],
                drift_energy_relative_max=float(max(drift)),min_module_integration=float(min(minimum)),
                max_module_sampled=float(magnitude.max()),output_blind_eta_zero=blind,validation_DOP853=checks)


def progression(u):
    u = np.clip(u,0.,1.)
    return u*u*(3.-2.*u)


def order_par_pairs(time, duration):
    s = progression(time/duration)
    angles = time[:,None]+2*np.pi*s[:,None]*np.arange(DIM_MOT)/DIM_MOT
    return np.abs(np.exp(1j*angles).mean(axis=1))


def arbitrate(mot, target):
    distance = np.count_nonzero(mot!=target)
    if distance < DIM_MOT/2:
        return 'A'
    if distance > DIM_MOT/2:
        return 'B'
    return 'suspension'


def calculation_pairs(output):
    t=np.linspace(0.,9.,901)
    durations=np.arange(3.,5.001,.25)
    orders=np.array([order_par_pairs(t,T) for T in durations])
    assert np.max(orders[:,t>=5.])<3e-15
    for v in orders:
        assert np.max(np.diff(v))<3e-15
    central=order_par_pairs(t,4.)
    write_table(output/'Ordre_collectif.csv',['t','min_T_sampled','T_4','max_T_sampled'],
                 zip(t,orders.min(axis=0),central,orders.max(axis=0)))
    threshold=math.sqrt(2)*EPSILON
    r=np.unique(np.r_[np.linspace(0.,1.,501),threshold])
    p_inf,p_sup=r/math.sqrt(2)-EPSILON,r+EPSILON
    write_table(output/'Lecteur_et_resolution.csv',['sensibilite','sign_minus_min','sign_minus_max','sign_plus_min','sign_plus_max'],
                 zip(r,-p_sup,-p_inf,p_inf,p_sup))
    assert abs(float((threshold/math.sqrt(2)-EPSILON)))<1e-15
    # Check the two readings on the same 16 complex coordinates.
    words=list(itertools.product([0,1],repeat=DIM_MOT))
    target=np.array([1,0,1,1,0,1,0,0]); alts=np.array(words)
    theta=7.+2*np.pi*np.arange(DIM_MOT)/DIM_MOT
    def read(delta):
        a=np.exp(.5j*(theta+delta)); b=np.exp(.5j*(theta-delta))
        order=np.abs(np.mean(a*b,axis=1)); channel=(a*b.conj()).real
        return order,(channel<0).astype(int)
    d1,retained=read(np.pi*alts)
    d2,erased=read(np.zeros_like(alts))
    assert np.all(retained==alts) and np.all(erased==0)
    max_d=float(max(d1.max(),d2.max()));assert max_d<3e-15
    issus={k:sum(arbitrate(row,target)==k for row in retained) for k in ['A','B','suspension']}
    assert issus=={'A':93,'B':93,'suspension':70} and arbitrate(erased[0],target)=='suspension'
    write_table(output/'Mots_et_reponses.csv',['mot_initial','readout_protegee','decision_protegee','readout_remise_a_zero','decision_after_effacement'],
                 ((''.join(map(str,m)),''.join(map(str,c)),arbitrate(c,target),''.join(map(str,v)),arbitrate(v,target)) for m,c,v in zip(words,retained,erased)))
    # The same overall movement accompanies two continuous dynamics of the phase shift.
    # The reset acts from t 4 to t 5 on a phase initially at an equilibrium.
    row_time = []
    for instant in np.linspace(0.,8.,161):
        theta_t = instant+2*np.pi*np.arange(DIM_MOT)*progression(instant/4.)/DIM_MOT
        phases_i = np.pi*alts
        phases_f = (1-progression(instant-4.))*phases_i
        def channels(d):
            start=np.exp(.5j*(theta_t+d))
            fin=np.exp(.5j*(theta_t-d))
            return start*fin, (start*fin.conj()).real
        u_i, v_i = channels(phases_i)
        u_f, v_f = channels(phases_f)
        deviation=float(np.max(np.abs(u_i-u_f)))
        assert deviation<3e-15
        if instant<=4.:
            assert np.array_equal(v_i,v_f)
        if instant>=5.:
            assert np.max(np.abs(v_f-1.))<2e-15
        row_time.append([instant,float(abs(u_i.mean(axis=1)).max()),float(abs(u_f.mean(axis=1)).max()),deviation])
    write_table(output/'Deux_dynamiques_meme_lecture.csv',
                 ['time','order_max_branch_protegee','order_max_branch_effacee','deviation_max_of_huit_produits'],row_time)
    # Checks at the edges of each sector invariant for the worst-case disturbance.
    kappa=1.; nu=.6; margin=kappa-nu
    for center in [0.,np.pi]:
        assert np.isclose(-kappa*np.sin(2*(center-np.pi/4))-nu,margin)
        assert np.isclose(-kappa*np.sin(2*(center+np.pi/4))+nu,-margin)
    return dict(type='Built geometric tell-tale, reduced units, no biological measures.',
                pairs=8,words_tested=len(words),words_protected=len(np.unique(retained,axis=0)),
                words_after_remise_a_zero=len(np.unique(erased,axis=0)),order_final_max=max_d,deviation_between_branches_max=float(np.max(np.array(row_time)[:,3])),
                durations_illustrees=durations.tolist(),epsilon_illustre=EPSILON,threshold_of_readout=threshold,
                kappa_illustration=kappa,perturbation_bounded_nu=nu,marge_invariante=margin,decisions=issus,
                mot_under_effacement='00000000',but_A='10110100',but_B='01001011')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--reference',type=Path,help='NP- of the historic replay R528. Otherwise, the prefix is recalculated.')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    h=calculation_h528(a.output,a.reference); phases=calculation_pairs(a.output)
    json_write(a.output/'Bilan_prolongement.json',{'H528':h,'pairs_of_phases':phases})
    print(f'{h["preparations"]} trajectoires, {h['samples']} instants, {phases['words_protected']} mots vérifiés.')


if __name__=='__main__':
    main()
