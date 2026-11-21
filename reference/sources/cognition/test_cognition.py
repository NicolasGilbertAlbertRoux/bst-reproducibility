'Crafter, uncertainty of published result; Horizon, predictive comparison.'
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import norm, t

ROOT = Path(__file__).resolve().parents[1]


def fit_logistic(design, response):
    beta = np.zeros(design.shape[1])
    for _ in range(60):
        predicted = expit(design @ beta)
        score = design.T @ (response - predicted)
        if np.max(np.abs(score)) < 1e-7:
            return beta
        information = design.T @ ((predicted * (1-predicted))[:,None] * design)
        beta += np.linalg.solve(information,score)
    raise ArithmeticError('Non-convergized regression')


def evaluate(records, human):
    unit = np.array([int(r['unit_anonyme']) for r in records])
    horizon = np.array([int(r["horizon_long"]) for r in records])
    delta = np.array([float(r["difference_gain_vu"]) for r in records]) / 10
    chosen = np.array([int(r['choice_minus_vu']) for r in records])
    designs = [np.column_stack((np.ones(len(delta)), delta, delta*horizon)),
               np.column_stack((np.ones(len(delta)), delta, delta*horizon, horizon))]
    # In both models the horizon may already change sensitivity to gains.
    unique = np.unique(unit)
    if human:
        folds = unit
    else:
        # GPT-3 has independent task files with no identifying participant.
        partitions = {int(task):i%10 for i,task in enumerate(np.random.default_rng(558).permutation(unique))}
        folds = np.array([partitions[int(task)] for task in unit])
    preds = np.full((2,len(chosen)), np.nan)
    for fold in np.unique(folds):
        test = folds == fold
        for j, design in enumerate(designs):
            beta=fit_logistic(design[~test],chosen[~test])
            preds[j,test] = design[test] @ beta
    assert np.isfinite(preds).all()
    neglog = np.logaddexp(0,preds)-chosen*preds
    brier = (expit(preds)-chosen)**2
    gain = neglog[0]-neglog[1]
    mean_gains = [float(gain[unit == who].mean()) for who in unique] if human else []
    return {"parties_3_1":len(chosen),'units':len(unique),"plis":len(np.unique(folds)),
            'perte_log_A_B_nats_par_choice':neglog.mean(axis=1).tolist(),
            "gain_A_moins_B":float(gain.mean()),
            "Brier_A_B":brier.mean(axis=1).tolist(),
            "personnes_gain_strictement_positif":int(sum(g>0 for g in mean_gains)) if human else None,
            'mean_ponderee_egalement_par_personne':float(np.mean(mean_gains)) if human else None}


def main():
    n, r = 10, .80
    t_value = r*math.sqrt((n-2)/(1-r*r))
    z = math.atanh(r)
    margin = norm.ppf(.975)/math.sqrt(n-3)
    p = float(2*t.sf(abs(t_value), n-2))
    corr = {'n_enfants_analysis_of_the_parole':n,'r_published_rounded':r,'p_bilateral_from_r_rounded':p,
            'p_published_non_fitted':.005,'IC95_approche_Fisher_depuis_r_rounded':
            [math.tanh(z-margin),math.tanh(z+margin)],
            'unit':"enfant","t":t_value,"ddl":n-2,
            'analysis':'association between two summaries of the session',
            'reanalysee_from':'published statistics, not individual Crafter newspapers'}
    assert abs(p-.005)<.001
    with (ROOT/"entrees/choix_horizon.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    groups = {name:evaluate([r for r in rows if r['group']==name],name=="Humains")
              for name in ["Humains","GPT-3"]}
    assert groups["Humains"]['units'] == 99
    output={"Crafter":corr,"Horizon":groups,
            "A":'Preferably constant, difference in gain effectively seen and interaction of this deviation with the horizon.',
            "B":'Same model with an additional horizon coefficient for choosing the less observed option.',
            'scope':'Retrospective prediction scores. The groups are adjusted separately. '
            'Every human is completely discarded when it is predicted. GPT-3 is a single historical model; '
            'The ten folds discard parts, never models. Both strategies are calculable. '
            'No data of experience or prior purpose in Horizon. The sense of causation in Crafter is not measured.'}
    (ROOT/"resultats/COGNITION.json").write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(output,indent=2,ensure_ascii=False))
if __name__ == "__main__":
    main()
