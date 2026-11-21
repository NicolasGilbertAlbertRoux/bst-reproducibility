'Register status and identifiable numerical coordinates of the 576 assemblies.'
import csv
import json
from pathlib import Path
from readout_ir import I,O,ROOT,decode,save

def read(n):return json.loads((O/(n+'_R573.json')).read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
    parent=decode('DECISION_R572.json')
    d=read('DECISION_GLOBAL')
    audit={r['row']:r for r in csv.DictReader((ROOT/'AUDIT_DES_68_LIGNES_R573.csv').open(encoding='utf-8-sig'))}
    base=decode('REGISTRE_68_LIGNES_R572.json');joint=read('COMBINATIONS_68')
    for r in base:
        line=r['row'];a=audit[line]
        r['revue_commune_R573']={k:a[k] for k in ['finding','under_condition_scientific','combinations_computed','established_parameters_retained','independants_of_setting','correspondence_indetermine','combinations_in_failure','records']}
        if line=='D04.4':r['conclusion_R573']='Net energy on two sides with the seven mobile centres, digital propagation correction, absorbent impedance and native time from 0 to 0.36. Question of sensitivity of the resolved flow in this preparation.'
        if line=='J00.2':r['conclusion_R573']='Validated in its documentary scope. Decision rendered, inherited frozen predictions, manuscripts, curves and deposition.'
        if line=='J00.1':
            covered=d['combinations_covering_67_checks_explicit']
            if covered:
                r['etat_before_R573']=r['state']
                r['state']='VERIFIED_NUMERICAL_CONDITIONNEL'
                r['established_of_scope']=True
                r['last_revision_of_clause']=573
                r['last_detail_of_clause']=dict(revision=573,state=r['state'],scope='Simultaneous compatibility of the calibrated bench covered by the remaining 67 lines, to the calculation or documentation spreadsheets indicated in the matrix.',conditions_common=['Maintain the sectoral assumptions and preparations of each item.','Maintain retrospective readers and calibrations, including the common gain in viscosity.','The full scientific scope of the synthesis contract retains its undocumented sub-conditions.'],evidence=['resultats/ASSEMBLAGES_68_R573.json','PARAMETRES_DES_FAMILLES_R573.json','resultats/CONDITIONS_DE_PORTEE_R573.json'],count_of_sets=covered)
            r['state_of_contract_global_scientific']='OPEN'
            r['conclusion_R573']=f'{covered} des 576 assemblages couvrent les 67 contrôles concrets. J00.1 est validée sous conditions dans le périmètre de ce banc. Aucun témoin intégral du contrat physique original n’est établi.'
            r['compatibilite_des_observables_etalonnees_R573']=dict(sets_with_67_comparisons_or_established=d['combinations_covering_67_checks_explicit'],examples=d['initial_with_67_checks'],under_conditions_a_interpreter=d['claims_with_comparison_partial'])

    write(ROOT/'REGISTRE_68_LIGNES_R573.json',base)
    gas=decode('AUDIT_COMBINAISONS_R570.json')['new_correspondence_gas']
    cosm=decode('COSMO_R479/entrees/CANDIDATS_COSMOLOGIQUES_R479.json');ca=read('COSMOLOGIE_ET_OSCILLATIONS')['families']
    spec=read('SPECTRUM_ET_THERMOCHIMIE')['families'];rr=read('REPONSE_COMMUNE_TROIS_BANDS')['families'];ac=read('UNITS_ET_RUPTURE')['actions_SFx']
    cl=read('AJUSTEMENT_ENERGIE_FREE')['branches'];alpha=read('RECALCUL_CHARGE_COMMUNE')
    write(ROOT/'PARAMETRES_DES_FAMILLES_R573.json',dict(
      combinations='Les six coordonnées de chaque identifiant T001–T576 figurent avec les résultats par ligne dans resultats/ASSEMBLAGES_68_R573.json.',
      kernel=dict(KAPPA=2,INERTIA=8),cosmology={v['family']:{**{k:cosm[i][k] for k in ('fraction_flux','sigma0','omega_m','h')},'masse_eV':v['masse_eV'],'omega_cdm_h2':v['omega_cdm_h2']} for i,v in enumerate(ca)},
      gas={'radii':gas['radii_recalcules'],'length_unitaire_m':gas['L_m'],'exponent':gas['b'],'branch':gas['combinaisons'],'corrections_de_B2_cm3_mol':{k:v[0]['coefficients'] for k,v in cl.items()}},
      fluorides={k:{**v,'coefficients_of_readout_lie':spec[k]['best_par_n_of_coefficients']['4']['coefficients'],'zero_cm1':rr[k]['zero_cm1'],'amortissement':rr[k]['zeta_retained']} for k,v in ac.items()},
      port_and_metric={k:{i:v[i] for i in ('alpha','G','Q_si_S0_sur_hbar_1','S0_sur_hbar_si_Q_1','metrics')} for k,v in alpha.items()},
      diffusion=read('INTERVALS_DIFFUSION')['families'],shear=read('SUMMARY_SHEAR'),
      others_conditions='The preparation data and assumptions of each sector are kept in the register and in its evidence. The mixtures where the physical parameters of two embodiments are not identified remain marked?.'))
    assert d['total']==len(audit)==len(base)==68 and len(joint['combinations'])==len({v['identifiant'] for v in joint['combinations']})==576
    assert all(len(x['codes'])==68 for x in joint['combinations']) and sum(r['established_of_scope'] for r in base)==d['rows_with_result_of_scope']
    shear=read('SUMMARY_SHEAR')
    stability=all(v['size_duration_estimators_admitted'] for v in shear['results'])
    area=shear['interval_gain_viscosite']
    fmt=lambda x:format(x,'.4f').replace('.',',')
    if area:
        note='The four viscosity windows of the two gases in long times admit a gain of reading which is common between '+fmt(area[0])+' et '+fmt(area[1])+'. '
    else:
        note='The four viscosity windows of the two gases do not have a single reading gain at the threshold of 10%. '
    note+=('The historical criteria for duration, size and agreement of the two estimators are also met.' if stability else 'At least one ratio of size or duration or a difference between estimators exceeds its historical threshold; a uniform gain cannot change such a ratio.')
    if not stability and shear['correction_of_size']['passe']:
        rr=shear['correction_of_size']['interval_ratio_256_over_500']
        note+=' By also calibrating the reading of the two sizes, a single relative factor of '+fmt(rr[0])+' to '+fmt(rr[1])+' is suitable for both gases and two estimators.'
    if shear['brut_admitted']:
        note+=' Le facteur unitaire suffit sur ce banc.'
    count=dict(parent['counts'])
    count.update(with_scoped_result=d['rows_with_result_of_scope'],without_qualified_result=68-d['rows_with_result_of_scope'],open=68-d['rows_with_result_of_scope'],synthesis=68-d['rows_with_result_of_scope'],conditional_of_scope=d['conditions_current'],summary_scientific_whole_open=True,controls_of_benchmark_common=d['combinations_covering_67_checks_explicit'],controls_scientific_integraux=d['controls_full'])
    verdict=f"{count['with_scoped_result']} lignes sur 68 ont un résultat acquis dans leur portée, dont {count['conditional_of_scope']} sous conditions explicites. J00.1 est validée sous conditions sur le banc étalonné, avec {count['controls_of_benchmark_common']} assemblages communs sur 576. J00.2 est validée dans sa portée documentaire. La validation scientifique intégrale de BST reste ouverte."
    write(ROOT/'DECISION_R573.json',dict(revision=573,date='2026-11-20',verdict=verdict,
      finding_shear=note,counts=count,counts_parents_R572=parent['counts'],contracts_integers_valid=parent['contracts_integers_valid'],atlas_original=parent['atlas_original'],structure={**parent['structure'],'familles_de_portee_R572':parent['structure']['families_of_scope'],'families_of_scope':{'C00–C11 with the same mass deviation':['Q3.2','Q6.1','Q6.2','D08.1','D08.2','G06.2'],'two scales of action for fluorides and fifteen collective preparations':['Q1.1','Q2.1','Q2.2','Q7.2','D01.2','D03.1','D03.2'],'charge, noyaux et Sirius':['G02.4','Q3.1','D01.1','D01.3','D11.1','Q5.3'],'gaz, branches et observateurs':['D05.1','D05.2','D05.3','D06.1','D06.2','Q4.1','D09.2']},'combinations_explicitement_examines':576,'all_the_conditions_common_etablies':False},
      decision_global=d,precision_domaine='Eleven domains represented, twenty-nine clauses directly assigned to domains and sixty-four in their relationship with dependencies. The other four are G07.1, G07.2, J00.1, J00.2.',
      field='The cross-section table categories qualify the element tested on the candidate. The clauses which have partial results shall retain the reservations of their original contract.',
      reserves=dict(SF6='Thermochemical scenarios of 1980 and 2015 kept separately. Harmonic and antiresonance adjustment. The windows cover two strong bands and a weak band.',
                    grid='Two faces, one preparation and t from 0 to 0.36. Correction of discretization of wave transport. Free developments on meshes 40–72.',
                    general='The presence of a spectrum of four collective coordinates shall be established. Its physical correspondence to the claimed area of material is not available. The ranges of the two shear estimators and the sonoluminescence band energy are explained separately.'),
      predictions_frozen='entrees/PREDICTIONS_GELEES_R572.json',scores_and_calibrations='Maintain the difference between compatibility with calibration data and predictive power of theory.'))
if __name__=='__main__':main()
