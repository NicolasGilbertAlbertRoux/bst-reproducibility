'Explicit testing of each assembly on the 68 clauses. An unconnected scope remains undetermined.'
import csv,hashlib,itertools,json
from collections import Counter
from pathlib import Path
import numpy as np
from readout_ir import ROOT,I,O,decode,save
def result(name):return json.loads((O/(name+'.json')).read_text())
def proof(name):return json.loads((I/'evidence'/(name+'.json')).read_text())
def writecsv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as out:
        w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
NEW='CALCUL OU COMPARAISON SUR LE JEU'
HER='PARAMETRY LOCAL ACQUIS'
DOC='INDEPENDENT PIECE OF ADJUSTMENT'
OLD='RELATED TO THE INDETERMINED GAME'
# Each entry specifies the sub-problem to which the common parameter has actually been applied.
domains=[
('doc','G00.1 G00.2 G05.2 G05.3 G07.1 G07.2 Q8.1 Q8.2 J00.2','Documentary scope, provenance, or human observations independent of physical settings.'),
('atomic','G02.4 Q3.1 D01.1 D01.3 D11.1','Even G, same load and even alpha in five lines, doublets, rays and bonds. Reduced mass distribution and low current correlations included. Independent electrical identification remains a matter of contract G02.'),
('cosmic','G06.2 Q6.1 Q6.2 D08.1 D08.2','Same background, material fraction, PPF closure and effective mass. CMB, BAO, growth, siren and projection parameters controlled together. G06.2 aims here for the cosmological contribution.'),
('nu','Q3.2','Same difference in effective mass on reactor oscillations and in cosmological completion at a massive mode.'),
('quant','Q1.1 Q2.2','Same number of actions as in the fluorinated family over fifteen preparations. Controlled harmonic classual system risks and spacings. The assignment of a physical hadronic spectrum is absent from this model.'),
('chem','Q7.2 D03.1 D04.5','Same energy standard, same number of actions, same stiffness. Separate frequencies, fluorine release and formation cycle in the specified branch.'),
('ir','D03.2 D03.3 G06.3','Vector reader common to six arms, response to zero and combination profile R570. Intensities in two regions, two inner windows and 6,970 low band readings.'),
('motion','D04.4','Seven free fields and centers, common impedance, two readings of the corrected flow of the discretization of the packet.'),
('gas','D05.1 D05.2 D09.2','Mêmes rayons d’amas. Viriels, calorimétrie de Kr/Xe et puissance des cinq gaz. Correction commune d’énergie libre. Rôle de la phase et de la durée sur la bulle borné aux préparations documentées.'),
('clock','Q5.3 D07.3','The same atomic frequency and the same metric transport the Sirius B signal. The solution with material producing the strong field metric remains distinct.'),
('native','D02.1','Le noyau canonique choisit κ=2 et inertie=8 dans les neuf intégrations R490 depuis une même préparation.'),
('beta','D11.2','Nuclear intervals per state are transferred by certificate R558 when alpha and the other parameters are in the box R531. Out of this box, nuclear proof by state remains limited.'),
('cavity','D09.1','Same archival radius and phase estimate of the eight xenon cycles under the R568 drive. The amplitude of the detector per cycle and the power in the other acid remain separate measurements.')]
unjoined={
'G01.1':'La métrique force–lecture de R326/R491 n’a pas été identifiée à la métrique d’énergie et de temps de la réduction SFx R480.',
'G01.2':'The passivity of the reduction R491 is controlled. The common balance with the calibrated optical ports of this assembly remains to be written.',
'G02.1':'Gauss and reciprocity on the static halo R435 are acquired. The transition from the standardization of the halo to the loader of the atomic family remains conditional.',
'G02.2':'The current on mobile windows R446 has its own trajectories. Its projection as a source of the three-band optical reader is not provided.',
'G02.3':'The loaded threshold depends on the vacuum mass introduced. The game tried does not fix this mass from the same sector as the atomic frequencies.',
'G03.1':'The projection lens R440 is transported in its preparations. Its relation to zeros and gains of the SF6 reader is not established.',
'G03.2':'The weights and the poles R446 are attached to a defined reader. The observation operators of this game also use another state space.',
'G04.1':'The identities of the reservoir are worth its historical achievements. The joint embodiment of the combination circuit and the cavity has not been identified.',
'G04.2':'The on-board loss balance is acquired in the native bank. The dissipation constant of the bubble bath and the infrared port were not transported by the same balance.',
'G05.1':'The gelled CO2 reading at 728 cm-1 retains its preparation parameters. The file does not contain its translation into the same clock law as SFx.',
'G06.1':'Source and reciprocal reading of the halo have the same operator on the grid R435. No value of the charge of the infrared reader is determined by this connection.',
'Q1.2':'Signal of rotation and controlled exchange on shoelaces. The identification of the excitations of the family common to the shoelaces remains a hypothesis.',
'Q2.1':'The point-one core bears U(1). Collective SU(3) links are constructed in an extended representation. The identification of physical degrees and sources in the same preparation remains conditional.',
'Q4.1':'The closure of the R485 transport uses its potentials and trajectories. The new free energy and the three-body parameters of the state equation alone do not specify a microscopic shear force.',
'Q4.2':'The three mechano-optical devices R559 have midfielders. The parameters of their carriers remain without dictionaries with those of the gas and of the fluoride.',
'Q5.1':"Schwarzschild and Kerr's vacuum solutions are retained. The branches of Sirius in weak field do not select a source of matter that produces these two funds.",
'Q5.2':'The sign of Kerr mode on a fixed background is checked. The Sirius model used here does not provide the bottom of this bench from its source.',
'Q7.1':'Local states sharing the same free fragment were found. Response trajectories and stability in a global family sector are not specified.',
'D01.2':'The geometric reading R571 brings the fluorides distances and rotations closer together. The identity of its preparations with all the states assigned by the spectral standard remains partial.',
'D02.2':'The propagation balances R446 are acquired in the waves. The loads and polarization applied to the family of matter are not transported by the same documented source.',
'D04.1':'The work and the forces R434 are controlled on the same native states of the mechanical bench. Their change of coordinates towards the spectral reduction R480 is missing.',
'D04.2':'The invariances R434 are acquired on its states. The covariance of the transition from these states to the new optical and combination port remains to be established.',
'D04.3':'The mechanical round trips R434 belong to their action. Correspondence with the other reduced actions of the game remains conditional.',
'D05.3':'The four coexistence values R441 have an Ar anchor. The transport of the term calorie added in both phases and at co-existence has no certificate in this game.',
'D06.1':'The shearings and viscosities R444 originate from archived potentials. No trajectory under a microscopic potential realizing the new free energy law is included.',
'D06.2':'Windows and seeds R444 describe the relaxation of the archived potential. The relaxation spectrum is not deduced from the state equation alone.',
'D07.1':'Spread of Schwarzschild verified on its background. The same source for this background and for the two stellar metrics of the assembly remains to be established.',
'D07.2':'Kerr scalar propagation verified on a fixed background. The gravitation of the matter of the common family is not specified by this result.',
'D10.1':'The theorems are conditioned to their representations. A physical isomorphism with the states of matter of the game does not appear in the verified hypotheses.',
'D10.2':'The carrier and the trace channel R 528 are explained mathematically. The dossier still distinguishes the native state of matter, the biological carrier and the behavioural reading.'}
# Reuse of the benches, no adjustment of which is modified by the six choices.
# These findings shall continue to be subject to the conditions, if any, of the original proof.
herites={
'G01.1':'The conjugation and the dimensions of the native embodiment retain the standardization of their operators. The units of the fluorinated reductions are also explained in UNITES-- RUPTURE-- R573.',
'G01.2':'Conservation and passivity in the already qualified implementation. The Schur remains positive and the Euler–Lagrange residue is controlled with the seven mobile centers.',
'G02.1':'Same static operator, same Gauss form, same direct and deputy controls. The result retains its static halo range.',
'G02.2':'Same operator and same spatial windows of the dynamic current of R446.',
'G02.3':'Threshold retained with the vacuum mass value and the sector condition explicitly kept in the bench.',
'G03.1':'Frozen lens on twelve preparations. The observation geometry and its imprint are repeated without recalibration.',
'G03.2':'The operators, their weights and their poles keep the values of the projection archived in each preparation.',
'G04.1':'The native reduction of the reservoir retains its weight, moments and causality in its realization.',
'G04.2':'The native reservoir and its current keep their production. The digital flow passing through the faces is also controlled on five grids.',
'G05.1':'CO2 prediction at 728 cm-1 preserved in its frozen preparation. None of the six choices in the game changes its entries.',
'G06.1':'The static operator and its reciprocity checks remain those of the qualified bench.',
'Q1.2':'A sign of rotation and exchange retained on the archived laces, in the topological scope of the contract.',
'Q4.2':'The distinct devices and media retain their calibrations and conditions indicated in the register.',
'Q5.1':'Schwarzschild and Kerr keep exactly their vacuum metrics and the same units in this calculation on a fixed background.',
'Q5.2':'The calculation of the scalar mode sign retains the background and conditions at the already qualified Kerr limits.',
'Q7.1':'The same free fragment of R481 and even no separation block in archived preparations. The fluorinated threshold is recalculated separately.',
'D02.2':'The propagation, polarization and balance of the field bank keep their operators and their preparation native.',
'D04.1':'The archived work defects are reread for SF4 and SF6, then supplemented by the mechanical residue on the seven centres.',
'D04.2':'The archived rotations preserve the forces in the two preparations SF4 and SF6, with a lack of covariance of less than 10-11.',
'D04.3':'The comparisons of pitch, return and duration retain their settings; the same pulse also has a time step control at 403.',
'D07.1':'Spray on the background of Schwarzschild of reference, unchanged in the 576 assemblies.',
'D07.2':'Propagation on the reference Kerr bottom, unchanged in the 576 assemblies.',
'D10.1':'Formal definitions and theorems retain their declared hypotheses.',
'D10.2':'The same carrier and even trace channel of the cognitive record. Empirical interpretation retains the limits of the human protocol R528.'}
def main():
    base=list(csv.DictReader((I/'CORRESPONDANCE_68_LIGNES_R572.csv').open(encoding='utf-8-sig')))
    rr=decode('REGISTRE_68_LIGNES_R572.json'); assert [r['row'] for r in rr]==[r['row'] for r in base] and len(set(r['row'] for r in rr))==68
    origin=decode('CONTRATS_ORIGINAUX.json');assert len(origin)==28
    tests={line:(key,note) for key,ls,note in domains for line in ls.split()}
    for line,v in unjoined.items():
        assert line not in tests;tests[line]=('unknown',v)
    for line,note in herites.items():
        assert line in unjoined
        tests[line]=('heritage',note)
    tests['D01.2']=('structure','The six S-F lengths of the parent used for the spectrum are recalculated in the diffraction unit. The other geometries of the panel retain their prior scope.')
    tests['D06.1']=('shear','The shear is calculated with the forces of the same gas family. Branch 0 is tested on six seeds, two sizes and four windows. Branch 1 has only the control runs to both gases. The values of the mixtures and of the entire temperature range are not deduced from these pure states.')
    tests['D06.2']=('stability_transport','The control of the durations, of the sizes and of the two estimators uses the thresholds R444. The diffusion and relaxation windows have their own calculation. The averages and the dispersion between seeds remain in the parts.')
    tests['Q4.1']=('diffusion','Two gases, two sizes, two steps, twelve trajectories under the forces of the viriels and the same sized law. A common correction of D and its relaxation time admits an intersection with the historic bench; the 54 windows per branch pass together.')
    tests['Q2.1']=('gauss','The dip R453 of the collective bench R478 is identical to that of the spectrum. The links are induced at each displacement. Gauss law, covariance and positivity retested for the two fluorinated families and the 15 geometries. Physical diving remains a declared hypothesis.')
    tests['D05.3']=('phase','Common free energy carries the same waist law and the viriels at both phases, latent heat and calorimetry. Its dense anchor is reduced argon. The control relates to the two thermal branches.')
    tests['J00.1']=('decision','Intersection of other required clauses, with J00.2 verified separately. A single family would be enough.')
    assert set(tests)==set(r['row'] for r in base),set(r['row'] for r in base)^set(tests)
    reports={'atomic':'RECALCUL_CHARGE_COMMUNE_R573','cosmic':'COSMOLOGIE_ET_OSCILLATIONS_R573','nu':'COSMOLOGIE_ET_OSCILLATIONS_R573','quant':'QUANTIFICATION_ET_FLUORIDES_R573','chem':'UNITS_ET_RUPTURE_R573','ir':'REPONSE_COMMUNE_TROIS_BANDS_R573','motion':'FLUX_CORRECTED_R573','gas':'AJUSTEMENT_ENERGIE_FREE_R573','clock':'RECALCUL_CHARGE_COMMUNE_R573','beta':'RECALCUL_CHARGE_COMMUNE_R573','structure':'GEOMETRY_ET_PORTS_R573','phase':'PHASE_COMMUNE_R573','gauss':'GAUSS_ET_QUANTIFICATION_R573','diffusion':'INTERVALS_DIFFUSION_R573','shear':'SUMMARY_SHEAR_R573','stability_transport':'SUMMARY_SHEAR_R573'}
    cos=result(reports['cosmic']);atoms=result(reports['atomic']);qnt=result(reports['quant']);mol=result(reports['chem'])['actions_SFx'];ir=result(reports['ir'])['families'];motions=result(reports['motion']);cal=result(reports['gas'])
    opt=decode('AUDIT_COMBINAISONS_R570.json')['new_correspondence_gas']['combinaisons'];nuc=decode('INTERSECTION_R531.json');coord=decode('RAYONS_R569.json')
    assert max(abs(x-y) for x,y in zip(coord['radii_recalcules'].values(),decode('CALIBRATION_OPTIQUE_R568.json')['r']))<1e-12
    cognitive=decode('COGNITION.json');assert cognitive['Horizon']['Humains']['parties_3_1']==3960 and cognitive['Crafter']['n_enfants_analysis_of_the_parole']==10
    native=proof('CROISEMENT_SUBSTRATE_R490');canonical=next(x for x in native['cases'] if x['KAPPA']==2 and x['INERTIA']==8 and x['role']=='coefficient_grid')
    assert native['all_pass'] and abs(canonical['max_relative_energy_drift'] if 'max_relative_energy_drift' in canonical else native['max_relative_energy_drift'])<.1
    cav=decode('BILAN_CAVITE_R568.json');assert len(cav['cycles'])>=2 and cav['conditions']['frequency_Hz']==37800
    shear=result(reports['shear'])
    diffusion=result(reports['diffusion'])['families']
    gauss=result(reports['gauss'])
    phases=result(reports['phase'])['branches']
    geometry=result(reports['structure'])
    spectra={k:v['best_par_n_of_coefficients']['4'] for k,v in result('SPECTRUM_ET_THERMOCHIMIE_R573')['families'].items()}
    stars=decode('H_D_METRIQUES_R559.json')['four_controls_spectrum_clock']
    stellar=decode('AUDIT_COMBINAISONS_R570.json')['nouvel_atome_noyaux_Sirius']['controls']
    assert geometry['count_preparations_lens']==12 and decode('RACCORD_FRAGMENTS_R481.json')['max_free_block_cross_norm']<1e-9
    assert max(v['covariance'] for v in geometry['checks_actions_archived'].values())<1e-11
    scopes=result('CONDITIONS_DE_PORTEE_R573')['conditions_recorded_from_the_contracts']
    under_conditions_sans_evidence=[k for k,v in scopes.items() if v['verification_over_the_sources']!='ETABLIE']
    ledger={}
    for r in base:
        lname=r['row'];e=[Path(f).name for f in r['evidence_of_clause'].split(' | ') if f]
        e=[('entrees/preuves/'+x if (I/'evidence'/x).is_file() else 'entrees/'+x) for x in e if (I/'evidence'/x).is_file() or (I/x).is_file()]
        if not e:e=['entrees/CONTRATS_ORIGINAUX.json','entrees/REGISTRE_68_LIGNES_R572.json']
        key,note=tests[lname]
        if lname in ('G01.1','G01.2','G03.1','D04.1','D04.2'):e+=['resultats/GEOMETRIE_ET_PORTS_R573.json']
        if key=='heritage': assert any('preuves/' in x for x in e) or len(e)>=2,lname
        if lname in ('D06.1','D06.2'):
            e+=['resultats/CISAILLEMENT_R573.json','resultats/TRANSPORT_COMMUN_R573.json']
            if (O/'CISAILLEMENT_COMPLET_R573.json').exists():e+=['resultats/CISAILLEMENT_COMPLET_R573.json']
        if lname in scopes:e+=['resultats/CONDITIONS_DE_PORTEE_R573.json']
        if key in reports:e+=['resultats/'+reports[key]+'.json']
        ledger[lname]=dict(exigence=r['exigence'],evidence=e,test=key,explication=note,conditions_historical=r['status_of_row'])
    combos=[];summary={r['row']:Counter() for r in base}
    for co,o,thermal,family,alpha,met in itertools.product(cos['families'],sorted(set(x['optical'] for x in opt)),[False,True],mol,atoms,['A','B']):
        gas=next(z for z in opt if z['thermique']==thermal and z['optical']==o)
        sol=next(z for z in cal['branches'][str(int(thermal))] if z['passe'])
        aa=atoms[alpha];cc=mol[family];pt=next(z for z in ir[family]['trials'] if z['zeta']==.025)
        spectrum=spectra[family]
        gates=dict(doc=True,heritage=True,
                   shear=(shear['interval_gain_viscosite'] is not None) if not thermal else None,
                   stability_transport=(all(z['size_duration_estimators_admitted'] for z in shear['results']) or shear['correction_of_size']['passe']) if not thermal else None,
                   diffusion=diffusion[str(int(thermal))]['passe'],gauss=gauss['passe'],phase=phases[str(int(thermal))]['passe'],structure=geometry['six_in_interval'],cosmic=co['compatible'],nu=co['compatible'] and abs(co['masse_eV']**2-cos['delta_m2_eV2'])<1e-11,
                   atomic=aa['maximum']<.1 and abs(aa['HD_erreur_relative'])<.1,quant=len(qnt['summary'][family]['geometries_under_10pct'])==15,
                   chem=(spectrum['error_max_frequency_pct']<=10 and 101.5951<=cc['D6']<=113.0675),
                   ir=pt['retention_8_aires_and_6970_points'],motion=max(motions['range_corrected_over_five_grids_pct'])<10 and max(motions['difference_au_champ_fin_3D_a_N72_pct'])<10,
                   gas=sol['passe'] and gas['dix_flux_deviation_max_pct']<=10 and gas['vingt_quatre_B2_B3_max_sur_marge']<=1, native=native['all_pass'],
                   beta=True if nuc['box']['alpha'][0]<=aa['alpha']<=nuc['box']['alpha'][1] else None,cavity=True,unknown=None)
        star=next(x for x in stars if x['metric']==met)
        bounds=next(v['Sirius_samples_CCD_intersection_km_s'] for v in stellar if v['metric']==met)
        gates['clock']=(abs(star['cz_km_s']-aa['metrics'][met]['cz_km_s'])<1e-6 and bounds['2'][0]<=bounds['2'][1])
        status={}
        for line,(tag,note) in tests.items():
            if tag=='decision':continue
            value=gates[tag]
            if value is None:status[line]=OLD
            elif not value:status[line]='FLIGHT OF THE GAME'
            else:status[line]={'doc':DOC,'heritage':HER}.get(tag,NEW)
        failure='FLIGHT OF THE GAME'
        if failure in status.values():status['J00.1']=failure
        elif OLD in status.values() or under_conditions_sans_evidence:status['J00.1']=OLD
        else:status['J00.1']=NEW
        for line,v in status.items():summary[line][v]+=1
        combos.append(dict(identifiant=f'T{len(combos)+1:03d}',cosmology=co['family'],optical=o,thermique=thermal,fluorides=family,port=alpha,metric=met,
                           codes=[{NEW:'C',HER:'H',DOC:'D',OLD:'?', 'FLIGHT OF THE GAME':'X'}[status[line]] for line in ledger],classifications=dict(Counter(status.values()))))
    local_cols=[i for i,line in enumerate(ledger) if line!='J00.1']
    traces=[x['identifiant'] for x in combos if all(x['codes'][i] in ('C','H','D') for i in local_cols)]
    out=[]
    for line,row in ledger.items():
        ct=summary[line];out.append(dict(row_field=line,exigence=row['exigence'],status_historique=row['conditions_historical'],status_of_scope_R573=('VALIDED UNDER THE DISCLOSED BANKNOR' if line=='J00.1' and traces else row['conditions_historical']),controls_of_benchmark_joint=(len(traces) if line=='J00.1' else ''),combinations_computed=ct[NEW],established_parameters_retained=ct[HER],independants_of_setting=ct[DOC],correspondence_indetermine=ct[OLD],combinations_in_failure=ct['FLIGHT OF THE GAME'],finding=row['explication'],under_condition_scientific=scopes.get(line,{}).get('condition',''),records=' | '.join(row['evidence'])))
    writecsv(ROOT/'AUDIT_DES_68_LIGNES_R573.csv',out)
    save('ASSEMBLAGES_68_R573.json',dict(order_of_clauses=list(ledger),codes={'C':NEW,'H':HER,'D':DOC,'?':OLD,'X':'FLIGHT OF THE GAME'},combinations=combos))
    best=min(combos,key=lambda u:(u['classifications'].get('FLIGHT OF THE GAME',0),u['classifications'].get(OLD,0)))
    save('DECISION_GLOBALE_R573.json',dict(total=68,combinations=len(combos),combinations_covering_67_checks_explicit=len(traces),initial_with_67_checks=traces[:8],controls_full=sum(x['codes'][list(ledger).index('J00.1')]=='C' for x in combos),best_example=best,
        established_previous_retained=sum(r['established_of_scope'] for r in rr),rows_with_result_of_scope=sum(r['established_of_scope'] for r in rr)+int(bool(traces)),conditions_current=sum(v['explicitement_conditionnel'].lower() in ('oui','true','1') for v in base)+int(bool(traces)),conditions=sum(v['explicitement_conditionnel'].lower() in ('oui','true','1') for v in base),
        ranking_une_meme_combination=best['classifications'],claims_with_comparison_partial=under_conditions_sans_evidence,clauses_non_transferees_best_set=[l for l,t in zip(ledger,best['codes']) if t=='?'],
        direction="576 assemblies of sectoral coordinates actually surveyed out of the 68 wordings. Historical evidence independent of the candidate's parameters and numerical tests have distinct codes. Digital tests remain in scope. D00.1 also receives a positive result as soon as a game covers the entire concrete bench. Its scientific validation code under the full contract conditions is given separately."))
    print(len(combos),best['identifiant'],best['classifications'])
if __name__=='__main__':
    main()
