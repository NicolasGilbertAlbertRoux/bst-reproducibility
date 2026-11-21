'Identify conditions that are not tested by a simple coverage of the 68 cells.'
import json
from readout_ir import I,O,ROOT,decode,save
def main():
    contracts={r['contract']:r['definition'] for r in decode('CONTRATS_ORIGINAUX.json')}
    q=decode('COLLECTIF_R478/resultats/QUANTIFICATION_COMMUNE_R478.json')['cases']
    assert len(q)==15 and all('hadron_spectrum_claim' in r for r in q)
    src=decode('preuves/ENERGIE_SPECTRALE_R569.json')
    p=decode('BILAN_CAVITE_R568.json')
    assert 'tension' in src['scope'] and 'mean' in src['scope']
    results={
      'Q2.2':dict(contract=contracts['Q2']['title'],exigence_original=contracts['Q2']['required_deliverable'],scope_covered='Four excitations and twelve levels of the collective bench with fifteen geometries. The two action scales calibrated in the fluorides are transported.',condition='Assignment of this spectrum to the physical states of the strong interaction is explicitly left open in the fifteen outputs. The effective gauge model is allowed, the level of spectrum allocation must accompany the result.',cases_sans_attribution=sum(not r['hadron_spectrum_claim'] for r in q),piece='entrees/COLLECTIF_R478/resultats/QUANTIFICATION_COMMUNE_R478.json'),
      'D09.2':dict(contract=contracts['D09']['title'],exigence_original=contracts['D09']['acceptance'],scope_covered='Integrated spectral measurements and comparisons on five gases. Radius and tension of eight cycles in the xenon preparation. Agreement on calibrated observables.',condition='For the synchronized Hopkins protocol, the voltage sets the time and relative amplitude. The band energy derived from the spectrum is averaged over periods. The total absolute energy and the cycle spectrum of these shots are not identified together in the available rooms.',piece='entrees/preuves/ENERGIE_SPECTRALE_R569.json',measurements_of_band=src['measurements'],etude='https://doi.org/10.1103/PhysRevLett.95.254301',frequency_Hz=p['conditions']['frequency_Hz']),
      'G02.4':dict(contract=contracts['G02']['title'],exigence_original=contracts['G02']['acceptance'],scope_covered='Three conventions transport load standardization to lines, rays, links and clocks. Atomic and nuclear tests keep their mistakes.',condition='The electrical identity independent of the calibration of the fine structure constant only retains the status of the proof in the register. Two standards of this constant do not demonstrate it by themselves.',piece='resultats/RECALCUL_CHARGE_COMMUNE_R573.json')}
    register={x['row']:x for x in decode('REGISTRE_68_LIGNES_R572.json')}
    results['Q2.2']['verification_over_the_sources']='ETABLIE' if all(r['hadron_spectrum_claim'] for r in q) else 'NON-PENPOSED'
    results['D09.2']['verification_over_the_sources']='NON-PENPOSED'
    operations=register['G02.4']['last_detail_of_clause']
    results['G02.4']['conditions_of_the_evidence_spectrale']=operations['conditions_common']
    results['G02.4']['verification_over_the_sources']='NON-PENPOSED' if any('encore ouverts' in x for x in operations['conditions_common']) else 'TO BE FROM'
    save('CONDITIONS_DE_PORTEE_R573.json',dict(acceptation_J00=contracts['J00']['acceptance'],conditions_recorded_from_the_contracts=results,decision='At the level of the calibrated comparison, the above recalculations count positively. For the universal claim, a box having a numerical comparison may bear an indeterminate sub-condition. No logical failure of all combinations is to be derived from this.'))
if __name__=='__main__':main()
