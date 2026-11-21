'Reread the cells and parts of each clause, and reproduce the main figures.'
from pathlib import Path
import csv,json,hashlib
import numpy as np
from readout_ir import ROOT,I,O,decode
def main():
    get=lambda n:json.loads((O/(n+'_R573.json')).read_text())
    registry=json.loads((ROOT/'REGISTRE_68_LIGNES_R573.json').read_text())
    tests=get('COMBINATIONS_68');d=get('DECISION_GLOBAL');li=list(csv.DictReader((ROOT/'AUDIT_DES_68_LIGNES_R573.csv').open(encoding='utf-8-sig')))
    assert len(registry)==len(li)==len(tests['order_of_clauses'])==68 and len(tests['combinations'])==576
    assert len({tuple(x[k] for k in ('cosmology','optical','thermique','fluorides','port','metric')) for x in tests['combinations']})==576
    for j,a in enumerate(li):
        assert a['row']==registry[j]['row']==tests['order_of_clauses'][j]
        assert all((ROOT/z).is_file() for z in a['records'].split(' | '))
        summary=[sum(v['codes'][j]==k for v in tests['combinations']) for k in ('C','H','D','?','X')]
        assert summary==[int(a[k]) for k in ('combinations_computed','established_parameters_retained','independants_of_setting','correspondence_indetermine','combinations_in_failure')]
    assert d['controls_full']==0 and d['established_previous_retained']==67 and d['conditions']==43
    assert d['conditions_current']==44 and d['rows_with_result_of_scope']==68 and sum(r['established_of_scope'] for r in registry)==68
    assert d['combinations_covering_67_checks_explicit']==192
    f=get('FLUX_CORRECTED');v=np.array([r['estimation_meme_sonde'] for r in f['estimations']]);np.testing.assert_allclose(100*abs(v[-2]/v[-1]-1),f['change_corrected_64_72_pct'])
    assert max(f['range_corrected_over_five_grids_pct'])<.061
    for k,ans in get('REPONSE_COMMUNE_TROIS_BANDS')['families'].items():
        y=next(a for a in ans['trials'] if a['zeta']==.025);assert y['deviation_huit_aires_pct']<5.71 and sum(a['in_U'] for a in y['profile_weak'].values())==6970
    assert get('COSMOLOGIE_ET_OSCILLATIONS')['compatibles']==12 and get('GAUSS_ET_QUANTIFICATION')['passe']
    assert all(v['passe'] and len(v['phase'])==22 for v in get('PHASE_COMMUNE')['branches'].values())
    assert all(x['passe'] and x['count_comparisons']==54 and x['intersection_common'][0]<x['intersection_common'][1] for x in get('INTERVALS_DIFFUSION')['families'].values())
    assert len(get('SHEAR_FULL')['trajectoires'])==48 and len(get('SHEAR_FULL')['groups'])==8
    assert len(get('TRANSPORT_COMMUN')['trajectoires'])==12 and len(get('SHEAR')['trajectoires'])==4
    assert hashlib.sha256((I/'PTB_RAW.zip').read_bytes()).hexdigest()=='bfa3f228a09cfd7044a388a7a073b593b33d3696252cb3b3e8df76a37b73a490'
    print('68 clauses - 576 games. Sums of columns, parts and comparisons verified.')
if __name__=='__main__':main()
