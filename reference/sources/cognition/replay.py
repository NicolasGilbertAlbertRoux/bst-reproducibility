'Recalculates then verifies the figures and the results of the extension towards the latent.'
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ICI=Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_field(output):
    ref=ICI/'results'
    with np.load(ref/'Trajectoires_H528_prolongees.npz') as before, np.load(output/'Trajectoires_H528_prolongees.npz') as after:
        assert set(before.files)==set(after.files)
        for key in before.files:
            np.testing.assert_allclose(before[key],after[key],rtol=5e-7,atol=2e-9,err_msg=key)
    # Words and their commitments are categories, compared literally.
    with (ref/'Mots_et_reponses.csv').open(newline='') as a, (output/'Mots_et_reponses.csv').open(newline='') as b:
        assert list(csv.reader(a))==list(csv.reader(b))
    for name_field in sorted(i.name for i in ref.glob('*.csv') if i.name!='Mots_et_reponses.csv'):
        x=np.genfromtxt(ref/name_field,delimiter=',',names=True,dtype=None,encoding='utf-8')
        y=np.genfromtxt(output/name_field,delimiter=',',names=True,dtype=None,encoding='utf-8')
        assert x.shape==y.shape and x.dtype.names==y.dtype.names
        for key in x.dtype.names:
            if x[key].dtype.kind in 'fi':
                np.testing.assert_allclose(x[key],y[key],rtol=5e-7,atol=2e-9,err_msg=f'{name_field}/{key}')
            else:
                np.testing.assert_array_equal(x[key],y[key],err_msg=f'{name_field}/{key}')
    h=read(output/'Bilan_prolongement.json');c=read(output/'Bilan_canal.json')
    assert h['H528']['preparations']==144 and h['H528']['samples']==2401 and len(h['H528']['validation_DOP853'])==4
    assert c['pairs_verifiees']==48 and c['deviation_relative_max_outside_zero']<.001 and c['contrast_a_zero']==0
    assert h['pairs_of_phases']['words_protected']==256 and h['pairs_of_phases']['words_after_remise_a_zero']==1
    return {'H528':h['H528'],'pairs_of_phases':h['pairs_of_phases'],'canal_H528':c}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    parser.add_argument('--reference',type=Path,help='Trajectories of the original R528 bench, if available.')
    args=parser.parse_args();out=args.output.resolve()
    if out==ICI or ICI in out.parents:
        parser.error('Choose a new case outside the sources.')
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        parser.error('The output folder shall be empty.')
    data=out/'results'
    cmd=[sys.executable,str(ICI/'calculer.py'),str(data)]
    if args.reference:cmd.extend(['--reference',str(args.reference.resolve())])
    subprocess.run(cmd,check=True)
    subprocess.run([sys.executable,str(ICI/'canal_h528.py'),str(data)],check=True)
    subprocess.run([sys.executable,str(ICI/'resolution_h528.py'),str(data),str(out/'figures')],check=True)
    summary=check_field(data)
    summary['resolution_and_transfer']=read(data/'Bilan_resolution.json')
    expected=read(ICI/'resultats/Bilan_resolution.json')
    measurements=summary['resolution_and_transfer']['transfer']['measurements']
    for new, old in zip(measurements, expected['transfer']['measurements'], strict=True):
        for key in ('minimum_q_sampled','t_premier_zero_Verlet','t_premier_zero_DOP853'):
            if old[key] is None:
                assert new[key] is None
            else:
                np.testing.assert_allclose(new[key], old[key], rtol=2e-6, atol=1e-7)
    subprocess.run([sys.executable,str(ICI/'illustrer.py'),str(data),str(out/'figures')],check=True)
    (out/'BILAN_LATENT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Five figures and the associated tables have been recalculated and verified.')


if __name__=='__main__':
    main()
