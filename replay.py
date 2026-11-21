"""Replay the BST manuscript, a single domain, the Lean proofs or the observatory."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import argparse, csv, hashlib, json, math, os, subprocess, sys

ROOT=Path(__file__).resolve().parent
from reference.stage import stage, digest
SECTIONS={'pair':'ouverture','reduced-dynamics':'reduction','substrate':'substrat','clocks-and-flux':'battement','atomic-modes':'d01','domain-observables':'domaines','foundations':'premiers-essais','cognition':'cognition','memory-extension':'latent','archived-movies':'animations','cosmology':'cosmologie','physics':'physique-rapide'}
def verify():
    registry=json.loads((ROOT/'checksums.json').read_text())
    for name,expected in registry.items():
        if digest(ROOT/name)!=expected:raise ValueError(f'File differs: {name}')
    return {'verified_files':len(registry)}
def check_observer(actual, expected, location='observer'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys()!=expected.keys():
            raise ValueError(f'Observer fields differ at {location}')
        for key, value in expected.items():
            check_observer(actual[key], value, f'{location}.{key}')
    elif isinstance(expected, list):
        if not isinstance(actual,list) or len(actual)!=len(expected):
            raise ValueError(f'Observer array differs at {location}')
        for i, (a,b) in enumerate(zip(actual,expected)):
            check_observer(a,b,f'{location}[{i}]')
    elif isinstance(expected,float):
        if not math.isclose(actual,expected,rel_tol=2e-8,abs_tol=2e-9):
            raise ValueError(f'Observer value differs at {location}')
    elif actual!=expected:
        raise ValueError(f'Observer value differs at {location}')

def run(args):
    if args.section=='list':print('\n'.join(['integrity',*SECTIONS,'proofs','observatory','projections','export-atlas','atlas','neon','extract','all','long']));return
    if not __debug__:raise RuntimeError('Run Python without -O. Scientific checks use assertions.')
    if args.section=='integrity':print(json.dumps(verify()));return
    if args.output.exists() and any(args.output.iterdir()):raise SystemExit('Choose an empty output directory.')
    args.output.mkdir(parents=True,exist_ok=True);out=args.output.resolve();report=verify()
    if args.section=='projections' and not args.element:
        from projections.rebuild import rebuild
        report['projections']=rebuild(out)
    elif args.section in ('projections','export-atlas'):
        target=['atom',args.element,str(out/(args.element+'.gif'))] if args.element else ['atlas','all',str(out/'neutral-atlas')]
        subprocess.run(['node',str(ROOT/'observatory/render.cjs'),*target],check=True)
        report['projection_movies']=len(list(out.rglob('*.gif')))
    elif args.section=='proofs':
        from proofs.check import check
        report['proofs']=check(out/'lean-report.json')
    elif args.section=='atlas':
        from observatory.catalogue import export
        report['atlas']=export(out,args.element)
    elif args.section=='neon':
        from observatory.catalogue import neon
        report['neon']=neon(args.pressure,args.interval)
    elif args.section=='extract':
        stage(out/'benchmark');report['working_tree']='benchmark'
    else:
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with TemporaryDirectory(prefix='bst-replay-') as tmp:
            base=stage(Path(tmp)/'benchmark')
            if args.section in ('observatory','all'):
                from observatory.build_data import materialize
                from observatory.catalogue import data
                actual=materialize(base,out/'observatory-data.js')
                check_observer(json.loads(json.dumps(actual)), data())
                report['observatory']='recomputed'
            if args.section=='long':
                dest=out/'physics';dest.mkdir()
                with ZipFile(base/'references/BST_R573_AUDIT_ET_COURBES.zip') as z:z.extractall(dest)
                with (out/'long-physics.log').open('w') as f:
                    subprocess.run(['sh','REJOUER.command','--complet'],cwd=dest/'BST_R573',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
                report['long_physics']='completed'
            for key in SECTIONS if args.section=='all' else [args.section]:
                if key not in SECTIONS:continue
                destination=out/key;print(f'Running {key}',flush=True)
                with (out/(key+'.log')).open('w') as f:
                    subprocess.run([sys.executable,str(base/'rejouer.py'),'--choix',SECTIONS[key],'--sortie',str(destination)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
                report[key]=json.loads((destination/'BILAN.json').read_text())
            if args.section=='all':
                from projections.rebuild import rebuild
                report['projections']=rebuild(out/'projections')
                from proofs.check import check
                from observatory.catalogue import export
                report['proofs']=check(out/'lean-report.json');report['atlas']=export(out/'atlas',None)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(out/'report.json')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('section',choices=['list','integrity',*SECTIONS,'proofs','observatory','projections','export-atlas','atlas','neon','extract','long','all'])
    p.add_argument('--output',type=Path,default=Path('bst-results'));p.add_argument('--element',help='Limit the atlas to one element symbol')
    p.add_argument('--pressure',type=float,default=3.1);p.add_argument('--interval',nargs=2,type=float,metavar=('LOW_NW','HIGH_NW'))
    run(p.parse_args())
if __name__=='__main__':main()
