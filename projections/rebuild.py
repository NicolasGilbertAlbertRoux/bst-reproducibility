"""Rebuild the approved plates and the projection inputs from the frozen sources."""
from pathlib import Path
import hashlib,json,os,subprocess,sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REFERENCES={'BST_Molecule_de_krypton.gif':'molecular-krypton.gif','BST_Couleur_du_dicarbone.gif':'excited-dicarbon.gif'}

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def rebuild(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,BST_PLATE_LANGUAGE='fr',BST_PROJECTION_WORK=str(out),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    for script,args in [('primordial.py',[]),('project.py',[]),('films.py',[]),('physical_views.py',[str(out/'extended.js')]),('thermal_crystals.py',[str(out/'crystals.js')]),('optics.py',[str(out/'optics.js')])]:
        subprocess.run([sys.executable,str(HERE/script),*args],env=env,check=True)
    movies=out/'movies'
    for old,new in REFERENCES.items():(movies/old).replace(movies/new)
    en=out/'en';en.mkdir(exist_ok=True)
    en_env=dict(env,BST_PROJECTION_WORK=str(en),BST_PLATE_LANGUAGE='en')
    (en/'cache').mkdir(exist_ok=True)
    import shutil
    for spec in ['Kr2','C2']:
        shutil.copyfile(out/'cache'/(spec+'.npz'),en/'cache'/(spec+'.npz'))
    for script in ('primordial.py','films.py'):
        subprocess.run([sys.executable,str(HERE/script)],env=en_env,check=True)
    target=movies/'en';target.mkdir(exist_ok=True)
    for path in (en/'movies').glob('*.gif'):
        path.replace(target/REFERENCES.get(path.name,path.name))
    subprocess.run([sys.executable,str(HERE/'build_assets.py'),str(movies),str(out/'cache'),str(out/'films.js')],env=env,check=True)
    result={}
    for name in ('beat-contact.gif','atomic-krypton.gif','molecular-krypton.gif','excited-dicarbon.gif'):
        a=movies/name;b=ROOT/'examples'/name;result[name]={'sha256':digest(a),'matches_reference':digest(a)==digest(b)}
    for name in ('beat-contact.gif','atomic-krypton.gif','molecular-krypton.gif','excited-dicarbon.gif'):
        a=target/name;b=ROOT/'examples'/'en'/name
        result['en/'+name]={'sha256':digest(a),'matches_reference':digest(a)==digest(b)}
    for f in ('films.js','extended.js','crystals.js','optics.js'):
        result[f]={'sha256':digest(out/f),'matches_reference':digest(out/f)==digest(ROOT/'observatory'/f)}
    (out/'projection-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    if not all(x['matches_reference'] for x in result.values()):raise ValueError('One or more rebuilt outputs differ. See projection-verification.json and compare the numerical metadata on this machine.')
    return result
