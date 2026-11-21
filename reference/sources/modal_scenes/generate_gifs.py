#!/usr/bin/env python3
'Generate all native GIFs in the catalogue, with fingerprint-verified take-up.'
from pathlib import Path
import argparse,hashlib,json,os,sys,platform
import numpy as np
from PIL import Image, __version__ as pillow_version
import render
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--catalogue',type=Path,default=ROOT/'catalogue.json')
    ap.add_argument('--sortie',type=Path,default=ROOT/'GIFS_R178')
    ap.add_argument('--especes',nargs='+',default=['all'])
    ap.add_argument('--liste',action='store_true')
    ap.add_argument('--frames',type=int,default=96);ap.add_argument('--size',type=int,default=500)
    ap.add_argument('--duration-ms',type=int,default=80)
    ap.add_argument('--magnification',type=float,default=8.);ap.add_argument('--radial-magnification',type=float,default=64.)
    ap.add_argument('--font',type=str);ap.add_argument('--force',action='store_true')
    a=ap.parse_args()
    if not 24<=a.frames<=1200:ap.error('--frames between 24 and 1200')
    if not 256<=a.size<=1600:ap.error('--size between 256 and 1600')
    if a.duration_ms<20 or a.duration_ms%10:ap.error('--duration-ms must be a multiple of 10, at least 20')
    if not 0<a.magnification<=100 or not 0<a.radial_magnification<=256:ap.error('grossissement hors limites')
    cat=a.catalogue.resolve();root=cat.parent;j=json.loads(cat.read_text());scenes=j['scenes']
    if j.get('schema')!=1:ap.error('version de catalogue non prise en charge')
    if any(x['kind'] not in ['native_nodes','molecular_current'] for x in scenes):ap.error('type of scene not supported')
    names=[x['species'] for x in scenes]
    if len(set(names))!=len(names):ap.error('duplicated species in the catalogue')
    if a.list:
        print(j['coverage'])
        for x in scenes:print(x['species'],x['kind'],x['source'])
        return
    unknown=set(a.species)-set(names)-{'all'}
    if unknown:ap.error('absent native data for: '+', '.join(sorted(unknown)))
    selected=scenes if 'all' in a.species else [x for x in scenes if x['species'] in a.species]
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    opts={k:getattr(a,k) for k in ['frames','size','duration_ms','magnification','radial_magnification','font']}
    f=render.police(15,a.font);font_path=getattr(f,'path',None)
    opts['renderer_environment']={'python':platform.python_version(),'numpy':np.__version__,'Pillow':pillow_version,
       'font_sha256':sha(Path(font_path)) if font_path and Path(font_path).is_file() else 'Pillow-default'}
    algorithm=sha(Path(__file__))+sha(ROOT/'rendu.py');records=[]
    for entry in selected:
        sp=entry['species']
        if not sp.replace('_','').isalnum():raise ValueError('invalid species name')
        for key in ['file','graph_file']:
            if key in entry:
                p=(root/entry[key]).resolve()
                if not p.is_relative_to(root):raise ValueError('entry outside the catalogue file')
                if sha(p)!=entry['sha256' if key=='file' else 'graph_sha256']:raise ValueError('different footprint: '+sp)
        d=render.charger(entry,root)
        variants=['motion','diffusion','superposition'] if entry['kind']=='native_nodes' else ['current_mode_lent','currents']
        for variant in variants:
            if variant=='diffusion' and 'graph_file' not in entry:continue
            dest=out/(sp+'_'+variant+'.gif');side=dest.with_suffix('.json')
            fp=hashlib.sha256(json.dumps([entry,variant,opts,algorithm],sort_keys=True).encode()).hexdigest()
            if not a.force and dest.exists() and side.exists():
                old=json.loads(side.read_text())
                if old.get('request_sha256')==fp and old.get('gif_sha256')==sha(dest):
                    print(sp,variant,': already checked, kept',flush=True);records.append(old);continue
            print(sp,variant,': calcul et rendu',flush=True)
            frames,meta=render.sequence(entry,d,variant,opts,root)
            tmp=dest.with_suffix('.gif.tmp')
            frames[0].save(tmp,format='GIF',save_all=True,append_images=frames[1:],duration=a.duration_ms,loop=0,disposal=2)
            with Image.open(tmp) as im:
                duration=0
                for k in range(im.n_frames):im.seek(k);im.load();duration+=im.info.get('duration',0)
                assert duration==a.frames*a.duration_ms
                assert im.n_frames>1
                meta.update(encoded_frames=im.n_frames,duration_ms=duration)
            os.replace(tmp,dest);meta.update(file=dest.name,gif_sha256=sha(dest),request_sha256=fp)
            meta['renderer_environment']=opts['renderer_environment']
            side.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n');records.append(meta)
            frames[min(len(frames)//4,len(frames)-1)].save(out/(dest.stem+'_apercu.png'))
    report={'revision':178,'coverage':j['coverage'],'species':sorted({r['species'] for r in records}),
            'gif_count':len(records),'records':records,'native_banks_modified':False,'physical_light_simulated':False}
    (out/'BILAN_GIFS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f"Terminé : {len(records)} GIFs ; {out}",flush=True)
if __name__=='__main__':main()
