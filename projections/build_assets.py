"""Pack self-contained GIF references and lossless channels for local HTML and Node."""
from pathlib import Path
from PIL import Image
import numpy as np
import zlib,base64,json
from films import SPECTRAL,spectral_colour

NAMES=['beat-contact','atomic-krypton','molecular-krypton','excited-dicarbon']

def code(b):return base64.b64encode(b).decode('ascii')

def build(movies,cache,path,english=None):
    data={'films':{k:code((Path(movies)/(k+'.gif')).read_bytes()) for k in NAMES},'colour':{str(x):spectral_colour(x) for x in SPECTRAL},'species':{}}
    english=Path(english) if english else Path(movies)/'en'
    data['english']={k:code((english/(k+'.gif')).read_bytes()) for k in NAMES}
    for key in ['Kr2','C2']:
        with np.load(Path(cache)/(key+'.npz')) as f:
            info=json.loads(str(f['info']));period=info['period_ps' if key=='Kr2' else 'period_fs'];records=[]
            for ch in f['channels']:
                image=Image.fromarray(ch.transpose(1,2,0)).resize((384,186),Image.Resampling.BOX)
                records.append(code(zlib.compress(np.asarray(image).tobytes(),7)))
            data['species'][key]={'planes':records,'distance':f['separation'].round(8).tolist(),'time':(period*f['phase']).round(7).tolist()}
    Path(path).write_text('window.BST_FILMS = '+json.dumps(data,separators=(',',':'))+';\n')

if __name__=='__main__':
    import sys
    build(*sys.argv[1:])
