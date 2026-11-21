"""Assemble neutral atlas sites and export a coherent beat in model display units."""
import argparse
import base64
import csv
import json
import zlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def read_js(path):
    text = path.read_text()
    return json.loads(text[text.index('{'):text.rindex('}') + 1])


def assemble(light, data):
    atlas, positions = {}, {}
    names = {4:'Kr', 5:'Xe', 6:'Rn'}
    for z,symbol,period,archived,total,nsurface,*rest in data['elements']:
        if symbol in light:
            rec = light[symbol]
            p = np.array(rec['points'])
            u = np.array(rec['u'])
            flags = np.asarray(rec['surface'])
            origin = np.ones(len(p))
            if symbol == 'H':
                # The neutral count model adds one central site to six relaxed surface coordinates.
                p = np.vstack([p, np.zeros(3)])
                u = np.vstack([u, np.zeros(3)])
                flags = np.r_[flags,0]
                origin = np.r_[origin,0]
            omega = rec['omega']
        else:
            shell = data['atoms'][symbol]
            directions = np.array(shell['points'])
            directions /= np.linalg.norm(directions,axis=1)[:,None]
            mode = np.array(shell['mode'])
            parent = positions[names[period-1]]
            assert len(parent) + len(directions) == total
            # Period closures fix the core count. Their placement within this new atom is a display hypothesis.
            # Equal mean site volume sets a radial scale, capped to retain a half-spacing gap to the shell.
            spacing = (4*np.pi/(3*total))**(1/3)
            core_scale = min((len(parent)/total)**(1/3), 1-spacing/2)
            p = np.vstack([parent*core_scale, directions])
            u = np.vstack([np.zeros_like(parent), directions*(mode/np.max(abs(mode)))[:,None]])
            flags = np.r_[np.zeros(len(parent)),np.ones(len(directions))]
            origin = flags.copy()
            omega = np.sqrt(shell['eigenvalue'])
        assert len(p) == total and int(flags.sum()) == nsurface and np.isfinite(p).all()
        positions[symbol] = p
        spacing = float(np.median(cKDTree(p).query(p,k=2)[0][:,1]))
        # Columns: equilibrium xyz, native displacement direction xyz, outer surface flag, source-coordinate flag.
        record = np.column_stack([p,u,flags,origin]).astype('<f4')
        atlas[symbol] = {'count':int(total),'surfaceCount':int(nsurface),'archivedCount':archived,
                         'sourceCount':int(origin.sum()),'spacing':spacing,'omega':float(omega),
                         'raw':base64.b64encode(zlib.compress(record.tobytes(),6)).decode('ascii')}
    return atlas


def export(element, phase, output):
    data = read_js(ROOT/'observatory/extended.js')['atlas'][element]
    t = 2*np.pi*phase
    records = np.frombuffer(zlib.decompress(base64.b64decode(data['raw'])),dtype='<f4').reshape(-1,8)
    radius = .46*data['spacing']*(1+.28*np.cos(t))
    speed = -.46*data['spacing']*.28*2*np.pi*np.sin(t)
    with output.open('w',newline='') as target:
        w=csv.writer(target);w.writerow(['site','x','y','z','radius','d_radius_d_cycle','outer_surface','source_position'])
        for i,row in enumerate(records):
            w.writerow([i,*(row[:3]+.095*np.cos(t)*row[3:6]),radius,speed,int(row[6]),int(row[7])])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('element');p.add_argument('output',type=Path);p.add_argument('--phase',type=float,default=0.)
    a=p.parse_args();export(a.element,a.phase,a.output)
