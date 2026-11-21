from pathlib import Path
import json,sys,unittest,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import render

class NativeDataTests(unittest.TestCase):
    def test_native_velocity_independent_difference(self):
        for e in json.loads((ROOT/'catalogue.json').read_text())['scenes']:
            if e['kind']!='native_nodes':continue
            d=render.charger(e,ROOT);w=d['omega_native'];m=d['modes'];t=.123/max(w);h=1e-4/max(w)
            q=lambda x:np.einsum('m,mnc->nc',np.cos(w*x),m)
            fd=(q(t+h)-q(t-h))/(2*h);v=np.einsum('m,mnc->nc',-w*np.sin(w*t),m)
            self.assertLess(np.linalg.norm(fd-v)/np.linalg.norm(v),1e-7)

    def test_turning_point_does_not_remove_stored_energy(self):
        for e in json.loads((ROOT/'catalogue.json').read_text())['scenes']:
            if e['kind']!='native_nodes':continue
            d=render.charger(e,ROOT);j=d['selected'];mode=d['modes'][j];w=d['omega_native'][j]
            kinetic=.5*np.sum((-w*mode*np.sin(0.))**2)
            potential=.5*w*w*np.sum(mode*mode)
            self.assertEqual(kinetic,0);self.assertGreater(potential,0)

    def test_grayscale_overlay_occupancy_is_bounded(self):
        canvas=np.zeros((30,30));b=render.sprite(12)
        for x,y in [(-1,-1),(15,15),(29,29),(15,15)]:render.store(canvas,x,y,b,.8)
        self.assertGreater(canvas.max(),0);self.assertLessEqual(canvas.max(),1);self.assertGreaterEqual(canvas.min(),0)

if __name__=='__main__':unittest.main()
