'Prolonged R569 potential coefficients in the region of coexistence.'
import numpy as np
from calorimetry import base,ext
from readout_ir import decode,save
def main():
    geo=decode('RAYONS_R569.json')
    scales={g:geo['lengths_calibrees_m'][g]/base.geometry[g]['sigma_m'] for g in base.GASES}
    grid=sorted(set([*np.arange(145.,506.,15.),160.,180.,215.,245.,270.,290.,298.15,300.,325.,348.15,350.,375.,400.,425.,450.,475.,500.]))
    base.TIME=grid;ext.TIME=grid;base.v.TEMPERATURES=np.array(grid)
    order=17;limit=20.
    models=ext.add_modes(base.prepare(scales,order,limit,2.45),scales,order,limit,2.45)
    out={}
    for g,((tails,weight,e2,e3,B),pair) in models.items():
        out[g]={}
        for m,v in enumerate(decode('THERMO_R569.json')['trials']):
            q=np.array(v['coefficients']);p0=e3+B@q[:6];rows=[]
            for i,t in enumerate(grid):
                p=p0+(B[:,0]*(t-350)/100*q[-1] if v['thermique'] else 0.)
                c=(pair[i]+tails[i]-weight@(np.exp(-e2/t)*np.expm1(np.clip(-p/t,-745,500))))*1e-12
                rows.append([t,base.second(g,t,scales[g])*1e-6,c])
            out[g][str(m)]=rows
    save('VIRIELS_ETENDUS_R573.json',{'order':order,'radius':limit,'calculs':out})
if __name__=='__main__':main()
