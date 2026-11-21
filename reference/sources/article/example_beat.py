'Reproduce the readings of a pair S14c. Curves in reduced units of S14c.'
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def simulate(*, direction_battement=1, stiffness_contact=3.0, dt=0.004, nb_step=40000, sample=25):
    # Values, preparation and division of stage verlet14c-energy-conserving-beat.py.
    g, k, radius, kappa, m, mu = 0.4, stiffness_contact, 0.8, 2.0, 1.0, 0.3
    if direction_battement not in (-1, 1) or k < 0 or dt <= 0 or nb_step < 1 or sample < 1:
        raise ValueError('Check the preparation, pitch and number of samples.')
    d, b, pd, pb = 2.0, 0.0, 0.0, direction_battement * np.sqrt(2 * mu * 0.30)

    def forces(d, b):
        o = max(0.0, 2*(radius+b)-d)
        return -g/max(d, 0.05)**2 + k*o, -kappa*b-2*k*o

    curve = []
    e_initial = pd**2/(2*m) + pb**2/(2*mu) - g/d
    drift = 0.0
    for step in range(nb_step + 1):
        o = max(0.0, 2*(radius+b)-d)
        e = pd**2/(2*m) + pb**2/(2*mu) - g/d + kappa*b**2/2 + k*o**2/2
        drift = max(drift, abs(e-e_initial))
        if step % sample == 0:
            curve.append((step*dt, d, b, o, k*o*pd/m, d*np.cos(np.pi/3), e))
        if step == nb_step:
            break
        fd, fb = forces(d,b)
        pd += dt*fd/2
        pb += dt*fb/2
        d += dt*pd/m
        b += dt*pb/mu
        fd, fb = forces(d,b)
        pd += dt*fd/2
        pb += dt*fb/2
        assert 0.06 < d <= 36.0
    return np.array(curve), drift


def write(out):
    out.mkdir(parents=True,exist_ok=True)
    arr,drift=simulate()
    with (out/'Lecture_deux_battements.csv').open('w',encoding='utf-8',newline='') as f:
        csv.writer(f).writerow(['t','d','b','recouvrement','puissance_contact_over_d','projection_60_degres','energy'])
        np.savetxt(f,arr,delimiter=',',fmt='%.12g')
    assert len(arr)==1601 and arr[:,3].max()>0 and arr[:,4].min()<0<arr[:,4].max()
    np.testing.assert_allclose(arr[:,5],arr[:,1]/2,rtol=1e-14)
    assert drift<.001
    summary=dict(preparation='G=0.4, E_b=0.30',horizon=160,samples=1601,energy_initial=float(arr[0,-1]),
               drift_absolute_max_over_40001_states=float(drift),d_extremes_sampled=[float(arr[:,1].min()),float(arr[:,1].max())])
    (out/'Lecture_deux_battements.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    fig,axes=plt.subplots(3,1,sharex=True,figsize=(8,5.7))
    short=arr[arr[:,0]<=20]
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',color='#dae3e2',lw=.7)
        ax.tick_params(labelsize=8)
    axes[0].plot(short[:,0],short[:,1],color='#155e60',label='Separation of centres')
    axes[0].plot(short[:,0],short[:,5],color='#b3542b',ls='--',label='View at 60 degrees')
    axes[0].legend(frameon=False,ncol=2,loc='upper right',fontsize=8)
    axes[0].set_ylabel('Longueur')
    axes[1].plot(short[:,0],short[:,2],color='#436587',label='Beat of the mantle')
    axes[1].fill_between(short[:,0],0,short[:,3],color='#97c6bb',label='Recouvrement')
    axes[1].legend(frameon=False,ncol=2,loc='upper right',fontsize=8)
    axes[1].set_ylabel('Longueur')
    axes[2].plot(short[:,0],short[:,4],color='#8e492e')
    axes[2].axhline(0,color='#888',lw=.55)
    axes[2].set_ylabel('Puissance')
    axes[2].set_xlabel('Model time')
    axes[0].set_title('Two beats seen by an outside observer',loc='left',weight='bold',pad=15)
    fig.subplots_adjust(left=.12,bottom=.12,right=.98,top=.91,hspace=.3)
    fig.savefig(out/'Lecture_deux_battements.png',dpi=160)
    plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);args=p.parse_args();write(args.output)
