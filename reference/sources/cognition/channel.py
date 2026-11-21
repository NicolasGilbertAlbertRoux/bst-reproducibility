'Contrast of the reading channel H528 between the two preparations at the same J.'
import argparse
import csv
import importlib.util
import itertools
from pathlib import Path
import shutil
import tempfile
import numpy as np
from compute import COGNITION, write_table, json_write


def compute(output):
    with tempfile.TemporaryDirectory() as tmp:
        source=Path(tmp)/'R528'
        shutil.copytree(COGNITION/'sources/R528',source)
        spec=importlib.util.spec_from_file_location('channel_historique',source/'tester_persistance_R528.py')
        h=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(h)
        eta_test=[0.,.025,.05,.10,.15,.2,.5,.8]
        all=[h.make_case(tag=tag,eta=eta,lam=.5,mass=.5,impulse=J,sign=m)
              for tag,eta,J,m in itertools.product(['m1','m2'],eta_test,[.02,.1,.2],[-1,1])]
        rec,_,_=h.integrate(all,dt=.00005,end=.04,sample=.01)
        tests=[];rel=0.;blind=0.
        for i in range(0,len(all),2):
            c=all[i];J=c['impulse'];eta=c['eta']
            estim=[]
            for k in [2,4]:
                t=rec[k,i,0]
                estim.append(6*(rec[k,i,1]-rec[k,i+1,1])/t**3)
            d=(4*estim[0]-estim[1])/3
            exact=2*J*h.K0*np.sinh(eta)
            if eta>0:
                rel=max(rel,abs(d-exact)/exact)
            else:
                blind=max(blind,abs(d))
            tests.append([c['tag'],eta,J,exact,d])
        assert rel<.001 and blind==0
        eta=np.linspace(0,.8,801)
        curve=2*h.K0*np.sinh(eta)
        write_table(output/'Canal_H528_continu.csv',['eta','contrast_min_J_002','contrast_J_010','contrast_max_J_020'],
                     zip(eta,.02*curve,.1*curve,.2*curve))
        write_table(output/'Canal_H528_controles.csv',['bath','eta','J','contrast_exact','estimation_trajectoires'],tests)
        ratio=dict(coefficient_k0=h.K0,contrast='abs(x_plus_tierce-x_moins_tierce)=2*abs(J)*k0*sinh(eta)',
                     eta_computed=eta_test,J_interval=[.02,.2],baths=['m1','m2'],
                     pairs_verifiees=len(tests),trajectoires_short=len(all),
                     deviation_relative_max_outside_zero=float(rel),contrast_a_zero=float(blind),
                     step=.00005,samples_sonde=[.02,.04],bound_continue_in_J=True)
        json_write(output/'Bilan_canal.json',ratio)
        print('Verification of the probe:',len(tests),'pairs, maximum relative deviation',rel)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);compute(a.output)
