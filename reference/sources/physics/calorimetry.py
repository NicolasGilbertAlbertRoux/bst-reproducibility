'Replay the caloric measurement with the two potentials already calibrated on the gases and light.'
import json,sys
import numpy as np
from scipy.constants import R
from readout_ir import I,decode,save
sys.path.insert(0,str(I/'VIRIELS_R569'/'code'))
import viriels_common as base
import refine_viriels as ext
def main():
    original=json.loads((I/'preuves/ENERGIE_GAZ_RESERVE_R442.json').read_text())['cases']
    optical=decode('RAYONS_R569.json')
    scales={g:optical['lengths_calibrees_m'][g]/base.geometry[g]['sigma_m'] for g in base.GASES}
    species=base.GASES;temperatures=[400.,450.,500.]
    grid=sorted(t+j for t in temperatures for j in [-2,-1,0,1,2])
    base.TIME=grid;ext.TIME=grid;base.v.TEMPERATURES=np.array(grid)
    records={}
    for order,rmax in [(17,20.),(25,30.)]:
        models=ext.add_modes(base.prepare(scales,order,rmax,2.45),scales,order,rmax,2.45)
        for reg in decode('THERMO_R569.json')['trials']:
            isthermal=reg['thermique'];pars=np.array(reg['coefficients'])
            cases=[]
            for g in species:
                (tails,weight,e2,e3,basis),pair=models[g]
                vir=[]
                for i,temp in enumerate(grid):
                    potential=e3+basis@pars[:6]
                    if isthermal:potential+=basis[:,0]*((temp-350.)/100.)*pars[-1]
                    triple=(pair[i]+tails[i]-weight@(np.exp(-e2/temp)*np.expm1(np.clip(-potential/temp,-745,500))))*1e-12
                    vir.append([1e-6*base.second(g,temp,scales[g]),triple])
                for temp in temperatures:
                    i=grid.index(temp);vm2,vm,y,vp,vp2=np.array(vir[i-2:i+3])
                    coeffs=np.stack((y,(vm2-8*vm+8*vp-vp2)/12,(-vm2+16*vm-30*y+16*vp-vp2)/12))
                    for row in (r for r in original if r['gas']==g and r['T_K']==temp):
                        p=row['bar']*1e5
                        B,C=coeffs[0];BT,CT=coeffs[1];BTT,CTT=coeffs[2]
                        ideal=p/(R*temp)
                        candidates=np.roots([C*ideal**2,B*ideal,1,-1]);rho=min(r.real*ideal for r in candidates if abs(r.imag)<1e-10 and r.real>0 and 1+2*B*r.real*ideal+3*C*(r.real*ideal)**2>0)
                        u=-R*temp**2*(BT*rho+.5*CT*rho**2)
                        cvres=-R*temp*((2*BT+temp*BTT)*rho+.5*(2*CT+temp*CTT)*rho**2)
                        cv=1.5*R+cvres;cvref=row['reference_Cv'];vr=cvref-1.5*R
                        cases.append(dict(gas=g,K=temp,bar=row['bar'],jets_viriels=coeffs.tolist(),reference_rho=row['reference_density'],reference_Ures=row['reference_Ures'],rho=rho,Cv_J_mol_K=cv,Cv_ref=cvref,Cv_res=cvres,Cv_res_ref=vr,Ures_J_mol=u,error_Cv=cv/cvref-1,error_Cv_res=cvres/vr-1,error_Ures=u/row['reference_Ures']-1,error_density=rho/row['reference_density']-1))
            records[f'{order}_{int(isthermal)}']=dict(rows=cases,maxima_pct={k:100*max(abs(r[k]) for r in cases) for k in ('error_density','erreur_Ures','erreur_Cv','erreur_Cv_res')})
            print(order,isthermal,records[f'{order}_{int(isthermal)}']['maxima_pct'],flush=True)
    save('CALORIMETRIE_ET_VIRIELS_R573.json',dict(meme_law_of_size=scales,results=records,scope='36 états Kr/Xe, 400, 450 et 500 K, six pressions entre 3 et 47 bar. Références d’équation d’état HEOS du banc archivé R442. Les lois de paire et à trois corps R569 sont conservées avec leur dépendance thermique. Jets symétriques à cinq températures séparées de 1 K. Cv total et excès sur 3R/2 publiés séparément.'))
if __name__=='__main__':main()
