'Pressure, co-existence and heat resulting from a single free energy per gas branch.'
import json
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.constants import R
from CoolProp.CoolProp import PropsSI
import phase_native as phase
from readout_ir import I,O,decode,save

def main():
    parent=json.loads((I/'preuves/RACCORD_PHASE_R441.json').read_text())
    raw=json.loads((O/'VIRIELS_ETENDUS_R573.json').read_text())['calculs']
    q=json.loads((O/'AJUSTEMENT_ENERGIE_LIBRE_R573.json').read_text())['branches']
    cal=json.loads((O/'CALORIMETRIE_ET_VIRIELS_R573.json').read_text())['results']
    radii=decode('RAYONS_R569.json')['radii_recalcules']
    original=phase.values
    res={}
    for m in ['0','1']:
        a,b=q[m][0]['coefficients']
        curves={}
        for g,v in raw.items():
            temp,B,C=np.array(v[m]).T
            curves[g]=CubicSpline(temp,np.column_stack((B,C)),axis=0)
        def jet(g,T):
            B,C=curves[g](T);delta=(a+b*np.log(radii[g]))*(400./T-1)**2
            return [T,B+delta*1e-6,C,0.]
        def jets(g,T):return [jet(g,T+t) for t in (-.08,-.04,0,.04,.08)]
        def shape(g,j,y):
            T,B,C,_=j;tc,rc,_=phase.CRITICAL[g]
            u=np.clip((T/tc-1.05)/(.32),0,1);w=u**3*(10+u*(-15+6*u))
            vd=np.array([B*rc*y+.5*C*(rc*y)**2,B*rc+C*rc**2*y,C*rc**2])
            if w==1:return vd
            return (1-w)*original(g,j,y)+w*vd
        phase.values=shape
        sat=[]
        for row in parent['phase']:
            g,T=row['gas'],row['T_K']
            if g not in raw:continue
            out=phase.phase(dict(gas_field=g,jet=jets(g,T)))
            out['reference']=row['reference'];out['role']=row['role']
            out['deviations']=(np.array(out['computed'])/row['reference']-1).tolist()
            sat.append(out)
        gases=[]
        for tr in parent['gas_transition']:
            g,T=tr['gas'],tr['T_K']
            if g not in raw:continue
            out=phase.gas(dict(gas_field=g,T_K=T,jet=jets(g,T),mass=PropsSI('molar_mass',{'Kr':'Krypton','Xe':'Xenon'}[g])),parent['pressures_transition_bar'])
            gases.append(dict(gas_field=g,T=T,deviations=(np.array(out)/np.array(tr['reference'])-1).tolist()))
        heat=[]
        for row in cal['25_'+m]['rows']:
            g,T,bar=row['gas'],row['K'],row['bar'];rr=phase.CRITICAL[g][1];js=jets(g,T);p=bar*1e5;yi=p/(R*T*rr)
            y=brentq(lambda yy:phase.pressure_mu_slope(g,js[2],yy)[0]-yi,.5*yi,2.2*yi)
            v=np.array([phase.values(g,j,y)[0] for j in js]);h=.04
            d=(v[0]-8*v[1]+8*v[3]-v[4])/(12*h);dd=(-v[0]+16*v[1]-30*v[2]+16*v[3]-v[4])/(12*h*h)
            u=-R*T*T*d;c=-R*T*(2*d+T*dd);vals=[rr*y,u,1.5*R+c,c]
            ref=[row['reference_rho'],row['reference_Ures'],row['Cv_ref'],row['Cv_res_ref']]
            heat.append(dict(gas_field=g,T=T,bar=bar,deviations=(np.array(vals)/ref-1).tolist()))
        aux=[]
        for g in raw:
            name={'Kr':'Krypton','Xe':'Xenon'}[g]
            for t in (1.28,1.30,1.32,1.34,1.36):
                T=phase.CRITICAL[g][0]*t;bars=[1.,11.,21.,31.,41.]
                out=phase.gas(dict(gas_field=g,T_K=T,jet=jets(g,T),mass=PropsSI('molar_mass',name)),bars)
                reference=[[PropsSI(k,'T',T,'P',bar*1e5,name) for k in ('Dmolar','Cvmolar','Cpmolar','isothermal_compressibility','speed_of_sound')] for bar in bars]
                aux.append(dict(gas_field=g,T=T,deviations=(np.array(out)/reference-1).tolist()))
        def maxi(rows):return 100*np.max([np.max(np.abs(x['deviations'])) for x in rows])
        by={k:float(maxi(val)) for k,val in [('phase',sat),('phase_reserve_parent',[x for x in sat if x['role']=='reserve']),('gas_transition',gases),('calorimetry',heat),('junction_50_states',aux)]}
        res[m]=dict(phase=sat,states_gas_intermediaires=gases,calorimetry=heat,check_junction=aux,maxima_pct=by,maximum_defect_Clapeyron_pct=100*max(abs(x['error_clapeyron']) for x in sat),passe=max(by.values())<10)
        print(m,by,flush=True)
    save('PHASE_COMMUNE_R573.json',dict(branches=res,construction='Énergie de Helmholtz. Partie diluée définie partout par les B2 et B3 du potentiel R569 et le correctif de R573. En phase dense, énergie réduite de référence argon et atténuation en densité R441 ; transition quintique commune sur T/Tc = 1,05 à 1,37 vers la loi des viriels. Paramètres de transition choisis pour cet essai. Pression et potentiel chimique égaux, pentes positives, contrôle de la tangente commune et de Clapeyron. HEOS est le comparateur thermodynamique. Aucune viscosité n’est calculée par cette énergie libre.'))
if __name__=='__main__':main()
