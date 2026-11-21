'The same electric port in atomic, nuclear and stellar computation; conversion between molecular reductions.'
import json,sys,math
from pathlib import Path
import numpy as np
from scipy.constants import hbar,Avogadro,c,h
from scipy.linalg import eigvalsh
from readout_ir import I,O,decode,save
sys.path.insert(0,str(I/'R531'))
import intersection_intervals as atom
def charge():
    cfg=decode('INTERSECTION_R531.json');historical=decode('H_D_METRIQUES_R559.json')
    mu=historical['mu_over_me'];phi=decode('mesures_R559.json')
    a0=.3**2/(4*math.pi);ans={}
    # Replay the original imprints at the same action and retain the instrumental margins of the auxiliary parameters.
    for name,a in [('charge_unitaire',a0)]+[(s['standard'],s['alpha']) for s in historical['two_standards']]:
        base=cfg['center'].copy();base['alpha']=a
        box=atom.make_box(base,.004)
        box['alpha']=[a,a]
        res=atom.bounds(box,strict=True)
        # The electric port interval is a diagnosis R502. Q remains unitary in the first test.
        rr=math.sqrt(4*math.pi*a)/.3;norm=4*math.pi*a/.3**2
        angular=res['maxima'].copy()
        max_lines=max(max(abs((1+z)*mu['1']-1) for z in s['error_interval']) for s in res['spectra'])
        fine=max(abs((1+x)*mu['1']-1) for x in res['fine_error_interval'])
        angular['H-frequencies']=max_lines;angular['Fine separation H']=fine
        scale=atom.me/(2*math.pi*atom.hb)
        delta_H=scale*atom.gap((1,1),(2,1),a,math)*(mu['2']-mu['1'])
        loc=scale*atom.gap((2,1),(3,2),a,math)*mu['1']
        clocks={}
        for x in historical['four_controls_spectrum_clock'][:2]:
            clock=x['frequency_Halpha_distant_ideale_Hz']/x['frequency_Halpha_locale_Hz']
            clocks[x['metric']]=dict(lapse=clock,Halpha_distant_Hz=loc*clock,cz_km_s=c*(1/clock-1)/1000)
        ans[name]=dict(alpha=a,G=.3,Q_si_S0_over_hbar_1=rr,S0_over_hbar_si_Q_1=norm,
            intervals_parameters=box,doublet_and_others_groups=angular,maximum=max(angular.values()),
            count_bonds=len(res['bulk']),displacement_HD_computed_Hz=delta_H,HD_error_relative=delta_H/historical['observation_Hz']-1,
            HD_residual_Hz=delta_H-historical['observation_Hz'],HD_u_measured_Hz=historical['u_observation_Hz'],
            metrics=clocks,maximum_brut_sans_mass_reduced=res['maximum'],Q_in_band_diagnostic_R502=bool(.9830730952207916<=rr<=1.0336485431383011))
    return ans
def affine(y,x):
    slope=(y[1]-y[0])/(x[1]-x[0]);return np.array([y[0]-slope*x[0],slope])
def scales():
    units=decode('UNITES_H2O_R524.json')
    em=units['energy_unit'];tm=units['time_unit']
    res=json.loads((O/'SPECTRE_ET_THERMOCHIMIE_R573.json').read_text())['families']
    rf=decode('RACCORD_FRAGMENTS_R481.json');sf=decode('CONTACT_LIBRE_ET_CYCLE_R571.json')
    hs=np.asarray(rf['common_hbar_interval']);old=rf['branches']['SF6_2']['cluster_zpe']
    delta=next(x['differentiel_energy_classique'] for x in decode('MODES_CONTRAINTS_R571.json') if x['indice']==3)
    cycle0=decode('INTERVALLES_SOURCES_R482.json')['upper_limits_and_formation_cycle']['SF4_prepare_quatre_ports']['cycle_native_endpoints']
    cycle1=decode('BRANCHES_MOLECULAIRES_R485.json')['rows'][4]['formation_cycle_native']
    ans={}
    for key,gr in res.items():
        fit=gr['best_par_n_of_coefficients']['4'];C=fit['C_kcal_mol'];hb=fit['hbar_native'];k=fit['stiffness_over_k0']*sf['k0']
        with np.load(I/'OPERATEURS_SF5_LIBRES_R571.npz') as z:
            w=np.sqrt(eigvalsh(z['H']+k*np.outer(z['grad_g'],z['grad_g']),z['M']))
        xx=C*np.array([1.,hb]);slope=.5*w.sum()-old
        d6=float((affine(rf['thresholds_common_free_endpoints']['SF6'],hs)+[delta+sf['lambda_action_native']**2/(2*k),slope])@xx)
        d4=float(affine(rf['thresholds_common_free_endpoints']['SF4'],hs)@xx)
        cycles=[float(affine(cy,hs)@xx) for cy in (cycle0,cycle1)]
        e=C*4184/Avogadro;t=hbar/(e*hb)
        assert abs(1/(2*math.pi*c*100*t)/fit['gain']-1)<1e-12
        assert 101.5951<d6<113.0675 and 0<d4<=87.1
        assert all(138.3-1e-8<=v<=150.3+1e-8 for v in (cycles if key=='two_preparations' else cycles[1:]))
        if key=='1995_branche4':assert 104.6845<=d6<=107.5526 and 36.3288<=cycles[1]-d6<=39.675
        ans[key]=dict(stiffness_over_k0=fit['stiffness_over_k0'],C_kcal_mol=C,hbar_native=hb,E_J=e,t_s=t,ratio_energy_SFx_over_H2O=e/em,ratio_time_SFx_over_H2O=t/tm,
            ratio_unit_action_SFx_over_H2O=e*t/(em*tm),gain_computed_si_meme_unit_time_H2O=1/(2*math.pi*c*100*tm),
            D4=d4,D6=d6,cycles=cycles,D5=[s-d6 for s in cycles],gain_cm1=fit['gain'])
    return dict(action_H2O=units,hbar_native_H2O=hbar/(em*tm),actions_SFx=ans,
        convention='Assumption tested: reuse without conversion a unit of time and energy between two documented reductions. The change of diagonal units above connects their SI stallions by construction. It does not yet express the variation of the two units as a function of a resolution of the substrate measured independently. Both scenarios retain the same status of the SF4–SF5–SF6 family each.')
if __name__=='__main__':
    out=charge();save('RECALCUL_CHARGE_COMMUNE_R573.json',out)
    unit=scales();save('UNITES_ET_RUPTURE_R573.json',unit)
    print({k:(v['maximum'],v['HD_erreur_relative']) for k,v in out.items()})
    print(unit['actions_SFx'])
