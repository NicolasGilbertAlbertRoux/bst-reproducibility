'Compare the instantaneous port to an anti-resonance port, to a single setting for all modes.'
import json,csv
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import null_space
from readout_ir import I,O,decode,save,WINDOWS,CORE
from crosscheck_spectrum import contact_readers
def main():
    s=json.loads((O/'SPECTRE_ET_THERMOCHIMIE_R573.json').read_text())
    weak=decode('CALIBRATION_SF6_R570.json')
    records={};w,B=contact_readers()
    template=list(csv.DictReader((I/'SPECTRES_SF6_R570.csv').open(encoding='utf-8-sig')))
    for scenario,g in s['families'].items():
        fit=g['best_par_n_of_coefficients']['4'];gain=fit['gain'];freq=gain*w
        first=(freq>=1100)&(freq<=1160);idx=np.flatnonzero(first)
        assert len(idx)==1
        # A second-order causal linear filter acts on the fundamental channel.
        # H(s)-(s2-omega02)/(s2-2-zetaomega0-somega02). It has a zero,
        # poles in the stable half-plane, and H(i omega) is 1.
        # The combination channel R570 is a separate channel, calibrated on data of the same pressure.
        terms=np.einsum('a,aim->im',fit['coefficients'],B)
        unfiltered=np.sum(terms**2,axis=0)
        targets=np.array([[v[b][reg]['area'] for b,reg in s['keys_aires']] for v in s['measurements'].values()])
        masks=np.array([(freq>= (WINDOWS if reg=='region' else CORE)[b][0])&(freq<=(WINDOWS if reg=='region' else CORE)[b][1]) for b,reg in s['keys_aires']])
        def transfer(om,ratio,zeta):
            d=1-(om/ratio)**2;return d*d/(d*d+(2*zeta*om/ratio)**2)
        widths=[.01,.025,.05,.1]
        entries=[]
        for zeta in widths:
            out=unfiltered*transfer(w,w[idx[0]],zeta)
            rr=(masks@out)/targets;cal=2/(rr.max()+rr.min());rr*=cal;out*=cal
            centers=[float(out[m]@freq[m]/sum(out[m])) for m in masks[[0,2]]]
            peak=1.e-19*out[first].sum();profile={}
            for p in ['225.53','745.66']:
                a=[r for r in template if r['pressure_Torr']==p];ux=np.array([float(r['U_k2']) for r in a]);x=np.array([float(r['nombre_onde_cm-1']) for r in a])
                y=np.array([float(r['section_PTB_cm2_par_molecule']) for r in a]);pred=np.array([float(r['lecture_G']) for r in a])+peak*np.exp(-.5*((x-freq[idx[0]])/.12)**2)/(np.sqrt(2*np.pi)*.12)
                err=np.abs((pred-y)/ux)
                profile[p]=dict(points=len(x),in_U=int(sum(err<=1)),maximum_over_U=float(max(err)))
            entries.append(dict(zeta=zeta,multiplier_amplitude=float(np.sqrt(cal)),deviation_huit_aires_pct=float(np.max(abs(rr-1))*100),two_centers_cm1=centers,aires=(masks@out).tolist(),profile_weak=profile,
              retention_8_aires_and_6970_points=bool((np.max(abs(rr-1))<=.1) and all(x['in_U']==x['points'] for x in profile.values()))))
        # Sensitivity of the depth of cancellation to the zero position; non-enlarged node not assimilated to an infinitely fine experimental line.
        chosen=next(x for x in entries if x['zeta']==.025);strength=unfiltered[idx[0]]*1e-19*chosen['multiplier_amplitude']**2
        # Punctual profile margin by adding a Gaussian line width of the R570 instrument.
        lim=[]
        for p in ['225.53','745.66']:
            aa=[r for r in template if r['pressure_Torr']==p];x=np.array([float(v['nombre_onde_cm-1']) for v in aa]);ix=abs(x-freq[idx[0]])<1.
            margin=np.array([float(v['section_PTB_cm2_par_molecule'])+float(v['U_k2'])-float(v['lecture_G']) for v in aa])[ix]
            kernel=np.exp(-.5*((x[ix]-freq[idx[0]])/.12)**2)/(np.sqrt(2*np.pi)*.12)
            lim.append(float(np.min(margin/kernel)))
        eta=min(lim)/strength
        t=.025*np.sqrt(eta/(1-eta));ratio=t+np.sqrt(1+t*t)
        records[scenario]=dict(gain=gain,mode=int(idx[0]+1),zero_cm1=float(freq[idx[0]]),zero_native=float(w[idx[0]]),zeta_retained=.025,interval_zero_point_cm1=[float(freq[idx[0]]/ratio),float(freq[idx[0]]*ratio)],area_of_mode_before_filter_cm_molecule=float(unfiltered[idx[0]]*1.e-19),area_weak_PTB_cm_molecule=weak['externe']['PTB_value_combinee'][0],trials=entries)
    save('REPONSE_COMMUNE_TROIS_BANDES_R573.json',dict(families=records,
        hypothese='The same causal dynamic port for fundamental transitions, a zero and a damping common to the 66 modes. The zero is calibrated in the mode which would fall into the weak band to the wrongly weak band; its permissible variation is quantified with fixed depreciation. The combination profile is that of R570 with its funds and its two pressures. Effective solution conditioned to an anti-resonance; its origin in the geometric dynamics and the intrinsic width of the dark line remain to be drawn. No Raman spectrum calculated by this reader.'))
    print([(k,[(v['zeta'],v['deviation_huit_aires_pct'],v['retention_8_aires_and_6970_points']) for v in vals['trials']],vals['interval_zero_point_cm1']) for k,vals in records.items()])
if __name__=='__main__':main()
