'Two measurements of the same energy transmission on each complete grid.'
import json
import numpy as np
from scipy.integrate import simpson
from readout_ir import I,O,decode,save
def main():
    sol=json.loads((O/'KIRCHHOFF_R573.json').read_text());sharp=np.array(sol[-1]['derivative_local']);probe=np.array(sol[-1]['sonde_native'])
    ref=np.array(decode('CONTROLE_EXTERIEUR_LOCAL_R572.json')['energy_last_step'])
    rows=[]
    for n in (40,48,56,64,72):
        name=f'MOUVEMENT_N{n}_impedance_dt0.09.json';file=next(f for f in (O/name,I/name) if f.exists());data=json.loads(file.read_text())
        aux=json.loads((O/f'VIDE_N{n}_dt.09_W13.json').read_text());wave=np.array(data['energy_nette_two_surfaces']);base=np.array(aux['energy'])
        t=np.array([r['t'] for r in data['historique']['wave']])
        independent=simpson(np.array([x['currents'] for x in data['historique']['wave']])-np.array([x['currents'] for x in data['historique']['background']]),x=t,axis=0)
        assert np.max(abs(independent-wave))<1e-15
        delta=wave-base;gain=2/(max(wave/ref)+min(wave/ref))
        body=np.array([x['positions'] for x in data['historique']['wave']])-np.array([x['positions'] for x in data['historique']['background']])
        rows.append(dict(N=n,sept_centers=wave.tolist(),vide_meme_grille=base.tolist(),correction_molecular=delta.tolist(),estimation_meme_sonde=(probe+delta).tolist(),estimation_derivative_local=(sharp+delta).tolist(),gain_minimax_face_common=gain,
             interval_gain_a_10pct=[max(.9*ref/wave),min(1.1*ref/wave)],error_of_gain_pct=100*max(abs(gain*wave/ref-1)),
             motion_max_of_center_sulfur=float(np.linalg.norm(body[:,0],axis=1).max()),motion_max_par_center=np.linalg.norm(body,axis=2).max(axis=0).tolist()))
    def group(key):return np.array([r[key] for r in rows])
    def rel(y,x):return (100*abs(y/x-1)).tolist()
    spread=(np.ptp(group('estimation_meme_sonde'),axis=0)/group('estimation_meme_sonde')[-1]*100).tolist()
    ans=dict(estimations=rows,energy_vide_continuous_sonde=probe.tolist(),energy_vide_continuous_derivative=sharp.tolist(),flux_fin_3D_over_trajectory_R572=ref.tolist(),
             range_corrected_over_five_grids_pct=spread,
             change_brut_64_72_pct=rel(group('sept_centers')[-2],group('sept_centers')[-1]),
             change_corrected_64_72_pct=rel(group('estimation_meme_sonde')[-2],group('estimation_meme_sonde')[-1]),
             deviation_at_field_fin_3D_a_N72_pct=rel(group('estimation_derivative_local')[-1],ref),
             last_deviation_quadratures_sonde_pct=rel(probe,np.array(sol[-2]['sonde_native'])),
             last_deviation_quadratures_derivative_pct=rel(sharp,np.array(sol[-2]['derivative_local'])),
             formula='Ecorr(N) - E7(N) - Evil(N) - Evid(continu). Same coordinates, same pulse, same dissipative edge and same probes for the first two terms. The residue contains the seven centres in reaction. The last integral is obtained by Kirchhoff for the initial data on the large circle, in two reading conventions.',
             scope='Propagation sur deux faces entre t=0 et 0,36, énergie native. Convergence observée de l’estimateur avec correction de discrétisation du paquet. Le champ fin à quatre composantes R572 sur la trajectoire imposée sert de comparaison distincte. La correction est appliquée à la lecture et ne réinjecte pas un champ fin dans les sept équations. Les variations chiffrées ne sont pas une borne universelle.')
    save('FLUX_CORRIGE_R573.json',ans);print(spread,ans['change_corrected_64_72_pct'],ans['difference_au_champ_fin_3D_a_N72_pct'])
if __name__=='__main__':main()
