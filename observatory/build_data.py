"""Extract observer inputs from the verified benchmark and recompute mode projections."""
from pathlib import Path
from zipfile import ZipFile
import argparse, csv, io, json, sqlite3
import numpy as np

HERE=Path(__file__).resolve().parent
def values(array):return np.asarray(array).round(10).tolist()
def materialize(staged, output):
    root=Path(staged)
    result={}
    def table(path):
        with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    db_path=root/'references/R573/entrees/ATLAS_IONS_ISOTOPES_R557.sqlite'
    with sqlite3.connect(db_path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        elements=db.execute('SELECT Z,symbol,period,neutral_total_archived,neutral_total_reconstructed_candidate,surface_nodes,positive_modes,omega_min,omega_max FROM ELEMENTS ORDER BY Z').fetchall()
        isotopes=db.execute('SELECT Z,A,state_label,stable,half_life_seconds,half_life_estimated,mass_estimated,physical_wave_count,mass_neutral_u,spin_parity,excitation_keV FROM NUCLIDES_NUBASE2020 ORDER BY Z,A,state_index').fetchall()
        ions=db.execute('SELECT Z,q,electrons_external,cumulative_ionization_eV,physical_wave_count FROM ETATS_DE_CHARGE ORDER BY Z,q').fetchall()
        beta=db.execute('SELECT parent,daughter,lifetime_observed_s FROM TRANSITIONS_BETA WHERE parent_q=0 ORDER BY A').fetchall()
        assert (len(elements),len(isotopes),len(ions))==(118,5842,7139)
        assert all(r[7] is None for r in isotopes) and all(r[4] is None for r in ions)
    result.update(elements=elements,isotopes=isotopes,ions=ions,beta=beta)
    local=root/'redaction/heritage_atomique'
    with np.load(next(local.glob('*atomic_bases*.npz'))) as bases,np.load(next(local.glob('*identity_operators*.npz'))) as ops:
        atom_modes={}
        for z in range(37,119):
            key=f'Z{z:03d}'
            shell=bases[key+'_shell']; h=ops[key+'_contact_load_identity_operator']
            eigen,basis=np.linalg.eigh(h); positive=np.flatnonzero(eigen>1e-10)
            j=positive[0]
            mask=np.isclose(eigen,eigen[j],atol=1e-12,rtol=1e-11)
            space=basis[:,mask]
            # Project the first supported coordinate onto the whole eigenspace.
            # This fixes the display orientation even if LAPACK rotates a degenerate basis.
            for coordinate in range(len(h)):
                projected=space @ space[coordinate,:]
                magnitude=np.linalg.norm(projected)
                if magnitude>1e-6:
                    v=projected/magnitude
                    break
            else:
                raise ValueError(f'Empty positive eigenspace for Z={z}')
            assert np.linalg.norm(h@v-eigen[j]*v)<1e-9
            atom_modes[elements[z-1][1]]={'points':values(shell),'mode':values(v),'eigenvalue':float(eigen[j]),'multiplicity':int(mask.sum())}
        result['atoms']=atom_modes
    with ZipFile(root/'references/BST_R573_AUDIT_ET_COURBES.zip') as z:
        def data(p):return json.loads(z.read('BST_R573/'+p+'.json'))
        state=data('entrees/ETAT_SF6_PARENT_R480'); geom=data('resultats/GEOMETRIE_ET_PORTS_R573')
        with np.load(io.BytesIO(z.read('BST_R573/entrees/OPERATEURS_SF6_PARENT_R480.npz'))) as op:
            freq=np.sqrt(op['omega_squared']);v=op['modes']
            np.testing.assert_allclose(v.T@op['M']@v,np.eye(66),atol=1e-8)
            np.testing.assert_allclose(op['H']@v,op['M']@v*freq**2,atol=1e-6)
            result['fluoride']={'base':state['chart']['contact_base'],'frame':state['chart']['contact_frame'],'angles':state['coordinates'][48:60],'radii':state['radii'],'scale':float(np.mean(np.asarray(geom['distances_SF_A'])/state['radii'])),'vectors':values(v[48:66].T),'frequencies':values(freq)}
        fields=data('entrees/preuves/OBSERVATEURS_CHAMPS_R446')
        result['field']=[{'t':[e['t'] for e in case['lectures']],
                          'directions':[p['direction'] for p in case['lectures'][0]['projecteur_exterieur']],
                          'powers':[[p['puissance'] for p in e['projecteur_exterieur']] for e in case['lectures']],
                          'current':[e['courant_integre'] for e in case['lectures']]} for case in fields['cas']]
        result['wavenumbers']=fields['nombres_onde_natifs']
        try:
            from .projection_inputs import additional_inputs
        except ImportError:
            from projection_inputs import additional_inputs
        with sqlite3.connect(db_path.resolve().as_uri()+'?mode=ro',uri=True) as catalogue:
            result['projectionInputs'] = additional_inputs(catalogue, z, result)
        phases=data('resultats/PHASE_COMMUNE_R573')
        result['thermal']={branch:[[e['gaz'],e['T_K'],e['calcule'],e['reference']] for e in record['phase']] for branch,record in phases['branches'].items()}
        with ZipFile(io.BytesIO(z.read('BST_R573/entrees/PTB_RAW.zip'))) as ptb:
            spectra={}
            for p in ptb.namelist():
                if '225.5Torr' not in p or not p.endswith('.dat'):continue
                w,y,u=np.loadtxt(io.BytesIO(ptb.read(p)),skiprows=2,unpack=True)
                # Preserve extrema in each displayed bin. Raw samples are retained in the inputs.
                groups=np.array_split(np.arange(len(w)),min(180,len(w)))
                a=sorted({int(g[j]) for g in groups for j in [np.argmin(y[g]),np.argmax(y[g])]})
                spectra[str(round(w[0]))]=[[round(float(w[j]),4),float(y[j])] for j in a]
            result['infrared']=spectra
        optical=data('entrees/CALIBRATION_OPTIQUE_R568')
        obs=optical['controle_spectral']['aire_par_sommet']['forme_commune']['observations']
        result['optical']=[r for r in obs]
        line_source=data('entrees/R531/entrees/RAIES_H_MESUREES_R502')
        result['hydrogen']=[[r['states'],r['frequency_Hz'],r['uncertainty_Hz']] for r in line_source['held_out_from_scale_fit'] if '4P' in r['label']]
        joint=data('resultats/ASSEMBLAGES_68_R573')
        result['jointIds']=joint['ordre_des_clauses']
        result['joint']=[[r['identifiant'],r['cosmologie'],{'aire_par_sommet':'area','forme_reduite':'shape'}[r['optique']],int(r['thermique']),{'1995_branche4':'branch-four','deux_preparations':'two-preparations'}[r['fluorures']],{'charge_unitaire':'unit','Rb':'Rb','Cs':'Cs'}[r['port']],r['metrique'],''.join(r['codes'])] for r in joint['assemblages']]
        assert sum(all(c not in '?X' for key,c in zip(result['jointIds'],r[-1],strict=True) if key!='J00.1') for r in result['joint'])==192
        result['neon']=data('entrees/PREDICTIONS_GELEES_R572')['predictions_gelee_avant_cette_revue']['familles']
    result['sky']={}
    for r in table(root/'cosmologie/BANDES_PLANCK.csv'):
        result['sky'].setdefault(r['famille'],{}).setdefault(r['canal'],[]).append([float(r[k]) for k in ('ell','observed','sigma','predicted','official_binned_bestfit')])
    assert sum(len(v) for s in result['sky'].values() for v in s.values())==2376
    with ZipFile(root/'references/BST_R178_GIFS_MAC.zip') as z:
        modes={}
        for symbol in ('H2O','CO2','Ar','Kr','Xe'):
            with np.load(io.BytesIO(z.read(f'BST_GIFS_MAC/donnees/SCENE_{symbol}.npz'))) as a:
                if symbol in ('H2O','CO2'):
                    modes[symbol]={'points':values(a['positions']),'currents':values(a['current_modes']),'frequencies':values(a['frequency_ratios'])}
                else:
                    slow=int(np.argmin(a['omega_native']))
                    modes[symbol]={'points':values(a['positions']),'mode':values(a['modes'][slow]),'omega':float(a['omega_native'][slow])}
        result['modalScenes']=modes
    result['baselineBeat']=[[float(r[k]) for k in ('t','d','b','recouvrement','energie')] for r in table(root/'redaction/donnees/Lecture_deux_battements.csv')][:201]
    Path(output).write_text('window.BST_DATA = '+json.dumps(result,separators=(',',':'),allow_nan=False)+';\n')
    return result
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('staged',type=Path)
    parser.add_argument('output',type=Path,nargs='?',default=HERE/'data.js')
    args=parser.parse_args();materialize(args.staged,args.output)
