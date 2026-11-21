'Recalculate the readings of the TSB sectors and the two scripts of the phase action.'
import argparse
import csv
import io
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "references"
REGISTER = REFERENCES / "R573/AUDIT_DES_68_LIGNES_R573.csv"
ORDER = "D01 D11 D02 D03 D04 D05 D06 D09 D07 D08 D10".split()


def fiche(archive, path):
    return json.loads(archive.read("BST_R573/" + path + ".json"))


def nuclear(archive):
    base = REFERENCES / "R573/entrees/ATLAS_IONS_ISOTOPES_R557.sqlite"
    with sqlite3.connect(base.resolve().as_uri() + "?mode=ro", uri=True) as db:
        def row(query):
            return db.execute(query).fetchone()
        atlas = {
            "masses": row("SELECT COUNT(*) FROM MASSES_AME2020")[0],
            'states_nucleaires': row("SELECT COUNT(*) FROM NUCLIDES_NUBASE2020")[0],
            'counts_waves_attribues': row("SELECT COUNT(physical_wave_count) FROM NUCLIDES_NUBASE2020")[0],
            'states_charge': row("SELECT COUNT(*) FROM ETATS_DE_CHARGE")[0],
            'counts_waves_ions_attribues': row("SELECT COUNT(physical_wave_count) FROM ETATS_DE_CHARGE")[0],
            'transitions_durations_lues_derivatives': row(
                "SELECT COUNT(*),COUNT(lifetime_observed_s),COUNT(lifetime_derived_s) FROM TRANSITIONS_BETA"),
            "cycles_defaut_max_eV": row("SELECT COUNT(*),MAX(ABS(loop_defect_eV)) FROM CYCLES_COMPTABLES"),
            "Sr_base_element": row("SELECT neutral_total_archived,surface_nodes,positive_modes FROM ELEMENTS WHERE Z=38"),
            "Sr_etats_base_reference_distincte": row(
                "SELECT COUNT(*),COUNT(DISTINCT neutral_reference_total) FROM NUCLIDES_NUBASE2020 WHERE Z=38"),
            "Sr87": db.execute("SELECT state_label,stable,excitation_keV,half_life_seconds,neutral_reference_total,"
                               "physical_wave_count FROM NUCLIDES_NUBASE2020 WHERE Z=38 AND A=87 ORDER BY state_index").fetchall(),
        }
    data = fiche(archive, "entrees/RACCORD_NUCLEAIRE_R558")
    beta = data["beta_by_state"]
    # The connector accepted in R559 reproduces the amplitudes by state of the certificate R558.
    accord = {isotope: {
        'band_suffisante_abs_M': beta[isotope]["parent_neutre"]["sufficient_abs_overlap_for_all_H531"],
        'bounds_error_relative': beta[isotope]["parent_neutre"]["relative_error_at_all_amplitudes_and_nuisances"],
    } for isotope in ("He-6", "C-14")}
    with REGISTER.open(encoding="utf-8-sig", newline="") as f:
        d112 = next(row for row in csv.DictReader(f) if row['row'] == "D11.2")
    o19 = fiche(archive, "entrees/BRANCHES_O19_R558")
    phase = np.array(o19['f2_over_f1'])
    control = o19['control'] ** 2 * phase
    np.testing.assert_allclose(control, o19['ratio_of_branches_computed_at_control'])
    assert control[0] >= o19["r_obs_B2_sur_B1"][0] and control[1] <= o19["r_obs_B2_sur_B1"][1]
    return {"atlas": atlas, "accord_par_etat_H531_parent_neutre": accord,
            'D11_2_register_R573': {key: d112[key] for key in (
                'status_de_portee_R573', 'combinations_computed', 'correspondence_indetermine', 'combinations_in_failure')},
            "O19_rapport_amplitudes": o19['rapport_M_suffisant_pour_la_box'],
            'O19_rapport_rate_au_temoin': control.tolist(),
            'O19_duree_partial_model_sur_observation': o19['verification_duration_partial_model_over_observation']}


def electromagnetism(archive):
    source = fiche(archive, "entrees/preuves/OBSERVATEURS_CHAMPS_R446")
    residuals, defects = [], []
    for cases in source['cases']:
        for window in cases['readouts']:
            for reader in window['projecteur_external']:
                direction = np.array(reader["direction"])
                np.testing.assert_allclose(direction @ direction, 1, atol=1e-14)
                amplitude = np.array(reader["amplitude_re"]) + 1j * np.array(reader["amplitude_im"])
                puissance = np.sum(abs(amplitude)**2, axis=1)
                np.testing.assert_allclose(puissance, reader["puissance"], rtol=3e-13, atol=3e-13)
                residuals.extend(abs(amplitude @ direction) / np.maximum(1, np.sqrt(puissance)))
                defects.extend(abs(puissance - reader["puissance"]) / np.maximum(1, puissance))
    assert max(residuals) < 1e-12
    return {"amplitudes": len(residuals), 'defect_transverse_normalized': float(max(residuals)),
            'defect_puissance_normalized': float(max(defects))}


def spectra(archive):
    source = fiche(archive, "resultats/REPONSE_COMMUNE_TROIS_BANDES_R573")['families']
    brut = archive.read("BST_R573/entrees/PTB_RAW.zip")
    data = {}
    with ZipFile(io.BytesIO(brut)) as z:
        for pressure, summary in (("225.53", "225.5"), ("745.66", "745.7")):
            name = f"SF6_296.1K_{summary}Torr_air_1100.0-1160.0_0.04.dat"
            x, sigma, uncertainty = np.loadtxt(io.BytesIO(z.read(name)), skiprows=2, unpack=True)
            assert np.all(np.diff(x) > 0) and np.all(uncertainty > 0)
            data[pressure] = {"points": len(x), 'area_signee_cm_par_molecule': float(np.trapezoid(sigma, x))}
    families = {}
    for name, bloc in source.items():
        trial = next(e for e in bloc['trials'] if e["zeta"] == bloc['zeta_retained'])
        for p, fiche_p in trial['profile_weak'].items():
            assert fiche_p['in_U'] == fiche_p["points"] == data[p]["points"]
        families[name] = {"zero_cm1": bloc["zero_cm1"], 'centers_cm1': trial['two_centers_cm1'],
                         'deviation_huit_aires_pct': trial['deviation_huit_aires_pct']}
    return {"PTB": data, 'families': families}


def mechanical(archive):
    f = fiche(archive, "resultats/FLUX_CORRIGE_R573")
    vacuum = np.array(f['energy_vide_continuous_sonde'])
    brut, corrected = {}, {}
    for measurement in f["estimations"]:
        n = measurement["N"]
        brut[n] = np.array(measurement['sept_centers'])
        corrected[n] = brut[n] - measurement["vide_meme_grille"] + vacuum
        np.testing.assert_allclose(corrected[n], measurement["estimation_meme_sonde"], atol=1e-17, rtol=0)
    def variations(series):
        return (100 * abs(series[64] - series[72]) / abs(series[72])).tolist()
    np.testing.assert_allclose(variations(corrected), f['change_corrected_64_72_pct'])
    return {'grids': sorted(corrected), "variations_brutes_pct": variations(brut),
            "variations_corrigees_pct": variations(corrected)}


def thermodynamics(archive):
    f = fiche(archive, "resultats/AJUSTEMENT_ENERGIE_LIBRE_R573")
    # Control status of argon radius derivatives in published units of B2.
    temperature, rho, rg = 400., .001, 8.314462618
    outputs = {}
    for name, branch in f["branches"].items():
        mode = branch[0]
        assert mode['powers'] == [2]
        a = mode["coefficients"][0]
        def free(t):
            return rg*t*rho*a*(400./t - 1)**2
        delta = .01
        cv_fd = -temperature*(free(temperature+delta)-2*free(temperature)+free(temperature-delta))/delta**2
        cv = -2*rg*rho*a
        np.testing.assert_allclose(cv_fd, cv, rtol=3e-8)
        outputs[name] = {"maxima_pct": mode["maxima_pct"], "delta_Cv_J_mol_K": cv,
                        'delta_Cv_difference_finite_J_mol_K': cv_fd}
    return outputs


def fluids(archive):
    f = fiche(archive, "resultats/BILAN_CISAILLEMENT_R573")
    intervals = []
    for gas in f['results']:
        for readout in gas['checks'].values():
            intervals.append(readout['interval_gain'])
    common = [max(x[0] for x in intervals), min(x[1] for x in intervals)]
    np.testing.assert_allclose(common, f['interval_gain_viscosite'])
    return {'branch': f['branch'], 'seeds': f['seeds'], "trajectoires": f["trajectoires"],
            'intervals': len(intervals), "intersection": common, 'gain_un_admitted': common[0] <= 1 <= common[1]}


def cavitation(archive):
    f = fiche(archive, "entrees/BILAN_CAVITE_R568")
    readouts = defaultdict(set)
    for measurement in f["cycles"]:
        readouts[(measurement["regime"], measurement["cycle"])].add(measurement['readout'])
    assert all(l == {"Mie", 'fitted'} for l in readouts.values())
    # What does a single average energy set, even if you have its exact value?
    n = len(readouts)
    mean = np.ones((1,n)) / n
    contrast = np.zeros(n)
    contrast[:2] = [.25, -.25]
    np.testing.assert_allclose(mean @ contrast, 0)
    return {"cycles_distincts": n, 'readouts': len(f["cycles"]), 'frequency_Hz': f["conditions"]['frequency_Hz'],
            'directions_libres_with_the_seule_mean': n - int(np.linalg.matrix_rank(mean)),
            'control_of_meme_mean': (1+contrast).tolist()}


def relativity(archive):
    f = fiche(archive, "entrees/preuves/HYDROGENE_SIRIUS")
    branches = {}
    for branch in f["sirius"]:
        cz = 299792.458 * (1/branch["N"]-1)
        np.testing.assert_allclose(cz, branch["cz_km_s"], atol=7e-9, rtol=0)
        branches[branch['name']] = {"cz_km_s": cz, 'intersection_main': branch["intersection_km_s"],
                                     "intersection_secondaire": branch["intersection_secondaire_km_s"]}
    return branches


def sky(archive):
    f = fiche(archive, "resultats/COSMOLOGIE_ET_OSCILLATIONS_R573")
    limits = f['thresholds_earlier']
    admitted = []
    for row in f['families']:
        ok = (max(row['chi2_cmb_par_band'].values()) < limits['CMB_par_band']
              and max(row["chi2_DESI"].values()) < limits["DESI_par_bloc"] and row["chi2_eBOSS"] < limits["eBOSS"])
        assert ok == row["compatible"]
        if ok:
            admitted.append(row['family'])
    return {'families': admitted, 'scores_bands': sum(len(x['chi2_cmb_par_band']) for x in f['families']),
            'closure_masse_eV': math.sqrt(f["delta_m2_eV2"]), 'thresholds_of_benchmark': limits}


def information():
    with (ROOT / "cognition/attendus/BALAYAGE_COGNITIF_R528.csv").open(encoding="utf-8-sig") as flux:
        rows = list(csv.DictReader(flux))
    pairs = defaultdict(list)
    for row in rows:
        preparation = tuple(row[k] for k in ("tag", "eta", "lam", "mass", "impulse"))
        pairs[preparation].append((int(row["sign"]), float(row["read_margin"])))
    assert all(sorted(s for s,_ in pair) == [-1,1] for pair in pairs.values())
    counts = Counter((int(l["sign"]),int(l["decoded_sign"])) for l in rows)
    def information_mutual(counts):
        total = sum(counts.values())
        pa, pb = Counter(), Counter()
        for (a,b),n in counts.items():
            pa[a] += n
            pb[b] += n
        return sum(n/total * math.log2(n*total/(pa[a]*pb[b])) for (a,b),n in counts.items())
    rest = Counter()
    for (a,b),n in counts.items():
        rest[(a,0)] += n
    margins = sorted(min(m for _,m in couple) for couple in pairs.values())
    errors = [0, 2, 3, 5, 10]
    return {"trajectoires": len(rows), 'pairs': len(pairs), "marge_minimale": min(margins),
            'bits_readout_constant': information_mutual(rest), 'bits_readout_sondee': information_mutual(counts),
            'pairs_guarantees': {str(e): sum(m > e for m in margins) for e in errors},
            'margins_par_pair_triees': margins}


def plot_information(summary, file):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    margins = np.asarray(summary['margins_par_pair_triees'])
    bounds = np.unique(np.r_[0, margins, 10, 1.03*margins.max()])
    guarantees = [np.count_nonzero(margins > threshold) for threshold in bounds]
    fig, ax = plt.subplots(figsize=(7.4, 3.0), layout="constrained")
    ax.step(bounds, guarantees, where="post", color="#136862", lw=2.3)
    for threshold, offset in ((2, (-33,-17)), (3, (9,7)), (10, (-42,8))):
        effective = summary['pairs_guarantees'][str(threshold)]
        ax.scatter([threshold], [effective], color="#93482d", s=23, zorder=3)
        ax.annotate(f"{effective} / 72", (threshold, effective), xytext=offset,
                    textcoords="offset points", fontsize=9, color="#723820")
    ax.set(xlabel='Error terminal on decision statistics (model units)',
           ylabel="Paires garanties", xlim=(0, bounds.max()+.15), ylim=(-3,78),
           yticks=[0,24,48,72])
    ax.grid(axis="y", alpha=.2)
    ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(file, dpi=160)
    plt.close(fig)


def action_of_phase():
    # Restriction to a large neutral circle of the S3, with three spatial directions on a torus.
    n, h = 12, 2*np.pi/12
    x, y, z = np.meshgrid(*([np.arange(n)*h]*3), indexing="ij")
    theta = .18*np.sin(x) + .09*np.cos(y) + .13*np.sin(z)
    derivative = -.36*np.cos(x) - .09*np.sin(y) - .13*np.cos(z)
    q = np.stack((np.sin(theta), np.cos(theta)), axis=-1)
    t = np.stack((np.cos(theta), -np.sin(theta)), axis=-1)
    tangents, differences = [], []
    for axis in range(3):
        neighbor = np.roll(q, -1, axis=axis)
        d = (np.roll(theta, -1, axis=axis)-theta)/h
        angle = np.arctan2(np.sum(neighbor*t, axis=-1),np.sum(neighbor*q, axis=-1))
        v = angle[...,None]*t/h
        np.testing.assert_allclose(v, d[...,None]*t, rtol=0, atol=8e-16)
        tangents.append(v)
        differences.append(d)
    w = derivative[...,None]*t
    vectors = [w, *tangents]
    def aire2(a,b):
        return np.sum(a*a,axis=-1)*np.sum(b*b,axis=-1)-np.sum(a*b,axis=-1)**2
    aires = [aire2(a,b) for i,a in enumerate(vectors) for b in vectors[i+1:]]
    error_area = max(float(np.max(abs(u))) for u in aires)
    assert error_area < 2e-16
    lag_field = (np.sum(w*w,axis=-1) - sum(np.sum(v*v,axis=-1) for v in tangents)
                 + sum(aires[:3]) - sum(aires[3:]))/2
    norm_gradient = sum(d*d for d in differences)
    lag_phase = (derivative**2-norm_gradient)/2
    np.testing.assert_allclose(lag_field, lag_phase, atol=6e-16, rtol=0)
    # Comparison of the action dimensioned on a slice with its standardized writing.
    chi, stiffness, length, dt = 3.2, 7.5, 2.3, .04
    speed = math.sqrt(stiffness/chi)
    action = np.sum(chi*derivative**2-stiffness*norm_gradient)*dt*h**3/2
    duration_reduced = speed*dt/length
    action_min = np.sum((length*derivative/speed)**2-length**2*norm_gradient)*(h/length)**3*duration_reduced/2
    action_unit = math.sqrt(chi*stiffness)*length**2
    np.testing.assert_allclose(action/action_unit, action_min, rtol=3e-14)
    return {"sites": n**3,'bound_aires_at_square': error_area,
            'deviation_max_densities': float(np.max(abs(lag_field-lag_phase))),
            'action_over_unit': float(action/action_unit),"action_minimale": float(action_min)}


def carte_of_clauses(archive, file):
    'Link each clause to its actual delivered parts and its reading control.'
    checks = {
        "D01": 'Paths d01. H-shares, six S-F distances, H–D displacement and two tests for Sr.',
        "D11": 'Base R557, Sr states and half-lives documented. Accord by condition He-6, C-14 and branches of O-19 (R558, R559, D11.2 of R573).',
        "D02": 'Transversality and power of the amplitudes of the reader R446 in reduced units.',
        "D03": 'Signed integration of the two PTB spectra of the weak band. Reading of the common adjustment.',
        "D04": 'Correction of the fluxes on five grids and comparison between 64 and 72.',
        "D05": 'Differentiation of the free energy at the radius of the argon at 400 K.',
        "D06": 'Intersection of the shear reading gain intervals, branch 0.',
        "D09": 'Eight distinct cycles under two readings of the radius. Indeterminacy with the average only.',
        "D07": 'Calculation of the Sirius B spectral shift for two funds from the lapses.',
        "D08": 'Joint Thresholds of the twelve families. The cosmological journey recalculates the Planck bands.',
        "D10": 'Mutual information and number of pairs which retain a numerical decoding margin.',
    }
    with REGISTER.open(encoding="utf-8-sig") as f:
        rows = [x for x in csv.DictReader(f) if x['row'].startswith("D")]
    all_records = set()
    with file.open("w", encoding="utf-8", newline="") as output:
        table = csv.writer(output)
        table.writerow(['row', "exigence", 'status_R573', 'finding_R573', 'under_condition_R573',
                        'records_in_BST_R573', 'check_of_readout_of_domaine'])
        for dom in ORDER:
            for x in (l for l in rows if l['row'].split(".")[0] == dom):
                records = x['records'].split(" | ")
                for p in records:
                    archive.getinfo("BST_R573/" + p)
                all_records.update(records)
                table.writerow([x['row'], x["exigence"], x['status_de_portee_R573'], x['finding'],
                                x['under_condition_scientific'], x['records'], checks[dom]])
    return [x['row'] for x in rows], len(all_records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    with ZipFile(REFERENCES/"BST_R573_AUDIT_ET_COURBES.zip") as archive:
        clauses, count_records = carte_of_clauses(archive, args.output/"Parcours_des_29_clauses.csv")
        summary = {'order': ORDER, "clauses_directes": clauses, 'records_distinct_of_register': count_records,
                 "D11": nuclear(archive),
                 "D02": electromagnetism(archive), "D03": spectra(archive), "D04": mechanical(archive),
                 "D05": thermodynamics(archive), "D06": fluids(archive), "D09": cavitation(archive),
                 "D07": relativity(archive), "D08": sky(archive), "D10": information(), "phase": action_of_phase()}
    assert len(clauses) == len(set(clauses)) == 29 and {c.split(".")[0] for c in clauses} == set(ORDER)
    (args.output/"Relecture_des_domaines.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    plot_information(summary["D10"], args.output/"Marge_de_lecture_H528.png")
    print(f"{len(clauses)} clauses directes, {len(ORDER)} domaines, {summary['D10']['pairs']} paires H528.")


if __name__ == "__main__":
    main()
