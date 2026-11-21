'Reconstruct the contact bank and test two readings of the strontium clock.'
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "redaction/donnees"
HERITAGE = ROOT / "redaction/heritage_atomique"
BASES = "v92_r1_upper_atomic_bases_Z37_Z118.npz"
OPERATORS = "v94_r9_upper_contact_load_identity_operators_Z37_Z118.npz"
ATLAS = "v94_r9_upper_contact_load_identity_operator_atlas_Z37_Z118.csv"


def read_field(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(name):
    with (HERITAGE / name).open(encoding="utf-8-sig", newline="") as flux:
        return {int(row_field["Z"]): row_field for row_field in csv.DictReader(flux)}


def reconstruct(directions, radius, parameters):
    n = len(directions)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, rtol=0, atol=2e-12)
    cosine = np.clip(directions @ directions.T, -1, 1)
    weights = np.exp(-(1-cosine)*n/(4*math.pi))
    np.fill_diagonal(weights, 0)
    laplacian = np.diag(weights.sum(axis=1)) - weights
    base = parameters["base"]
    ref = parameters["references_H_Ne"]
    coupling = base["base_coupling"] * ref["area"] / (4*math.pi*radius**2/n)
    local = base["base_local"] * (ref["radius"]/radius)**2
    graph = base["base_graph"] * (n/radius**2) / ref["density"]
    identite = np.eye(n)
    proj = identite - np.ones((n, n))/n
    retro = (coupling + local)*identite + graph*laplacian
    schur = coupling*identite - coupling**2*np.linalg.solve(retro, identite)
    matrix = proj @ schur @ proj / n
    matrix = (matrix + matrix.T)/2
    return matrix, laplacian, coupling, local, graph


def collection():
    parameters = read_field(HERITAGE / "Parametres_V81_R9.json")
    for name, digest in parameters['files_of_rejeu'].items():
        with (HERITAGE / name).open("rb") as flux:
            assert hashlib.file_digest(flux, "sha256").hexdigest() == digest, name
        source = parameters['checksums_entrees_declarees_par_R9'].get(name)
        if source:
            assert digest == source, name
    atlas = rows(ATLAS)
    assert set(atlas) == set(range(37, 119))
    errors = []
    with np.load(HERITAGE / BASES, allow_pickle=False) as bases, np.load(
            HERITAGE / OPERATORS, allow_pickle=False) as matrices:
        for z, row in atlas.items():
            prefix = f"Z{z:03d}"
            u = bases[prefix + "_shell"]
            radius = float(bases[prefix + "_radius"])
            matrix, lap, c, local, g = reconstruct(u, radius, parameters)
            original = matrices[prefix + "_contact_load_identity_operator"]
            errors.append(float(np.max(np.abs(matrix-original))))
            eig = np.linalg.eigvalsh(matrix)
            assert abs(eig[0]) < 1e-12 and eig[1] > 1e-12
            np.testing.assert_allclose([eig[1], eig[-1]], [float(row["minimum_positive_eigenvalue"]),
                                       float(row["maximum_positive_eigenvalue"])], atol=2e-13, rtol=0)
            if z == 38:
                contact = rows("v93_r1_upper_dynamic_contact_atlas_Z37_Z118.csv")[38]
                assert row["symbol"] == contact["symbol"] == "Sr"
                sr = {'waves': int(bases[prefix + "_count"]), "directions": len(u),
                      "contacts_acceptes": int(contact["accepted_contacts"]), 'radius_reduced': radius,
                      'values_eigen_positives': eig[1:].tolist(), 'value_nulle': float(eig[0]),
                      'coupling': c, "local": local, 'graph': g}
                modes_lap = np.linalg.eigvalsh(lap)[1:]
                errors_scale = []
                factors = (0.8, 1.3)
                for factor in factors:
                    modified = reconstruct(u, factor*radius, parameters)[0]
                    errors_scale.append(float(np.max(abs(modified*factor**2 - matrix))))
                assert max(errors_scale) < 2e-13
                eps = 1e-4
                lo = reconstruct(u, radius*math.exp(-eps), parameters)[0]
                hi = reconstruct(u, radius*math.exp(eps), parameters)[0]
                vit = (np.log(np.linalg.eigvalsh(hi)[1:]) - np.log(np.linalg.eigvalsh(lo)[1:]))/(4*eps)
                np.testing.assert_allclose(vit, -1, atol=2e-9, rtol=0)
                sr['scale'] = {'factors_radius': factors, 'deviation_homogeneite_max': max(errors_scale),
                                 'slopes_d_log_omega_sur_d_log_R': vit.tolist()}
    assert len(errors) == parameters["nombre_operateurs_R9"] == 82 and max(errors) < 2e-13
    return {'operators': len(errors), 'deviation_absolute_max_aux_matrices_archived': max(errors)}, sr, modes_lap


def trial_clock(sr, modes_lap, nu1, nu0):
    # Assumption tested: each line from the same fundamental corresponds to a positive mode.
    values = np.asarray(sr['values_eigen_positives'])
    condition = float(values[-1]/values[0])
    range = math.sqrt(condition)
    ratio = nu1/nu0
    # Dynamic variant: the load and its feedback each have the same inertia.
    c = sr['coupling']
    a = c + sr["local"] + sr['graph']*modes_lap
    blocks = np.empty((len(a), 2, 2))
    blocks[:, 0, 0] = c
    blocks[:, 0, 1] = blocks[:, 1, 0] = -c
    blocks[:, 1, 1] = a
    angular_frequencies = np.sqrt(np.linalg.eigvalsh(blocks))
    all = np.sort(angular_frequencies.ravel())
    low, high = np.triu_indices(len(all), k=1)
    frequencies = nu1 * all[low] / all[high]
    best = int(np.argmin(abs(frequencies-nu0)))
    return {'ratio_measurement_nu1_over_nu0': ratio, "condition_K": condition,
            'ratio_harmonique_max': range, 'bound_frequency_harmonique_Hz': nu1/range,
            'bound_condition_inertie_87Sr': ratio**2/condition,
            'two_fields': {"modes": len(all), 'pairs': len(low),
                             'angular_frequencies_basses_and_hautes': angular_frequencies.tolist(),
                             'ratio_max_band_lower': float(angular_frequencies[-1, 0]/angular_frequencies[0, 0]),
                             'ratio_max_band_upper': float(angular_frequencies[-1, 1]/angular_frequencies[0, 1]),
                             'frequency_la_plus_near_Hz': float(frequencies[best]),
                             'difference_minimum_absolu_Hz': float(abs(frequencies[best]-nu0))}}


def control_angular(condition_k):
    # L'S-1. The dimensionless matrices are expressed in the base m'-1.0,1.
    m = np.arange(-1, 2, dtype=float)
    z = np.diag(m)
    rise = np.diag(np.sqrt(2-m[:-1]*(m[:-1]+1)), -1)
    x = np.kron(z,z) + (np.kron(rise,rise.T) + np.kron(rise.T,rise))/2
    values, counts = np.unique(np.round(np.linalg.eigvalsh(x), 10), return_counts=True)
    np.testing.assert_array_equal(values, [-2, -1, 1])
    np.testing.assert_array_equal(counts, [1, 3, 5])
    data = read_field(DATA / "Niveaux_SrI.json")["niveaux_5s5p_3P"]
    e0, e1, e2 = [data[str(j)]['energy'] for j in (0, 1, 2)]
    expected = e1 - (e2-e1)/2
    # Increase in uncertainties by the weighted sum, without assuming independence.
    u_max = data["0"]["u"] + 1.5*data["1"]["u"] + .5*data["2"]["u"]
    return {"multiplicites": counts.tolist(), "L_point_S": values.tolist(),
            "rapport_E2_sur_E1": e2/e1, 'bound_condition_inertia_terme_integer': (e2/e0)**2/condition_k,
            'temoin_linear_J0_cm_1': expected, 'deviation_sign_cm_1': expected-e0,
            'bound_superieure_u_deviation_cm_1': u_max,
            'correction_quadratique_inversee_over_target_cm_1': (e0-expected)/3}


def domain():
    archive = ROOT/"references/BST_R573_AUDIT_ET_COURBES.zip"
    with ZipFile(archive) as z:
        def section(name):
            return json.loads(z.read("BST_R573/"+name))
        charge = section("resultats/RECALCUL_CHARGE_COMMUNE_R573.json")
        geo = section("resultats/GEOMETRIE_ET_PORTS_R573.json")
        h = section("entrees/R531/entrees/RAIES_H_MESUREES_R502.json")
        hd = section("entrees/preuves/RECUL_H_D.json")
    distances = geo["distances_SF_A"]
    plage = geo["intervalle_diffraction_A"]
    assert len(distances) == 6 and all(plage[0] <= x <= plage[1] for x in distances)
    readouts = {}
    for key in ("charge_unitaire", "Rb", "Cs"):
        family = charge[key]
        assert abs(family['displacement_HD_computed_Hz'] - hd["observation_Hz"] - family['HD_residual_Hz']) < .001
        readouts[key] = {'max_deviation_relative_five_raies': family['doublet_and_others_groups']['H-frequencies'],
                         'residual_HD_Hz': family['HD_residual_Hz'], 'u_measurement_HD_Hz': hd["u_observation_Hz"]}
    return {"H_raie_etalon_pour_lecture_relative": h["calibration"]["label"],
            'H_hors_setting_de_cette_echelle': len(h["held_out_from_scale_fit"]),
            "lectures_R573": readouts, "distances_SF_angstrom": distances, "plage_rg_angstrom": plage}


def figure(output, values, clock, nu0, nu1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, (left_field, right_field) = plt.subplots(1, 2, figsize=(8.0, 2.8),
                                         gridspec_kw={"width_ratios": [1, 1.8]})
    fig.subplots_adjust(left=.085, right=.98, top=.85, bottom=.22, wspace=.6)
    green, ochre, ink = "#196b70", "#a14f2d", "#233941"
    left_field.plot(range(1, len(values)+1), np.asarray(values)/values[0], "o",
                ms=3.6, color=green)
    left_field.set(xticks=[1, 5, 9, 13, 17], xlabel="Mode de contact",
               ylabel="Raideur / raideur minimale", ylim=(.998, 1.018))
    left_field.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.3f}".replace(".", ",")))
    left_field.set_title("17 modes positifs", loc="left", fontsize=10, fontweight="bold")
    bound = clock['bound_frequency_harmonique_Hz'] / 1e12
    near = clock['two_fields']['frequency_la_plus_near_Hz'] / 1e12
    right_field.axvline(nu0/1e12, color=ink, linestyle=":", lw=1.1)
    right_field.plot(nu0/1e12, 2, "D", color=ink, ms=6)
    right_field.plot([bound, nu1/1e12], [1, 1], color=green, linewidth=5, solid_capstyle="butt")
    right_field.plot(near, 0, "o", color=ochre, ms=6)
    for x, y, lab in ((nu0/1e12, 2, "429,228"), (bound, 1, "431,519"), (near, 0, "430,171")):
        right_field.annotate(lab, (x, y), xytext=(4, 7), textcoords="offset points", color=ink, fontsize=8)
    right_field.set(yticks=[2, 1, 0], yticklabels=['Clock', "Inertie commune", 'Two fields'],
               ylim=(-.4, 2.5), xlim=(428.7, 435.3), xlabel='Frequency (THz)')
    right_field.set_title("Test sur le strontium 87", loc="left", fontsize=10, fontweight="bold")
    for ax in (left_field, right_field):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", color="#dde5e6", linewidth=.6)
        ax.set_axisbelow(True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    infos, sr, lap = collection()
    optical = read_field(DATA / "Entrees_strontium.json")
    nu1 = optical["courtillot"]["transition_g_vers_P1_kHz"]*1000
    nu0 = float(optical["bipm"]['frequency_recommandee_Hz'])
    clock = trial_clock(sr, lap, nu1, nu0)
    assert clock['bound_frequency_harmonique_Hz'] > nu0
    assert clock['two_fields']['pairs'] == 561
    output = {'collection': infos, "Sr": sr, "Sr87": clock,
              "terme_electronique_SrI": control_angular(clock["condition_K"]), "D01": domain()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure(args.output.with_name("Modes_du_strontium.png"), sr['values_eigen_positives'], clock, nu0, nu1)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
