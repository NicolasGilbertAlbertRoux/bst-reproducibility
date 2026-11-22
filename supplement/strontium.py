"""Estimate strontium clock frequencies from frozen contact geometry or off-clock spectroscopy."""
from decimal import Decimal
import json
import math
from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.spatial import SphericalVoronoi
from scipy.spatial.transform import Rotation

from reference.sources.article.atomic_spectra_and_geometry import reconstruct, BASES, OPERATORS


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def contacts(geometry, directions):
    n = len(directions)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, atol=2e-12, rtol=0)
    static, laplacian, c, local, graph = reconstruct(directions, geometry["radius"], geometry["parameters"])
    # The old unweighted mean-zero subspace is kept for both fields.
    basis = linalg.helmert(n).T
    identity = np.eye(n - 1)
    restore = (c + local) * identity + graph * (basis.T @ laplacian @ basis)
    stiffness = np.block([[c * identity, -c * identity], [-c * identity, restore]])
    areas = SphericalVoronoi(directions).calculate_areas()
    np.testing.assert_allclose(sum(areas), 4 * np.pi, atol=2e-13, rtol=0)
    return stiffness, basis, areas / np.mean(areas), static, c, restore


def modes(stiffness, basis, weights):
    mass = basis.T @ (weights[:, None] * basis)
    frequencies = np.sqrt(linalg.eigvalsh(stiffness, linalg.block_diag(mass, mass)))
    if not np.all(np.isfinite(frequencies)) or frequencies[0] <= 0:
        raise ValueError("Nonpositive contact mode")
    return frequencies


def training_lines(values, isotope):
    suffix = 1 if isotope == "87" else 0
    return tuple(values[absolute] - suffix * values[shift]
                 for absolute, shift in (("red88", "red_shift"), ("quadrupole88", "quadrupole_shift"),
                                         ("blue88", "blue_shift")))


def select_levels(ordered_modes, f1, f2):
    # The triplet hypothesis needs a lower candidate for J=0 in the same branch.
    ranked = []
    for branch_number, branch in enumerate(np.split(ordered_modes, 2)):
        for middle in range(1, len(branch) - 1):
            for upper in range(middle + 1, len(branch)):
                estimated_lower = (3 * branch[middle] - branch[upper]) / 2
                lower = int(np.argmin(abs(branch[:middle] - estimated_lower)))
                scaled = f1 / branch[middle]
                error = abs(scaled * branch[upper] - f2)
                ranked.append((error, branch_number, middle, upper, lower, scaled * branch[lower],
                               scaled * branch[upper]))
    ranked.sort()
    error, branch, middle, upper, lower, frequency, fitted = ranked[0]
    return {"branch": ["low", "high"][branch], "mode_ranks_one_based": [lower + 1, middle + 1, upper + 1],
            "clock_Hz": frequency, "P2_fit_minus_measured_Hz": fitted - f2, "training_pairs": len(ranked),
            "runner_up_extra_absolute_P2_error_Hz": ranked[1][0] - error}


def contact_predictions(geometry, values):
    directions = np.array(geometry["directions"])
    stiff, basis, weights, static, coupling, restore = contacts(geometry, directions)
    static_error = float(np.max(abs(static - np.array(geometry["published_static_matrix"]))))
    if static_error > 2e-13:
        raise ValueError("Published static contact matrix not recovered")
    equal = modes(stiff, basis, np.ones(len(weights)))
    area = modes(stiff, basis, weights)
    restore_eig = linalg.eigvalsh(restore)
    blocks = [np.array([[coupling, -coupling], [-coupling, eig]]) for eig in restore_eig]
    analytic = np.sort(np.sqrt(np.linalg.eigvalsh(blocks)).ravel())
    np.testing.assert_allclose(equal, analytic, atol=1e-13, rtol=0)
    # Under a rigid rotation or relabeling of all sites the physical spectral answer is invariant.
    rotation = Rotation.from_rotvec([0.63, -0.52, 0.44]).as_matrix()
    perm = np.random.default_rng(22112026).permutation(len(weights))
    s2, q2, w2, *_ = contacts(geometry, (directions @ rotation.T)[perm])
    rotated = modes(s2, q2, w2)
    np.testing.assert_allclose(area, rotated, atol=1e-13, rtol=0)
    np.testing.assert_allclose(weights[perm], w2, atol=1e-13, rtol=0)
    # Direct check that lower/upper branches do not cross as the weights are introduced.
    gaps = []
    for strength in np.linspace(0, 1.05, 22):
        w = modes(stiff, basis, 1 + strength * (weights - 1))
        gaps.append(w[len(weights)-1] - w[len(weights)-2])
    if min(gaps) < 0.5:
        raise ValueError("Branch separation check failed")
    adiabatic = np.sqrt((coupling - coupling**2 / restore_eig) / (1 + (coupling / restore_eig)**2))
    result = {"weights_min_max": [min(weights), max(weights)], "weights": weights.tolist(),
              "static_max_error": static_error, "rotation_permutation_max_error": float(max(abs(rotated-area))),
              "equal_mass_analytic_max_error": float(max(abs(equal-analytic))),
              "min_sampled_gap_reduced": min(gaps), "adiabatic_frequency_ratio_max": max(adiabatic)/min(adiabatic),
              "equal_modes_reduced": equal.tolist(), "area_modes_reduced": area.tolist(), "isotopes": {}}
    for isotope in ("87", "88"):
        f1, f2, _ = training_lines(values, isotope)
        tests = {}
        for strength in (0.95, 0.99, 1.0, 1.01, 1.05):
            w = modes(stiff, basis, 1 + strength * (weights - 1))
            tests[f"{strength:.2f}"] = select_levels(w, f1, f2)
        result["isotopes"][isotope] = {
            "P1_scale_Hz": f1, "P2_training_Hz": f2,
            "equal_mass": select_levels(equal, f1, f2), "solid_angle": tests["1.00"],
            "sensitivity_weight_contrast": tests,
            "adiabatic_lower_bound_Hz": f1 * min(adiabatic) / max(adiabatic)}
    return result


def spectral_predictions(values):
    singlet_rate = 1 / values["singlet_lifetime"] - values["singlet_leak"]
    triplet_rate = 1 / values["triplet_lifetime"]
    out = {}
    for isotope in ("87", "88"):
        f1, f2, bright = training_lines(values, isotope)
        r = (triplet_rate / singlet_rate) * (bright / f1)**3
        singlet_fraction = r / (1 + r)
        displacement = singlet_fraction * (bright - f1)
        linear = (3 * f1 - f2) / 2
        out[isotope] = {"clock_linear_Hz": linear, "singlet_fraction": singlet_fraction,
                        "unmixed_P1_minus_P1_Hz": displacement, "clock_mixed_Hz": linear + 1.5 * displacement,
                        "equivalent_d_cm_inverse": displacement / (2 * 299792458 * 100)}
    return out


def data_uncertainties(observations, values):
    # Shared variables are displaced once and coherently for both isotopes before differencing.
    deviations = []
    for key, entry in observations.items():
        u = math.hypot(*entry["u"])
        plus = spectral_predictions(dict(values, **{key: values[key] + u}))
        minus = spectral_predictions(dict(values, **{key: values[key] - u}))
        deviations.append([(plus[i]["clock_mixed_Hz"] - minus[i]["clock_mixed_Hz"]) / 2
                           for i in ("87", "88")])
    shifts = np.array(deviations)
    return {"87_Hz": float(linalg.norm(shifts[:, 0])), "88_Hz": float(linalg.norm(shifts[:, 1])),
            "88_minus_87_Hz": float(linalg.norm(shifts[:, 1] - shifts[:, 0]))}


def geometry_from_stage(stage):
    directory = stage / "redaction" / "heritage_atomique"
    with np.load(directory / BASES, allow_pickle=False) as sites, np.load(directory / OPERATORS, allow_pickle=False) as stiffness:
        return {"parameters": read(directory / "Parametres_V81_R9.json"),
                "directions": sites["Z038_shell"], "radius": float(sites["Z038_radius"]),
                "published_static_matrix": stiffness["Z038_contact_load_identity_operator"]}


def estimate(geometry, observations):
    values = {key: value["value"] for key, value in observations.items()}
    return {"contacts": contact_predictions(geometry, values), "spectroscopy": spectral_predictions(values),
            "spectroscopy_u_inputs": data_uncertainties(observations, values)}

def best_after_seeing_clock(modes, f1, target):
    w = np.asarray(modes)
    i, j = np.triu_indices(len(w), k=1)
    frequencies = f1 * w[i] / w[j]
    best = np.argmin(abs(frequencies - target))
    return {"pairs": len(i), "clock_Hz": frequencies[best], "error_Hz": frequencies[best] - target}


def compare(predictions, references):
    geometry = predictions["contacts"]
    result = {"by_isotope": {}}
    targets = {k: Decimal(v["value"]) for k, v in references["clock_Hz"].items()}
    for isotope, exact_target in targets.items():
        target = float(exact_target)
        runs = geometry["isotopes"][isotope]
        spec = predictions["spectroscopy"][isotope]
        errors = {"contact_equal_rule": runs["equal_mass"]["clock_Hz"] - target,
                  "contact_area_rule": runs["solid_angle"]["clock_Hz"] - target,
                  "spectral_linear": spec["clock_linear_Hz"] - target, "spectral_mixed": spec["clock_mixed_Hz"] - target}
        result["by_isotope"][isotope] = {
            "target_Hz": str(exact_target), "signed_errors_Hz": errors,
            "contact_reduction_same_rule": abs(errors["contact_equal_rule"] / errors["contact_area_rule"]),
            "spectroscopy_reduction_same_inputs": abs(errors["spectral_linear"] / errors["spectral_mixed"]),
            "adiabatic_positive_error_lower_bound_Hz": runs["adiabatic_lower_bound_Hz"] - target,
            "posthoc_all_pairs_centroid": {
                model: best_after_seeing_clock(geometry[model + "_modes_reduced"], runs["P1_scale_Hz"], target)
                for model in ("equal", "area")},
            "area_weight_contrast": {key: {"error_Hz": val["clock_Hz"] - target, "ranks": val["mode_ranks_one_based"],
                                            "branch": val["branch"]}
                                     for key, val in runs["sensitivity_weight_contrast"].items()}}
    original = references["paper_87Sr_F9_over_2"]
    old = best_after_seeing_clock(geometry["equal_modes_reduced"], original["ground_to_P1_Hz"], float(targets["87"]))
    if abs(abs(old["error_Hz"]) - 943017312660) > 10:
        raise ValueError("Did not recover the published two-field diagnostic")
    direct_difference = Decimal(original["ground_to_P1_Hz"] - original["P0_to_P1_Hz"])
    result["published_87"] = {"posthoc_equal_F9_over_2": old, "Ritz_Hz": str(direct_difference),
                              "Ritz_error_Hz": str(direct_difference - targets["87"]), "Ritz_u_Hz": original["indirect_u_Hz"]}
    isotope_target = float(targets["88"] - targets["87"])
    c = geometry["isotopes"]
    s = predictions["spectroscopy"]
    shifts = {"contact_area_Hz": c["88"]["solid_angle"]["clock_Hz"] - c["87"]["solid_angle"]["clock_Hz"],
              "spectral_mixed_Hz": s["88"]["clock_mixed_Hz"] - s["87"]["clock_mixed_Hz"]}
    result["isotope_88_minus_87"] = {"reference_Hz": isotope_target, "predictions": shifts,
                                     "signed_errors_Hz": {name: shift - isotope_target for name, shift in shifts.items()}}
    return result


def scale_checks(predictions, comparison):
    rows = {}
    for isotope, scored in comparison["by_isotope"].items():
        target = float(scored["target_Hz"])
        red = predictions["contacts"]["isotopes"][isotope]["P1_scale_Hz"]
        interval = red - target
        errors = scored["signed_errors_Hz"]
        u = predictions["spectroscopy_u_inputs"][isotope + "_Hz"]
        rows[isotope] = {
            "red_minus_clock_interval_Hz": interval,
            "absolute_errors_pct_of_clock": {key: 100 * abs(err) / target for key, err in errors.items()},
            "absolute_errors_pct_of_red_minus_clock": {key: 100 * abs(err) / interval for key, err in errors.items()},
            "predict_red_itself_error_pct_of_clock": 100 * abs(interval) / target,
            "predict_red_itself_error_pct_of_interval": 100.0,
            "spectral_mixed_residual_over_input_u": abs(errors["spectral_mixed"]) / u,
        }
    return rows


def run(stage, output):
    output.mkdir(parents=True, exist_ok=True)
    predictions = estimate(geometry_from_stage(stage), read(ROOT / "sr-inputs.json")["observations"])
    write(output / "predictions.json", predictions)
    references = read(ROOT / "sr-references.json")
    # The archived isotope-87 clock, comparison line and Ritz interval are independently checked.
    old = read(stage / "redaction" / "donnees" / "Entrees_strontium.json")
    assert references["clock_Hz"]["87"]["value"] == old["bipm"]["frequence_recommandee_Hz"]
    anchor = references["paper_87Sr_F9_over_2"]
    assert anchor["ground_to_P1_Hz"] == old["courtillot"]["transition_g_vers_P1_kHz"] * 1000
    assert anchor["P0_to_P1_Hz"] == old["courtillot"]["intervalle_P0_vers_P1_kHz"] * 1000
    comparison = compare(predictions, references)
    comparison["scale_checks"] = scale_checks(predictions, comparison)
    write(output / "comparison.json", comparison)
    scored = comparison["by_isotope"]["87"]["signed_errors_Hz"]
    np.testing.assert_allclose([scored["contact_area_rule"], scored["spectral_mixed"]],
                               [356120579445., -42724556312.5], atol=10., rtol=0)
    return {"isotopes": [87, 88], "relative_error_denominators": ["clock", "red_minus_clock"],
            "Sr87_signed_errors_Hz": scored, "Sr87_scales": comparison["scale_checks"]["87"],
            "Sr87_ritz_error_Hz": comparison["published_87"]["Ritz_error_Hz"]}
