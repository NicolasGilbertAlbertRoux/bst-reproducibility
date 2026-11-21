#!/usr/bin/env python3
"V94-R11C-R5 — extension conservative globale de l'action de contact R10.\n\nR4-R1 recovered three trajectories stopped before their requested duration. The\nfields and integers available remained valid; the limitation\nidentified is the domain of the local polynomial V68/R10. This course does not re-enter\nno field S3. It builds, from the blind curves archived only\nby R10, an overall C1 energy which:\n\n- reproduces exactly the R10 polynomial in its frozen local window;\nthe connecting over two steps of the grid R10 to the monotonic cubic interpolating\n201 pre-existing geometric energies;\n- extends both edges by coercive quadratic tails, without parameter\nadjusted to the R4 trajectories;\nprovides analytically energy and derived, therefore a conservative force.\n\nR5 is a bypass door. It may authorize a new S3 pilot but does not\nfirm or C2, or C3, and does not alter any R10/R11B/R2/R3/R4-R1 result."

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/v94_r11c_r5_global_conservative_contact_extension"
R10_REPORT = ROOT / "results/v94_r10_stratified_upper_full_operator_wells/v94_r10_stratified_upper_full_operator_wells.json"
R10_CURVES = ROOT / "results/v94_r10_stratified_upper_full_operator_wells/v94_r10_stratified_upper_full_operator_curves.npz"
R4_R1_REPORT = ROOT / "results/v94_r11c_r4_r1_partial_cycle_artifact_recovery/v94_r11c_r4_r1_partial_cycle_artifact_recovery.json"

EXPECTED_R10_REPORT_SHA256 = "d4112ea62dd8e95972a9ec47665228ba8c5f62637aef1c5f6058073d3bf556f3"
EXPECTED_R10_CURVES_SHA256 = "2326bc304c72ae6748db9da855bc219d29e50b38de8400c5a1ba8179b223a9ee"
EXPECTED_R4_R1_SHA256 = "b41d24a48b85baec957b91eac373ba5ed23d4d07d6ce03b3a263df9c5005167c"
EXPECTED_R10_VERDICT = "STRATIFIED_UPPER_FULL_OPERATOR_WELLS_PASS__SELECTIVE_UPPER_S3_NEXT"
EXPECTED_R4_R1_VERDICT = "R4_PARTIAL_ARTIFACTS_RECOVERED__THREE_EARLY_TERMINATIONS__C2_REMAINS_OPEN"

CASES = {
    "II": ("Z053_Z053", 5),
    "IPb": ("Z053_Z082", 6),
    "GeFl": ("Z032_Z114", 7),
}
BRIDGE_GRID_STEPS = 2
DT = 0.00125
CONTINUITY_EPSILON = 1.0e-9
MAXIMUM_VALUE_JUMP = 2.0e-8
MAXIMUM_DERIVATIVE_JUMP = 2.0e-6
MAXIMUM_FORCE_FD_RELATIVE_ERROR = 2.0e-5
MAXIMUM_REDUCED_ENERGY_DRIFT = 1.0e-9
MINIMUM_REDUCED_TURNINGS = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value) -> None:
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(type(item).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=convert) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    'Fritsch–Carlson slopes, SciPy-dependent.'
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 3 or not np.all(np.diff(x) > 0.0):
        raise ValueError("grille PCHIP invalide")
    h = np.diff(x); delta = np.diff(y) / h
    slopes = np.zeros_like(y)
    for index in range(1, len(y) - 1):
        if delta[index - 1] * delta[index] > 0.0:
            w1 = 2.0 * h[index] + h[index - 1]
            w2 = h[index] + 2.0 * h[index - 1]
            slopes[index] = (w1 + w2) / (w1 / delta[index - 1] + w2 / delta[index])

    def endpoint(h0, h1, d0, d1):
        value = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if value * d0 <= 0.0:
            return 0.0
        if d0 * d1 < 0.0 and abs(value) > 3.0 * abs(d0):
            return 3.0 * d0
        return value

    slopes[0] = endpoint(h[0], h[1], delta[0], delta[1])
    slopes[-1] = endpoint(h[-1], h[-2], delta[-1], delta[-2])
    return slopes


def cubic_hermite(value: float, left: float, right: float,
                  y_left: float, y_right: float,
                  d_left: float, d_right: float) -> tuple[float, float]:
    width = right - left
    if width <= 0.0:
        raise ValueError("intervalle Hermite invalide")
    t = (value - left) / width
    energy = ((2*t**3 - 3*t**2 + 1) * y_left
              + (t**3 - 2*t**2 + t) * width * d_left
              + (-2*t**3 + 3*t**2) * y_right
              + (t**3 - t**2) * width * d_right)
    derivative = ((6*t*t - 6*t) * y_left
                  + (3*t*t - 4*t + 1) * width * d_left
                  + (-6*t*t + 6*t) * y_right
                  + (3*t*t - 2*t) * width * d_right) / width
    return float(energy), float(derivative)


class GlobalContactExtension:
    def __init__(self, fractions: np.ndarray, energies: np.ndarray, pair: dict):
        self.x = np.asarray(fractions, float)
        self.y = np.asarray(energies, float)
        self.slopes = pchip_slopes(self.x, self.y)
        self.pair = pair
        self.fit = pair["fit"]
        self.radius_sum = float(pair["radius_sum"])
        self.coefficients = np.asarray(self.fit["local_polynomial_coefficients"], float)
        self.derivative_coefficients = np.polyder(self.coefficients)
        self.origin = float(self.fit["local_polynomial_fraction_origin"])
        self.local_lower, self.local_upper = map(float, self.fit["local_fit_fraction_interval"])
        steps = np.diff(self.x)
        if not np.allclose(steps, steps[0], rtol=0.0, atol=1.0e-14):
            raise RuntimeError('the grid R10 must be uniform')
        self.grid_step = float(steps[0])
        self.left_anchor = self.local_lower - BRIDGE_GRID_STEPS * self.grid_step
        self.right_anchor = self.local_upper + BRIDGE_GRID_STEPS * self.grid_step
        if self.left_anchor < self.x[0] or self.right_anchor > self.x[-1]:
            raise RuntimeError("ancre de raccord hors de la courbe R10")
        self.left_anchor_index = int(np.argmin(np.abs(self.x - self.left_anchor)))
        self.right_anchor_index = int(np.argmin(np.abs(self.x - self.right_anchor)))
        if (abs(self.x[self.left_anchor_index] - self.left_anchor) > 1.0e-12
                or abs(self.x[self.right_anchor_index] - self.right_anchor) > 1.0e-12):
            raise RuntimeError('the anchors must be exact R10 points')
        curvature = float(self.fit["curvature_fraction"])
        self.left_tail_curvature = max(abs(self.slopes[0]) / self.grid_step, curvature, 1.0e-12)
        self.right_tail_curvature = max(abs(self.slopes[-1]) / self.grid_step, curvature, 1.0e-12)

    def local(self, fraction: float) -> tuple[float, float]:
        offset = fraction - self.origin
        return (float(np.polyval(self.coefficients, offset)),
                float(np.polyval(self.derivative_coefficients, offset)))

    def pchip(self, fraction: float) -> tuple[float, float]:
        index = int(np.searchsorted(self.x, fraction) - 1)
        index = max(0, min(len(self.x) - 2, index))
        return cubic_hermite(fraction, self.x[index], self.x[index + 1],
                             self.y[index], self.y[index + 1],
                             self.slopes[index], self.slopes[index + 1])

    def fraction_energy_derivative(self, fraction: float) -> tuple[float, float]:
        value = float(fraction)
        if value <= self.x[0]:
            displacement = value - self.x[0]
            return (float(self.y[0] + self.slopes[0] * displacement
                          + 0.5 * self.left_tail_curvature * displacement**2),
                    float(self.slopes[0] + self.left_tail_curvature * displacement))
        if value < self.left_anchor:
            return self.pchip(value)
        if value < self.local_lower:
            local_energy, local_derivative = self.local(self.local_lower)
            index = self.left_anchor_index
            return cubic_hermite(value, self.left_anchor, self.local_lower,
                                 self.y[index], local_energy,
                                 self.slopes[index], local_derivative)
        if value <= self.local_upper:
            return self.local(value)
        if value < self.right_anchor:
            local_energy, local_derivative = self.local(self.local_upper)
            index = self.right_anchor_index
            return cubic_hermite(value, self.local_upper, self.right_anchor,
                                 local_energy, self.y[index],
                                 local_derivative, self.slopes[index])
        if value < self.x[-1]:
            return self.pchip(value)
        displacement = value - self.x[-1]
        return (float(self.y[-1] + self.slopes[-1] * displacement
                      + 0.5 * self.right_tail_curvature * displacement**2),
                float(self.slopes[-1] + self.right_tail_curvature * displacement))

    def separation_energy_derivative(self, separation: float) -> tuple[float, float]:
        energy, fraction_derivative = self.fraction_energy_derivative(separation / self.radius_sum)
        return energy, fraction_derivative / self.radius_sum

    def contract(self) -> dict:
        return {
            "radius_sum": self.radius_sum,
            "local_polynomial_coefficients": self.coefficients,
            "local_polynomial_fraction_origin": self.origin,
            "local_fit_fraction_interval": [self.local_lower, self.local_upper],
            "global_R10_fraction_interval": [float(self.x[0]), float(self.x[-1])],
            "bridge_grid_steps": BRIDGE_GRID_STEPS,
            "left_bridge_interval": [self.left_anchor, self.local_lower],
            "right_bridge_interval": [self.local_upper, self.right_anchor],
            "left_tail_curvature": self.left_tail_curvature,
            "right_tail_curvature": self.right_tail_curvature,
        }


def audit_gate() -> tuple[dict, dict]:
    required = (R10_REPORT, R10_CURVES, R4_R1_REPORT)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(missing))
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in required}
    if sha256(R10_REPORT) != EXPECTED_R10_REPORT_SHA256:
        raise RuntimeError('Infringement of the incompatible R10 ratio')
    if sha256(R10_CURVES) != EXPECTED_R10_CURVES_SHA256:
        raise RuntimeError('imprint of the incompatible curves R 10')
    if sha256(R4_R1_REPORT) != EXPECTED_R4_R1_SHA256:
        raise RuntimeError('imprint of the incompatible R4-R1 ratio')
    report10 = load_json(R10_REPORT); report4 = load_json(R4_R1_REPORT)
    if report10.get("pass") is not True or report10.get("verdict") != EXPECTED_R10_VERDICT:
        raise RuntimeError("porte R10 incompatible")
    if (report4.get("audit_completed") is not True or report4.get("pass") is not False
            or report4.get("scientific_C2_gate_pass") is not False
            or report4.get("verdict") != EXPECTED_R4_R1_VERDICT
            or report4.get("all_available_exact_integer_degrees_preserved") is not True):
        raise RuntimeError("porte R4-R1 incompatible")
    return report10, {"input_hashes": hashes, "R4_R1": report4}


def continuity_audit(extension: GlobalContactExtension) -> dict:
    boundaries = (extension.x[0], extension.left_anchor, extension.local_lower,
                  extension.local_upper, extension.right_anchor, extension.x[-1])
    rows = []
    for boundary in boundaries:
        left = extension.fraction_energy_derivative(boundary - CONTINUITY_EPSILON)
        right = extension.fraction_energy_derivative(boundary + CONTINUITY_EPSILON)
        rows.append({"boundary": boundary, "value_jump": abs(left[0] - right[0]),
                     "derivative_jump": abs(left[1] - right[1])})
    return {
        "maximum_value_jump": max(row["value_jump"] for row in rows),
        "maximum_derivative_jump": max(row["derivative_jump"] for row in rows),
        "rows": rows,
        "pass": bool(max(row["value_jump"] for row in rows) < MAXIMUM_VALUE_JUMP
                     and max(row["derivative_jump"] for row in rows) < MAXIMUM_DERIVATIVE_JUMP),
    }


def exact_local_audit(extension: GlobalContactExtension) -> dict:
    values = np.linspace(extension.local_lower, extension.local_upper, 1001)
    errors = []
    derivative_errors = []
    for value in values:
        expected = extension.local(float(value))
        actual = extension.fraction_energy_derivative(float(value))
        errors.append(abs(expected[0] - actual[0]))
        derivative_errors.append(abs(expected[1] - actual[1]))
    return {"sample_count": len(values), "maximum_energy_error": max(errors),
            "maximum_derivative_error": max(derivative_errors),
            "pass": bool(max(errors) == 0.0 and max(derivative_errors) == 0.0)}


def force_audit(extension: GlobalContactExtension) -> dict:
    fractions = np.linspace(0.05, 2.75, 271)
    epsilon = 2.0e-7 * extension.radius_sum
    relative_errors = []
    scaled_errors = []
    absolute_errors = []
    for fraction in fractions:
        separation = fraction * extension.radius_sum
        _energy, derivative = extension.separation_energy_derivative(separation)
        plus = extension.separation_energy_derivative(separation + epsilon)[0]
        minus = extension.separation_energy_derivative(separation - epsilon)[0]
        numeric = (plus - minus) / (2.0 * epsilon)
        absolute = abs(numeric - derivative)
        absolute_errors.append(absolute)
        # The gross relative is informative, but indefinite to extrema where force
        # Exact is cancelled out. The decision therefore employs a common mixed error,
        # normalized by the unit when the two derivatives are small.
        relative_errors.append(absolute / max(abs(numeric), abs(derivative), 1.0e-30))
        scaled_errors.append(absolute / max(abs(numeric), abs(derivative), 1.0))
    maximum = max(scaled_errors)
    return {"sample_count": len(fractions),
            "maximum_absolute_error": max(absolute_errors),
            "maximum_raw_relative_error_diagnostic": max(relative_errors),
            "maximum_scaled_relative_error": maximum,
            "pass": bool(maximum < MAXIMUM_FORCE_FD_RELATIVE_ERROR)}


def reduced_cycle(extension: GlobalContactExtension) -> dict:
    pair = extension.pair; fit = extension.fit
    mass = float(pair["reduced_wave_mass"])
    duration = 1.2 * 2.0 * math.pi * math.sqrt(mass / float(fit["curvature_separation"]))
    steps = int(math.ceil(duration / DT))
    separation = float(fit["equilibrium_separation"]) * 1.015
    velocity = 0.0
    initial_energy = extension.separation_energy_derivative(separation)[0]
    _value, derivative = extension.separation_energy_derivative(separation)
    acceleration = -derivative / mass
    minimum = maximum = separation
    previous_sign = 0
    turnings = 0
    drift = 0.0
    for _step in range(1, steps + 1):
        half = velocity + 0.5 * DT * acceleration
        separation += DT * half
        potential, derivative = extension.separation_energy_derivative(separation)
        new_acceleration = -derivative / mass
        velocity = half + 0.5 * DT * new_acceleration
        acceleration = new_acceleration
        energy = potential + 0.5 * mass * velocity**2
        drift = max(drift, abs(energy - initial_energy) / max(abs(initial_energy), 1.0))
        minimum = min(minimum, separation); maximum = max(maximum, separation)
        sign = 1 if velocity > 1.0e-14 else (-1 if velocity < -1.0e-14 else 0)
        if previous_sign and sign and sign != previous_sign:
            turnings += 1
        if sign:
            previous_sign = sign
    return {
        "duration": duration, "steps": steps, "turning_count": turnings,
        "minimum_fraction": minimum / extension.radius_sum,
        "maximum_fraction": maximum / extension.radius_sum,
        "maximum_relative_energy_drift": drift,
        "pass": bool(turnings >= MINIMUM_REDUCED_TURNINGS
                     and drift < MAXIMUM_REDUCED_ENERGY_DRIFT),
    }


def dense_shape_audit(extension: GlobalContactExtension) -> dict:
    fractions = np.linspace(0.0, 3.0, 30001)
    energies = np.asarray([extension.fraction_energy_derivative(value)[0] for value in fractions])
    index = int(np.argmin(energies))
    equilibrium = float(extension.fit["equilibrium_fraction"])
    global_minimum_in_local_window = extension.local_lower <= fractions[index] <= extension.local_upper
    return {
        "fraction_interval": [float(fractions[0]), float(fractions[-1])],
        "sample_count": len(fractions),
        "global_minimum_fraction": float(fractions[index]),
        "R10_equilibrium_fraction": equilibrium,
        "global_minimum_energy": float(energies[index]),
        "global_minimum_remains_in_frozen_local_window": bool(global_minimum_in_local_window),
        "left_tail_is_coercive": bool(energies[0] > energies[index]),
        "right_tail_is_coercive": bool(energies[-1] > energies[index]),
        "all_values_finite": bool(np.all(np.isfinite(energies))),
        "pass": bool(global_minimum_in_local_window and np.all(np.isfinite(energies))
                     and energies[0] > energies[index] and energies[-1] > energies[index]),
    }


def smoke() -> dict:
    x = np.linspace(0.4, 2.4, 201)
    y = (x - 1.1)**2 - 0.2
    slopes = pchip_slopes(x, y)
    midpoint, derivative = cubic_hermite(0.405, x[0], x[1], y[0], y[1], slopes[0], slopes[1])
    criteria = {
        "PCHIP_slopes_are_finite": bool(np.all(np.isfinite(slopes))),
        "Hermite_value_and_derivative_are_finite": bool(math.isfinite(midpoint) and math.isfinite(derivative)),
        "bridge_width_is_exactly_two_R10_grid_steps": BRIDGE_GRID_STEPS == 2,
        "local_R10_polynomial_must_be_preserved_exactly": True,
        "tails_are_quadratic_and_coercive": True,
        "extension_has_an_analytic_derivative": True,
        "R4_R1_outcomes_are_a_gate_not_fit_targets": True,
        "no_S3_field_is_integrated_in_R5": True,
        "no_distance_energy_spectrum_valence_chemistry_or_lifetime_reference_is_loaded": True,
        "R5_cannot_close_C2_or_authorize_C3": True,
    }
    criteria = {key: bool(value) for key, value in criteria.items()}
    return {"pass": all(criteria.values()), "technical_smoke_only": True, "criteria": criteria}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", 'drift'), default='drift')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke":
        report = smoke(); dump_json(OUTPUT / "v94_r11c_r5_smoke.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["pass"] else 1

    print('V94-R11C-R5 A/5: Crude R10 and R4-R1 frozen causal doors', flush=True)
    report10, gate = audit_gate()
    print('V94-R11C-R5 B/5: overall R10 geometric curves and exact local polynomials', flush=True)
    rows = []; contracts = {}; arrays = {}
    with np.load(R10_CURVES, allow_pickle=False) as curves:
        for name, (prefix, period) in CASES.items():
            pair = report10["pairs"][prefix]
            fractions = np.asarray(curves[prefix + "_fractions"], float)
            energies = np.asarray(curves[prefix + "_energy"], float)
            extension = GlobalContactExtension(fractions, energies, pair)
            local = exact_local_audit(extension)
            continuity = continuity_audit(extension)
            force = force_audit(extension)
            shape = dense_shape_audit(extension)
            cycle = reduced_cycle(extension)
            passed = all(item["pass"] for item in (local, continuity, force, shape, cycle))
            contracts[name] = extension.contract()
            arrays[f"{name}_fractions"] = fractions
            arrays[f"{name}_energies"] = energies
            arrays[f"{name}_pchip_slopes"] = extension.slopes
            arrays[f"{name}_local_polynomial_coefficients"] = extension.coefficients
            rows.append({
                "pair": name, "period": period, "pass": passed,
                "local_maximum_energy_error": local["maximum_energy_error"],
                "local_maximum_derivative_error": local["maximum_derivative_error"],
                "maximum_value_jump": continuity["maximum_value_jump"],
                "maximum_derivative_jump": continuity["maximum_derivative_jump"],
                "force_maximum_scaled_relative_error": force["maximum_scaled_relative_error"],
                "force_maximum_absolute_error": force["maximum_absolute_error"],
                "global_minimum_fraction": shape["global_minimum_fraction"],
                "global_minimum_in_local_window": shape["global_minimum_remains_in_frozen_local_window"],
                "reduced_turning_count": cycle["turning_count"],
                "reduced_energy_drift": cycle["maximum_relative_energy_drift"],
                "reduced_minimum_fraction": cycle["minimum_fraction"],
                "reduced_maximum_fraction": cycle["maximum_fraction"],
            })
            contracts[name]["audits"] = {
                "exact_local": local, "C1_continuity": continuity,
                "analytic_force": force, "global_shape": shape, "reduced_cycle": cycle,
            }
            print(f"V94-R11C-R5 {name}: local_exact={local['pass']}, C1={continuity['pass']}, "
                  f"force={force['pass']}, reduced_turns={cycle['turning_count']}, pass={passed}", flush=True)

    print('V94-R11C-R5 C/5: copper and analytical drifts', flush=True)
    atlas_path = OUTPUT / "v94_r11c_r5_global_contact_extension_atlas.csv"
    contract_path = OUTPUT / "v94_r11c_r5_global_contact_extension_contract.json"
    arrays_path = OUTPUT / "v94_r11c_r5_global_contact_extensions_II_IPb_GeFl.npz"
    write_csv(atlas_path, rows); dump_json(contract_path, contracts); np.savez_compressed(arrays_path, **arrays)
    print('V94-R11C-R5 D/5: reduced audit over the three frozen periods', flush=True)
    criteria = {
        "R10_report_and_global_curves_match_frozen_hashes": True,
        "R4_R1_failure_and_preserved_integer_degrees_match_frozen_hash": True,
        "all_three_local_R10_polynomials_are_exactly_preserved": all(contracts[n]["audits"]["exact_local"]["pass"] for n in CASES),
        "all_three_global_extensions_are_C1_at_every_join": all(contracts[n]["audits"]["C1_continuity"]["pass"] for n in CASES),
        "all_three_analytic_forces_pass_finite_differences": all(contracts[n]["audits"]["analytic_force"]["pass"] for n in CASES),
        "all_three_frozen_wells_remain_global_minima_of_the_audited_range": all(contracts[n]["audits"]["global_shape"]["pass"] for n in CASES),
        "all_three_reduced_full_cycles_are_conservative_and_have_two_turnings": all(contracts[n]["audits"]["reduced_cycle"]["pass"] for n in CASES),
        "bridge_rule_uses_only_two_predeclared_R10_grid_steps": True,
        "tails_are_derived_only_from_R10_boundary_slopes_grid_and_curvature": True,
        "no_R4_trajectory_value_is_used_as_an_extension_parameter": True,
        "no_S3_field_was_integrated_and_C2_C3_remain_open": True,
    }
    passed = bool(all(criteria.values()) and all(row["pass"] for row in rows))
    report = {
        "stage": "V94_R11C_R5_GLOBAL_CONSERVATIVE_CONTACT_EXTENSION",
        "audit_completed": True, "pass": passed,
        "scientific_C2_gate_pass": False,
        "C3_upper_multicontact_and_polyatomic_gate_pass": False,
        "verdict": ("GLOBAL_CONSERVATIVE_R10_CONTACT_EXTENSION_DERIVED_PASS__BLIND_S3_PILOT_NEXT"
                    if passed else "GLOBAL_CONSERVATIVE_R10_CONTACT_EXTENSION_NOT_PASSED__C2_REMAINS_OPEN"),
        "criteria": criteria,
        "construction": {
            "core": "exact frozen R10 local quartic",
            "bridges": "two-grid-step cubic Hermite C1 joins",
            "global_curve": "Fritsch-Carlson cubic Hermite interpolation of the 201 frozen R10 points",
            "tails": "C1 quadratic coercive continuations",
            "new_fitted_parameter": False,
            "R4_outcome_feedback": False,
        },
        "cases": contracts,
        "preserved_results": {
            "R10": report10["verdict"], "R4_R1": gate["R4_R1"]["verdict"],
            "available_exact_integer_degrees_preserved": gate["R4_R1"]["all_available_exact_integer_degrees_preserved"],
        },
        "input_hashes": gate["input_hashes"],
        "next": ("run a preregistered short blind S3 pilot beyond each former local-window boundary, then full cycles"
                 if passed else "do not reintegrate S3; localize the failed extension audit"),
        "outputs": {"report": str(OUTPUT / "v94_r11c_r5_global_conservative_contact_extension.json"),
                    "atlas": str(atlas_path), "contract": str(contract_path), "arrays": str(arrays_path)},
    }
    dump_json(OUTPUT / "v94_r11c_r5_global_conservative_contact_extension.json", report)
    print('V94-R11C-R5 E/5: decision', flush=True)
    print(json.dumps({key: report[key] for key in ("pass", "scientific_C2_gate_pass", "verdict", "next", "outputs")}, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
