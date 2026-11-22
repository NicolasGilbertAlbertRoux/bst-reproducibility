"""Compare the archived joint benchmark and the strontium spacing tests on a fixed grid."""

import csv
import json
from collections import Counter
from zipfile import ZipFile

from supplement import strontium as sr


def read_benchmark(stage):
    with ZipFile(stage / "references" / "BST_R573_AUDIT_ET_COURBES.zip") as archive:
        table = json.loads(archive.read("BST_R573/resultats/ASSEMBLAGES_68_R573.json"))
        decision = json.loads(archive.read("BST_R573/DECISION_R573.json"))
    clauses = table["ordre_des_clauses"]
    assemblies = table["assemblages"]
    assert len(set(clauses)) == len(clauses) == 68
    assert len({row["identifiant"] for row in assemblies}) == len(assemblies) == 576
    return clauses, assemblies, decision


def benchmark_coverage(clauses, assemblies):
    # J00.1 summarizes the 67 other clauses. Counting it as its own input would be circular.
    decision_column = clauses.index("J00.1")
    covered = {}
    classifications = {}
    for assembly in assemblies:
        codes = assembly["codes"]
        assert len(codes) == len(clauses)
        counts = Counter(codes[:decision_column] + codes[decision_column + 1 :])
        identifier = assembly["identifiant"]
        covered[identifier] = set(counts) <= {"C", "H", "D"}
        classifications[identifier] = dict(counts)
    return covered, classifications


def score_spacings(shapes, predictions, comparison):
    levels = predictions["contacts"]["isotopes"]
    anchor = levels["87"]["P1_scale_Hz"]
    # Both isotopes have the same central geometric assignment. Each keeps its measured red line.
    for values in levels.values():
        assert values["solid_angle"]["branch"] == shapes["central_branch"]
        assert values["solid_angle"]["mode_ranks_one_based"] == shapes["central_ranks"]
    grid = []
    for point in shapes["contrast_tests"]:
        by_isotope = {}
        for isotope, terms in levels.items():
            red = terms["P1_scale_Hz"]
            upper = terms["P2_training_Hz"]
            target = float(comparison["by_isotope"][isotope]["target_Hz"])
            assert upper > red > target
            scale = red / anchor
            clock_error = point["tracked_frequency_Hz"] * scale - target
            upper_error = point["tracked_upper_frequency_Hz"] * scale - upper
            mixed_error = comparison["by_isotope"][isotope]["signed_errors_Hz"][
                "spectral_mixed"
            ]
            errors = {
                "clock_gap_geometry": 100 * clock_error / (red - target),
                "upper_gap_geometry": 100 * upper_error / (upper - red),
                "clock_gap_lifetimes": 100 * mixed_error / (red - target),
            }
            by_isotope[isotope] = {
                "signed_frequency_errors_Hz": {
                    "clock_geometry": clock_error,
                    "upper_geometry": upper_error,
                    "clock_lifetimes": mixed_error,
                },
                "spacing_denominators_Hz": {
                    "red_minus_clock": red - target,
                    "upper_minus_red": upper - red,
                },
                "signed_errors_pct_of_spacing": errors,
                "within_five_percent": all(
                    abs(error) <= 5 for error in errors.values()
                ),
            }
        grid.append(
            {
                "contrast": point["weight_contrast"],
                "tracked_ranks": point["tracked_global_ranks"],
                "minimum_squared_shape_overlap": min(point["shape_overlap_squared"]),
                "isotopes": by_isotope,
                "strontium_pass": all(
                    row["within_five_percent"] for row in by_isotope.values()
                ),
            }
        )
    return grid


def run(stage, results):
    output = results / "joint-validation"
    output.mkdir(parents=True, exist_ok=True)
    clauses, assemblies, decision = read_benchmark(stage)
    covered, counts = benchmark_coverage(clauses, assemblies)
    expected = decision["decision_globale"][
        "assemblages_couvrant_67_controles_explicites"
    ]
    assert sum(covered.values()) == expected
    grid = score_spacings(
        sr.read(results / "atomic-correspondence" / "shape-predictions.json"),
        sr.read(results / "strontium" / "predictions.json"),
        sr.read(results / "strontium" / "comparison.json"),
    )
    with (output / "intersection.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "assembly",
                "contrast",
                "benchmark_67_pass",
                "strontium_pass",
                "joint_pass",
            ]
        )
        for point in grid:
            accepted = 0
            for identifier, benchmark_pass in covered.items():
                both = benchmark_pass and point["strontium_pass"]
                writer.writerow(
                    [
                        identifier,
                        point["contrast"],
                        int(benchmark_pass),
                        int(point["strontium_pass"]),
                        int(both),
                    ]
                )
                accepted += both
            point["compatible_assemblies"] = accepted
    example = decision["decision_globale"]["premiers_avec_67_controles"][0]
    assert covered[example]
    scope = decision["comptes"]
    report = {
        "tolerance_pct_of_each_spacing": 5,
        "assemblies": len(assemblies),
        "benchmark_checks_per_assembly": len(clauses) - 1,
        "assemblies_passing_benchmark": sum(covered.values()),
        "example_assembly": example,
        "example_classifications": counts[example],
        "benchmark_code_key": {
            "C": "calculated or compared on this assembly",
            "H": "local result retained at fixed parameters",
            "D": "evidence independent of this setting",
        },
        "registered_rows_with_scoped_result": scope["with_scoped_result"],
        "rows_with_explicit_conditions": scope["conditionnels_de_portee"],
        "whole_contracts_met": scope["whole_criteria_validated"],
        "whole_contracts_total": scope["whole_criteria_total"],
        "grid": grid,
        "scope": "The five existing strontium contrasts are scored together. The upper spacing contributed to the initial mode assignment. The clock measurement is used for scoring and for reporting which contrast passes. Contrast is local to the strontium mass operator. It is separate from the six assembly coordinates. Reusing the same strontium readout for the 576 rows tests coexistence; it does not test a shared law for this coefficient in other domains.",
    }
    assert [p["contrast"] for p in grid] == [0.95, 0.99, 1.0, 1.01, 1.05]
    assert [p["compatible_assemblies"] for p in grid] == [0, 0, 0, 0, 192]
    sr.write(output / "joint-checks.json", report)
    return report
