"""Reconnect the original R16 atomic signatures to the two-field strontium calculation."""

import csv
import hashlib
import importlib.util
import io
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy import linalg, optimize

from supplement import strontium as sr

ARCHIVE = Path(__file__).with_name("atomic-signatures.zip")
ARCHIVE_SHA256 = "eb9fa65e615e160925a9fce85645a45ffcc65aa8fcd083f179f3f155ec00a183"
PREFIX = "bst_atomic_signature_r16/"
COLUMNS = ("dipole_weight", "quadrupole_weight", "octupole_weight", "residual_weight")


def restore_bank(stage_geometry, output):
    """Use the archived code and its four geometry/operator inputs. Keep the original archive intact."""
    with ARCHIVE.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != ARCHIVE_SHA256:
            raise ValueError("R16 source archive differs from the recorded version")
    with (
        ZipFile(ARCHIVE) as z,
        tempfile.TemporaryDirectory(prefix="bst-r16-") as temporary,
    ):
        source = "run_atomic_signature_r16.py"
        for name in (
            source,
            "inputs/v94_r6_contact_load_identity_operators_H_Kr.npz",
            "inputs/v94_r9_upper_contact_load_identity_operators_Z37_Z118.npz",
            "inputs/v94_r4_lower_common_action_bases_H_Kr.npz",
            "inputs/v92_r1_upper_atomic_bases_Z37_Z118.npz",
        ):
            z.extract(PREFIX + name, temporary)
        spec = importlib.util.spec_from_file_location(
            "bst_r16_frozen", Path(temporary) / PREFIX / source
        )
        r16 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(r16)
        stores, get = r16.load_banks()
        try:
            np.testing.assert_array_equal(
                get(38, "shell"), stage_geometry["directions"]
            )
            np.testing.assert_array_equal(
                get(38, "contact_load_identity_operator"),
                stage_geometry["published_static_matrix"],
            )
            spaces = r16.multipole_spaces(stage_geometry["directions"])
            _, modes, summaries = r16.spectral_bank(get)
        finally:
            for bank in stores:
                bank.close()
        references = {}
        for filename, rows in [
            ("BST_ATOMIC_MODE_BANK_Z001_Z118_R16.csv", modes),
            ("BST_ATOMIC_SIGNATURE_SUMMARY_Z001_Z118_R16.csv", summaries),
        ]:
            original = z.read(PREFIX + "outputs/" + filename)
            destination = output / filename
            r16.write_csv(destination, rows)
            references[filename] = {
                "archived_sha256": hashlib.sha256(original).hexdigest(),
                "recomputed_sha256": hashlib.sha256(
                    destination.read_bytes()
                ).hexdigest(),
            }
        original_modes = list(
            csv.DictReader(
                io.StringIO(
                    z.read(
                        PREFIX + "outputs/BST_ATOMIC_MODE_BANK_Z001_Z118_R16.csv"
                    ).decode()
                )
            )
        )
    assert len(summaries) == 118 and len(modes) == len(original_modes) == 8812
    for archived, computed in zip(original_modes, modes, strict=True):
        assert (
            int(archived["Z"]),
            int(archived["mode_index"]),
            int(archived["band_id"]),
        ) == (computed["Z"], computed["mode_index"], computed["band_id"])
    np.testing.assert_allclose(
        [float(r["eigenvalue"]) for r in original_modes],
        [r["eigenvalue"] for r in modes],
        rtol=1e-9,
        atol=2e-13,
    )
    # Individual eigenvectors within a degenerate band depend on the solver. Sum the shape weights over each band.
    bands = np.array([(r["Z"], r["band_id"]) for r in modes])
    starts = np.r_[0, 1 + np.where(np.any(bands[1:] != bands[:-1], axis=1))[0]]
    original_weights = np.array(
        [[float(r[k]) for k in COLUMNS] for r in original_modes]
    )
    weights = np.array([[r[k] for k in COLUMNS] for r in modes])
    np.testing.assert_allclose(
        np.add.reduceat(original_weights, starts, axis=0),
        np.add.reduceat(weights, starts, axis=0),
        atol=2e-7,
        rtol=1e-8,
    )
    sr_modes = [r for r in original_modes if int(r["Z"]) == 38]
    counts = {
        name: sum(r["dominant_geometry"] == name for r in sr_modes)
        for name in ("dipole", "quadrupole", "octupole", "residual")
    }
    return spaces, {
        "atoms": len(summaries),
        "positive_modes": len(modes),
        "Sr_static_positive_modes": len(sr_modes),
        "Sr_static_dominant_geometry_counts": counts,
        "files": references,
        "R16_source_sha256": ARCHIVE_SHA256,
    }


def solve(stiffness, basis, weights, strength):
    site_weights = 1 + strength * (weights - 1)
    mass = basis.T @ (site_weights[:, None] * basis)
    mass = linalg.block_diag(mass, mass)
    values, vectors = linalg.eigh(stiffness, mass)
    assert values.min() > 0
    return np.sqrt(values), vectors, mass


def normalized_overlap(reference, varied, metric):
    def unit_vectors(matrix):
        return matrix / np.sqrt(np.sum(matrix * (metric @ matrix), axis=0))

    return (unit_vectors(reference).T @ metric @ unit_vectors(varied)) ** 2


def compare_modes(geometry, spaces, values):
    directions = geometry["directions"]
    stiffness, basis, weights, *_ = sr.contacts(geometry, directions)
    central, reference, metric = solve(stiffness, basis, weights, 1.0)
    f1, f2, _ = sr.training_lines(values, "87")
    assigned = sr.select_levels(central, f1, f2)
    branch_size = basis.shape[1]
    offset = branch_size if assigned["branch"] == "high" else 0
    chosen = offset + np.array(assigned["mode_ranks_one_based"]) - 1
    fields = np.stack(
        [basis @ reference[:branch_size], basis @ reference[branch_size:]]
    )
    total = np.sum(fields**2, axis=(0, 1))
    projected = np.array(
        [
            sum(np.sum((spaces[degree].T @ field) ** 2, axis=0) for field in fields)
            / total
            for degree in (1, 2, 3)
        ]
    )
    projected = np.vstack([projected, np.maximum(0.0, 1 - projected.sum(axis=0))])
    np.testing.assert_allclose(projected.sum(axis=0), 1, atol=2e-12, rtol=0)
    shapes = {
        str(index + 1): dict(zip(COLUMNS, projected[:, index].tolist(), strict=True))
        for index in chosen
    }
    tests = []
    for strength in (0.95, 0.99, 1.0, 1.01, 1.05):
        freq, vectors, _ = solve(stiffness, basis, weights, strength)
        overlap = normalized_overlap(reference, vectors, metric)
        before, after = optimize.linear_sum_assignment(-overlap)
        np.testing.assert_array_equal(before, np.arange(len(freq)))
        tracked = after[chosen]
        # The assignment maximizes squared shape overlap for all 34 modes; neither frequency nor J is an input to this step.
        selected = sr.select_levels(freq, f1, f2)
        tests.append(
            {
                "weight_contrast": strength,
                "tracked_global_ranks": (tracked + 1).tolist(),
                "shape_overlap_squared": overlap[chosen, tracked].tolist(),
                "reselected_branch": selected["branch"],
                "reselected_within_branch_ranks": selected["mode_ranks_one_based"],
                "tracked_frequency_Hz": float(f1 * freq[tracked[0]] / freq[tracked[1]]),
                "tracked_upper_frequency_Hz": float(
                    f1 * freq[tracked[2]] / freq[tracked[1]]
                ),
            }
        )
    return {
        "central_branch": assigned["branch"],
        "central_ranks": (chosen + 1).tolist(),
        "shape_fractions": shapes,
        "metric": "Both physical displacement fields contribute to the R16 shape fractions with the Euclidean site norm. Tracking uses the fixed central mass inner product, squared and normalized for each pair of vectors.",
        "scope": "The central assignment uses the two training lines. Shape tracking identifies continuations of those vectors. Electronic total angular momentum is not an observable of the tracking metric.",
        "contrast_tests": tests,
    }


def run(stage, output):
    output.mkdir(parents=True, exist_ok=True)
    geometry = sr.geometry_from_stage(stage)
    spaces, bank = restore_bank(geometry, output)
    values = {
        k: r["value"]
        for k, r in sr.read(sr.ROOT / "sr-inputs.json")["observations"].items()
    }
    correspondence = compare_modes(geometry, spaces, values)
    sr.write(output / "shape-predictions.json", correspondence)
    # Open the clock table only after the shape matches and tracked frequencies have been written.
    clock = float(sr.read(sr.ROOT / "sr-references.json")["clock_Hz"]["87"]["value"])
    close = [
        r
        for r in correspondence["contrast_tests"]
        if r["weight_contrast"] in (0.99, 1.0, 1.01)
    ]
    result = {
        "bank": bank,
        "Sr87_central_ranks": correspondence["central_ranks"],
        "Sr87_at_nearby_contrasts": [
            {
                "contrast": r["weight_contrast"],
                "tracked": r["tracked_global_ranks"],
                "frequency_reselection": r["reselected_within_branch_ranks"],
                "minimum_shape_overlap_squared": min(r["shape_overlap_squared"]),
                "tracked_residual_Hz": r["tracked_frequency_Hz"] - clock,
            }
            for r in close
        ],
    }
    assert correspondence["central_branch"] == "low" and correspondence[
        "central_ranks"
    ] == [1, 4, 15]
    assert all(
        r["tracked_global_ranks"] == [1, 4, 15]
        and min(r["shape_overlap_squared"]) > 0.994
        for r in close
    )
    sr.write(output / "correspondence.json", result)
    return result
