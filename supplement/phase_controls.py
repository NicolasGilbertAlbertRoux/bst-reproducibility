"""Cross macroscopic order and relative-phase reset in the manuscript's eight-pair model."""
import itertools
import json

import numpy as np

from reference.sources.cognition.compute import DIM_MOT, order_par_pairs, progression


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    words = np.asarray(list(itertools.product((0, 1), repeat=DIM_MOT)))
    initial_phases = np.pi * words
    sites = np.arange(DIM_MOT)
    times = np.linspace(0, 8, 161)
    archived_order = order_par_pairs(times, 4.0)
    cases = []
    maximum_visible_difference = 0.0
    maximum_fine_difference = 0.0
    for disorder in (False, True):
        for reset in (False, True):
            for step, t in enumerate(times):
                # Aliases and the drive are the same as in calculation_pairs in the archived replay.
                spreading = progression(t / 4) if disorder else 0.0
                orientation = t + 2 * np.pi * sites * spreading / DIM_MOT
                delta = (1 - progression(t - 4)) * initial_phases if reset else initial_phases
                plus = np.exp(0.5j * (orientation + delta))
                minus = np.exp(0.5j * (orientation - delta))
                visible = plus * minus
                fine = plus * minus.conj()
                expected_visible = np.exp(1j * orientation)
                expected_fine = np.exp(1j * delta)
                maximum_visible_difference = max(maximum_visible_difference, float(np.max(abs(visible - expected_visible))))
                maximum_fine_difference = max(maximum_fine_difference, float(np.max(abs(fine - expected_fine))))
                order = abs(visible.mean(axis=1))
                np.testing.assert_allclose(order, archived_order[step] if disorder else 1., atol=3e-15, rtol=0)
            decoded = (fine.real < 0).astype(int)
            np.testing.assert_array_equal(decoded, np.zeros_like(words) if reset else words)
            count = len(np.unique(decoded, axis=0))
            # A reset also makes the full terminal pair coordinates identical for every input.
            if reset:
                assert np.all(plus == plus[0]) and np.all(minus == minus[0])
            cases.append({"collective_disordering": disorder, "relative_phase_reset": reset,
                          "final_order_max": float(max(order)), "distinct_final_words": count})
    assert maximum_visible_difference < 3e-15 and maximum_fine_difference < 3e-15
    result = {"initial_words": len(words), "times_per_word_per_case": len(times), "cases": cases,
              "maximum_visible_channel_error": maximum_visible_difference,
              "maximum_relative_phase_channel_error": maximum_fine_difference,
              "scope": "Independent imposed controls in a finite wave model; no biological death observed."}
    (output / "four-controls.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
