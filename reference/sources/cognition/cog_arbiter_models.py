#!/usr/bin/env python3
'BST–COG A1: adversarial toy models for persistence and correction.\n\nThese models are intentionally substrate-neutral and do not load V94.  They\nseparate four questions that are often conflated:\n\n1. Can a process keep an identity-like state persistent?\n2. Can it stop an unstable loop and restore a prior symmetry?\n3. Can it converge to a new, singular target?\n4. Is the target/change ontologically undetermined?\n\nThe first three are functional.  The fourth is not inferred from them.'

from __future__ import annotations

import argparse
import cmath
import json
import math
import random
from dataclasses import asdict, dataclass


def persistence_rate(sequence: list[int]) -> float:
    if len(sequence) < 2:
        return 1.0
    return sum(a == b for a, b in zip(sequence, sequence[1:])) / (len(sequence) - 1)


def occupancy_entropy(sequence: list[int], branch_count: int) -> float:
    counts = [0] * branch_count
    for value in sequence:
        counts[value] += 1
    total = len(sequence)
    entropy = 0.0
    for count in counts:
        if count:
            p = count / total
            entropy -= p * math.log(p)
    return entropy / math.log(branch_count)


def iid_branches(branch_count: int, steps: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(branch_count) for _ in range(steps)]


def sticky_random_branches(
    branch_count: int, steps: int, seed: int, stay_probability: float
) -> list[int]:
    rng = random.Random(seed)
    current = rng.randrange(branch_count)
    out = [current]
    for _ in range(steps - 1):
        if rng.random() >= stay_probability:
            current = rng.randrange(branch_count)
        out.append(current)
    return out


def lcg_branches(branch_count: int, steps: int, seed: int) -> list[int]:
    'A fully deterministic hidden-seed sequence.'
    state = seed & 0xFFFFFFFF
    out = []
    for _ in range(steps):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        out.append((state >> 16) % branch_count)
    return out


def rotate(value: complex, angle: float) -> complex:
    return value * cmath.exp(1j * angle)


@dataclass(frozen=True)
class TrajectoryMetrics:
    model: str
    final_error: float
    maximum_error: float
    return_delay_after_shock: int | None
    replay_identical: bool | None
    target_switch_error: float | None


def unstable_spiral(
    steps: int = 400,
    gain: float = 1.03,
    angle: float = 0.37,
    initial: complex = 1e-6 + 0j,
) -> list[complex]:
    out = [initial]
    for _ in range(steps):
        out.append(gain * rotate(out[-1], angle))
    return out


def noisy_unstable_spiral(
    seed: int,
    steps: int = 400,
    gain: float = 1.03,
    angle: float = 0.37,
    noise_scale: float = 1e-4,
) -> list[complex]:
    rng = random.Random(seed)
    out = [0j]
    for _ in range(steps):
        noise = noise_scale * complex(rng.gauss(0, 1), rng.gauss(0, 1))
        out.append(gain * rotate(out[-1], angle) + noise)
    return out


def corrective_spiral(
    steps: int = 400,
    angle: float = 0.37,
    contraction: float = 0.82,
    shock_step: int = 80,
    shock: complex = 1 + 0.5j,
    switch_step: int | None = None,
    new_target: complex = 2 + 1j,
) -> tuple[list[complex], list[complex]]:
    'A deterministic, memory-bearing countermodel to metaphysical inference.\n\nIt restores a selected target and can switch to a new one, proving that\nthose functional signatures alone do not establish ontic freedom.'
    state = 0j
    target = 0j
    states = [state]
    targets = [target]
    for step in range(1, steps + 1):
        if step == shock_step:
            state += shock
        if switch_step is not None and step == switch_step:
            target = new_target
        error = state - target
        state = target + contraction * rotate(error, angle)
        states.append(state)
        targets.append(target)
    return states, targets


def first_return_delay(
    states: list[complex], target: complex, shock_step: int, tolerance: float
) -> int | None:
    for step in range(shock_step, len(states)):
        if abs(states[step] - target) <= tolerance:
            return step - shock_step
    return None


def trajectory_metrics() -> list[TrajectoryMetrics]:
    open_loop = unstable_spiral()
    noisy_a = noisy_unstable_spiral(seed=20260817)
    noisy_replay = noisy_unstable_spiral(seed=20260817)
    restored, _ = corrective_spiral()
    retargeted, targets = corrective_spiral(switch_step=160)

    return [
        TrajectoryMetrics(
            model="deterministic_unstable_open_loop",
            final_error=abs(open_loop[-1]),
            maximum_error=max(map(abs, open_loop)),
            return_delay_after_shock=None,
            replay_identical=True,
            target_switch_error=None,
        ),
        TrajectoryMetrics(
            model="iid_random_kicks_on_unstable_loop",
            final_error=abs(noisy_a[-1]),
            maximum_error=max(map(abs, noisy_a)),
            return_delay_after_shock=None,
            replay_identical=noisy_a == noisy_replay,
            target_switch_error=None,
        ),
        TrajectoryMetrics(
            model="deterministic_corrector_restore",
            final_error=abs(restored[-1]),
            maximum_error=max(map(abs, restored)),
            return_delay_after_shock=first_return_delay(restored, 0j, 80, 1e-3),
            replay_identical=True,
            target_switch_error=None,
        ),
        TrajectoryMetrics(
            model="deterministic_corrector_retarget",
            final_error=abs(retargeted[-1] - targets[-1]),
            maximum_error=max(abs(x - m) for x, m in zip(retargeted, targets)),
            return_delay_after_shock=None,
            replay_identical=True,
            target_switch_error=abs(retargeted[-1] - (2 + 1j)),
        ),
    ]


def run_audit(branch_count: int = 8, steps: int = 100_000) -> dict:
    iid = iid_branches(branch_count, steps, seed=20260817)
    sticky = sticky_random_branches(
        branch_count, steps, seed=20260817, stay_probability=0.99
    )
    seeded_a = lcg_branches(branch_count, steps, seed=20260817)
    seeded_b = lcg_branches(branch_count, steps, seed=20260817)
    seeded_c = lcg_branches(branch_count, steps, seed=20260818)
    trajectories = trajectory_metrics()
    by_name = {row.model: row for row in trajectories}

    report = {
        "scope": {
            "branch_count": branch_count,
            "branch_steps": steps,
            "physical_V94_imported": False,
            "models_are_toy_countermodels": True,
        },
        "branch_models": {
            "iid_random": {
                "persistence_rate": persistence_rate(iid),
                "normalized_occupancy_entropy": occupancy_entropy(iid, branch_count),
            },
            "sticky_random_with_memory_rule": {
                "persistence_rate": persistence_rate(sticky),
                "normalized_occupancy_entropy": occupancy_entropy(sticky, branch_count),
            },
            "hidden_seed_lcg": {
                "persistence_rate": persistence_rate(seeded_a),
                "normalized_occupancy_entropy": occupancy_entropy(seeded_a, branch_count),
                "same_seed_exact_replay": seeded_a == seeded_b,
                "different_seed_different_history": seeded_a != seeded_c,
            },
        },
        "trajectory_models": [asdict(row) for row in trajectories],
        "criteria": {
            "iid_random_has_only_chance_level_identity_persistence": abs(
                persistence_rate(iid) - 1 / branch_count
            ) < 0.01,
            "randomness_plus_memory_can_be_persistent": persistence_rate(sticky) > 0.98,
            "hidden_seed_is_exactly_replayable": seeded_a == seeded_b,
            "unstable_loop_amplifies_nonzero_asymmetry": (
                by_name["deterministic_unstable_open_loop"].final_error > 0.1
            ),
            "random_kicks_do_not_restore_the_unstable_loop": (
                by_name["iid_random_kicks_on_unstable_loop"].final_error > 0.1
            ),
            "deterministic_feedback_restores_after_shock": (
                by_name["deterministic_corrector_restore"].final_error < 1e-12
                and by_name["deterministic_corrector_restore"].return_delay_after_shock
                is not None
            ),
            "deterministic_feedback_can_retarget_new_symmetry": (
                by_name["deterministic_corrector_retarget"].target_switch_error < 1e-12
            ),
        },
    }
    report["pass"] = all(report["criteria"].values())
    report["verdict"] = (
        "FUNCTIONAL_ARBITER_MINIMUM_DERIVED__ONTOLOGICAL_ORIGIN_REMAINS_OPEN"
        if report["pass"]
        else "A1_TOY_AUDIT_FAILED"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-count", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100_000)
    args = parser.parse_args()
    report = run_audit(args.branch_count, args.steps)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
