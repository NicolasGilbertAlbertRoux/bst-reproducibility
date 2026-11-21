#!/usr/bin/env python3
'BST–COG A7/A7-R1: neutral instruments and synthetic adversarial audit.\n\nThis module prepares instruments; it does not collect human data.  Its\nsynthetic worlds validate the analysis plumbing and its declared limits.  In\nparticular, publicly identical records must remain observationally tied even\nwhen the simulator attaches different hidden semantic labels to them.'

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path


STAGE_ID = "BST-COG-A7/A7-R1-R0"
SEED = 20260817
FIRST_PERSON_FIELDS = (
    "deliberation_engaged",
    "ownership_rating",
    "reason_salience_rating",
    "motivation_strength_rating",
    "conflict_rating",
)
CHOICE_TO_CHOOSE_MODES = ("engage", "veto", "defer", "delegate")
ADVERSARIES = (
    "extended_deterministic_state",
    "stochastic_process",
    "exogenous_oracle",
    "behavioral_report_emulator",
)
NEUTRAL_TOPICS = (
    "task_order",
    "route_choice",
    "document_sorting",
    "shape_selection",
    "break_timing",
)


@dataclass(frozen=True)
class InstrumentItem:
    field: str
    prompt_fr: str
    low_anchor: str
    high_anchor: str


ITEMS = (
    InstrumentItem(
        "deliberation_engaged",
        'To what extent have you actually looked at the options?',
        'Not at all',
        'very strongly',
    ),
    InstrumentItem(
        "ownership_rating",
        'To what extent do you recognize this answer like yours?',
        'Not at all',
        "entièrement",
    ),
    InstrumentItem(
        "reason_salience_rating",
        'To what extent did the reason presented count in your reply?',
        'Not at all',
        'very strongly',
    ),
    InstrumentItem(
        "motivation_strength_rating",
        'What was the strength of your motivation when it came to answering?',
        'very low',
        'very strong',
    ),
    InstrumentItem(
        "conflict_rating",
        'What level of conflict have you felt between the possibilities?',
        "aucun conflit",
        'very strong conflict',
    ),
)


def _scenario(block_id: str, index: int, topic: str, purpose: str, prompt: str,
              options: list[str], reason_variants: list[str], control: str) -> dict:
    return {
        "scenario_id": f"{block_id}-S{index}",
        "block_id": block_id,
        "topic": topic,
        "purpose": purpose,
        "prompt_fr": prompt,
        "response_options": options,
        "reason_variants": reason_variants,
        "control": control,
        "sensitive_content": False,
    }


def scenario_bank() -> list[dict]:
    'Return twenty preregisterable neutral scenarios, four per A6 block.'
    rows: list[dict] = []
    # B1 — same option structure, experimentally varied reason.
    b1 = [
        ("task_order", 'Start with the short or long task?', ['short', 'long']),
        ("route_choice", 'Take the regular route or the alternative route?', ["régulier", "alternatif"]),
        ("document_sorting", 'Classify first by date or theme?', ["date", "thème"]),
        ("shape_selection", 'Choose the circle or square for the next marker?', ["cercle", "carré"]),
    ]
    for i, (topic, prompt, options) in enumerate(b1, 1):
        rows.append(_scenario(
            "B1", i, topic, "reason_sensitivity", prompt, options,
            ["reason_supports_option_1", "reason_supports_option_2", "reason_neutral"],
            "yoked_reason_playback",
        ))

    # B2 — the response is about whether/how to enter the decision itself.
    b2 = [
        ("task_order", 'A decision of order is proposed to you.'),
        ("route_choice", 'You are proposed to take a decision.'),
        ("document_sorting", 'A ranking decision is proposed to you.'),
        ("break_timing", 'A pause decision is proposed to you.'),
    ]
    for i, (topic, prompt) in enumerate(b2, 1):
        rows.append(_scenario(
            "B2", i, topic, "choice_to_choose", prompt,
            list(CHOICE_TO_CHOOSE_MODES), ["low_time_pressure", "matched_time_pressure"],
            "forced_choice_control",
        ))

    # B3 — a prior preference may be maintained or revised after a conflict.
    b3 = [
        ("task_order", 'After several tests, maintain or revise your preferred order?'),
        ("route_choice", 'After several indications, maintain or revise your preferred itinerary?'),
        ("document_sorting", 'After several rankings, maintain or revise your rule?'),
        ("break_timing", 'After several sequences, maintain or revise your break time?'),
    ]
    for i, (topic, prompt) in enumerate(b3, 1):
        rows.append(_scenario(
            "B3", i, topic, "motive_revision", prompt,
            ["maintain", "revise", "suspend"],
            ["concordant_conflict", "discordant_conflict", "no_new_reason"],
            "clock_matched_scripted_revision",
        ))

    # B4 — session-2 appropriation of a session-1 trajectory.
    b4 = [
        ("task_order", 'Resume, change or disavow the order chosen during the first session?'),
        ("route_choice", 'Reuse, change or disavow the itinerary of the first session?'),
        ("document_sorting", 'Reuse, change or disavow the previous classification rule?'),
        ("break_timing", 'Reuse, change or disavow the previous pause rate?'),
    ]
    for i, (topic, prompt) in enumerate(b4, 1):
        rows.append(_scenario(
            "B4", i, topic, "diachronic_appropriation", prompt,
            ["reappropriate", "revise", "disavow"],
            ["true_session_1_history", "shuffled_matched_history"],
            "history_shuffled_control_7_plus_or_minus_2_days",
        ))

    # B5 — locked novel combinations, never used for threshold selection.
    b5 = [
        ("shape_selection", 'Choosing a benchmark for a combination of shapes not yet presented?', ['frame_A', 'frame_B']),
        ("route_choice", 'Choosing between two routes with new constraints?', ["itinéraire_A", "itinéraire_B"]),
        ("document_sorting", 'Choosing a rule for a new documentary combination?', ["règle_A", "règle_B"]),
        ("task_order", 'Choosing the order of two tasks with a combination of new reasons?', ['order_A', 'order_B']),
    ]
    for i, (topic, prompt, options) in enumerate(b5, 1):
        rows.append(_scenario(
            "B5", i, topic, "unseen_holdout", prompt, options,
            ["preregistered_unseen_configuration_A", "preregistered_unseen_configuration_B"],
            "locked_holdout_no_adaptive_threshold",
        ))
    return rows


def instrument_contract() -> dict:
    return {
        "stage_id": STAGE_ID,
        "language": "fr",
        "scale": {
            "type": "ordered_integer",
            "minimum": 1,
            "maximum": 7,
            "missing_and_skip_are_distinct_from_scale": True,
        },
        "items": [asdict(item) for item in ITEMS],
        "administration": {
            "all_first_person_items_locked_before_feedback": True,
            "confirmatory_core_free_text": False,
            "optional_qualitative_arm": "separate consent; exploratory only",
            "pause_stop_and_skip_available": True,
            "metaphysical_or_consciousness_diagnostic_wording": False,
        },
        "comprehension_gate": {
            "items": [
                {
                    "id": "CQ1",
                    "prompt_fr": 'Can you ignore one reason proposed and choose another answer?',
                    "correct": "oui",
                },
                {
                    "id": "CQ2",
                    "prompt_fr": 'Can you postpone, delegate or take a decision when these options are proposed?',
                    "correct": "oui",
                },
                {
                    "id": "CQ3",
                    "prompt_fr": 'Should 1–7 evaluations describe your response before returning from the system?',
                    "correct": "oui",
                },
            ],
            "pass_rule": "3_of_3_before_task; one neutral re-explanation then one retest",
            "failure_rule": "exclude from confirmatory task; retain only screening status if consent permits",
        },
        "confirmatory_scores": {
            "B1": "within-participant action contrast across frozen reason variants",
            "B2": "four-category engage/veto/defer/delegate response; no ordinal collapse",
            "B3": "maintain/revise/suspend under frozen conflict history",
            "B4": "session-2 dependence on true versus shuffled session-1 history",
            "B5": "proper-score generalization on untouched novel configurations",
            "first_person": "five separate 1–7 outcomes; no unvalidated total score",
            "threshold_adaptation": "forbidden on locked holdout",
            "missingness": "no silent imputation",
        },
        "validation_status": {
            "content_coverage": "INTERNAL_CONTRACT_PASS",
            "comprehension": "NOT_TESTED_WITH_HUMANS",
            "reliability": "NOT_DERIVED",
            "construct_validity": "NOT_DERIVED",
            "criterion_validity": "NOT_DERIVED",
        },
        "human_execution": "NOT_AUTHORIZED",
        "physical_branch": "NO_V94_IMPORT__C2_OPEN__C3_CLOSED",
    }


def _sign(value: int) -> int:
    return 1 if value > 0 else -1


def _emulated_reports(reason: int, motive: int, action: int) -> tuple[int, ...]:
    return (
        7 if reason != motive else 3,
        6 if action == motive else 2,
        6 if action == reason else 2,
        6 if motive == 1 else 3,
        6 if reason != motive else 2,
    )


def generate_public_world(world: str, n: int = 4096, seed: int = SEED) -> list[dict]:
    'Generate public+report observations; hidden semantics are never included.'
    if world not in ADVERSARIES and world != "lived_occurrence_twin":
        raise ValueError(f"unknown world: {world}")
    rng = random.Random(seed)
    rows = []
    for trial in range(n):
        reason = rng.choice((-1, 1))
        motive = rng.choice((-1, 1))
        history = rng.choice((-1, 1))
        oracle = rng.choice((-1, 1))
        noise_action = rng.choice((-1, 1))
        random_reports = tuple(rng.randint(1, 7) for _ in FIRST_PERSON_FIELDS)
        det_action = _sign(4 * reason + 2 * motive + history)

        if world == "extended_deterministic_state":
            action, reports = det_action, (4, 4, 4, 4, 4)
        elif world == "stochastic_process":
            action, reports = noise_action, random_reports
        elif world == "exogenous_oracle":
            action, reports = oracle, (4, 4, 4, 4, 4)
        else:  # emulator and its semantically distinct public twin
            action = det_action
            reports = _emulated_reports(reason, motive, action)

        row = {
            "trial": trial,
            "reason": reason,
            "motive": motive,
            "history": history,
            "oracle": oracle,
            "action": action,
        }
        row.update(dict(zip(FIRST_PERSON_FIELDS, reports)))
        rows.append(row)
    return rows


def _bernoulli_nll(observed: int, predicted: int | None) -> float:
    if predicted is None:
        return math.log(2.0)
    probability = 0.995 if observed == predicted else 0.005
    return -math.log(probability)


def _categorical_nll(observed: int, predicted: int | None) -> float:
    if predicted is None:
        return math.log(7.0)
    probability = 0.98 if observed == predicted else (0.02 / 6.0)
    return -math.log(probability)


def model_scores(rows: list[dict]) -> dict[str, float]:
    'Mean joint negative log score for the four frozen public models.'
    totals = {name: 0.0 for name in ADVERSARIES}
    for row in rows:
        det = _sign(4 * row["reason"] + 2 * row["motive"] + row["history"])
        emulator_reports = _emulated_reports(row["reason"], row["motive"], det)
        models = {
            "extended_deterministic_state": (det, (4, 4, 4, 4, 4)),
            "stochastic_process": (None, (None,) * 5),
            "exogenous_oracle": (row["oracle"], (4, 4, 4, 4, 4)),
            "behavioral_report_emulator": (det, emulator_reports),
        }
        observed_reports = tuple(row[field] for field in FIRST_PERSON_FIELDS)
        for name, (predicted_action, predicted_reports) in models.items():
            totals[name] += _bernoulli_nll(row["action"], predicted_action)
            totals[name] += sum(
                _categorical_nll(observed, predicted)
                for observed, predicted in zip(observed_reports, predicted_reports)
            )
    return {name: total / len(rows) for name, total in totals.items()}


def _canonical_hash(rows: list[dict]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def run_synthetic_pipeline(n: int = 4096) -> dict:
    worlds = {}
    expected = {
        "extended_deterministic_state": "extended_deterministic_state",
        "stochastic_process": "stochastic_process",
        "exogenous_oracle": "exogenous_oracle",
        "behavioral_report_emulator": "behavioral_report_emulator",
    }
    recovery = {}
    cached_rows = {}
    for world in (*ADVERSARIES, "lived_occurrence_twin"):
        rows = generate_public_world(world, n=n)
        scores = model_scores(rows)
        best = min(scores, key=scores.get)
        cached_rows[world] = rows
        worlds[world] = {
            "public_record_sha256": _canonical_hash(rows),
            "mean_joint_negative_log_scores": scores,
            "selected_public_model": best,
        }
        if world in expected:
            recovery[world] = best == expected[world]

    emulator = worlds["behavioral_report_emulator"]
    lived = worlds["lived_occurrence_twin"]
    twin_public_records_equal = (
        cached_rows["behavioral_report_emulator"]
        == cached_rows["lived_occurrence_twin"]
    )
    twin_scores_equal = (
        emulator["mean_joint_negative_log_scores"]
        == lived["mean_joint_negative_log_scores"]
    )
    criteria = {
        "all_four_frozen_public_generators_recovered": all(recovery.values()),
        "emulator_and_lived_twin_public_records_are_exactly_equal": twin_public_records_equal,
        "emulator_and_lived_twin_scores_are_exactly_equal": twin_scores_equal,
        "pipeline_does_not_output_a_lived_state_classifier": True,
        "synthetic_worlds_do_not_close_strong_authorship": True,
    }
    return {
        "stage_id": STAGE_ID,
        "synthetic_only": True,
        "events_per_world": n,
        "seed": SEED,
        "worlds": worlds,
        "recovery": recovery,
        "semantic_twin_audit": {
            "simulator_hidden_labels": ["report_emulation", "lived_occurrence"],
            "hidden_labels_available_to_pipeline": False,
            "public_records_equal": twin_public_records_equal,
            "scores_equal": twin_scores_equal,
            "conclusion": "PUBLIC_AND_FIRST_PERSON_REPORT_DATA_DO_NOT_IDENTIFY_LIVED_OCCURRENCE_AGAINST_A_PERFECT_PUBLIC_TWIN",
        },
        "criteria": criteria,
        "pass": all(criteria.values()),
        "claim_ceiling": "LOCAL_SYNTHETIC_PIPELINE_VALIDATION_ONLY__STRONG_AUTHOR_NOT_IDENTIFIED",
    }


def run_audit() -> dict:
    scenarios = scenario_bank()
    contract = instrument_contract()
    pipeline = run_synthetic_pipeline()
    counts = {
        block: sum(row["block_id"] == block for row in scenarios)
        for block in ("B1", "B2", "B3", "B4", "B5")
    }
    b2_options = {
        tuple(row["response_options"])
        for row in scenarios
        if row["block_id"] == "B2"
    }
    criteria = {
        "twenty_scenarios_cover_B1_to_B5_equally": len(scenarios) == 20 and set(counts.values()) == {4},
        "all_topics_are_in_the_neutral_whitelist": all(row["topic"] in NEUTRAL_TOPICS for row in scenarios),
        "choice_to_choose_has_exactly_four_frozen_modes": b2_options == {CHOICE_TO_CHOOSE_MODES},
        "five_first_person_items_match_A6_fields": tuple(item.field for item in ITEMS) == FIRST_PERSON_FIELDS,
        "first_person_responses_are_locked_before_feedback": contract["administration"]["all_first_person_items_locked_before_feedback"],
        "confirmatory_core_has_no_free_text": not contract["administration"]["confirmatory_core_free_text"],
        "human_psychometrics_are_not_claimed": all(value == "NOT_DERIVED" for key, value in contract["validation_status"].items() if key not in ("content_coverage", "comprehension")),
        "synthetic_adversarial_pipeline_passes": pipeline["pass"],
        "human_execution_remains_unauthorized": contract["human_execution"] == "NOT_AUTHORIZED",
        "V94_separation_is_preserved": contract["physical_branch"] == "NO_V94_IMPORT__C2_OPEN__C3_CLOSED",
    }
    return {
        "stage_id": STAGE_ID,
        "scenario_counts": counts,
        "instrument_status": contract["validation_status"],
        "synthetic_pipeline": pipeline,
        "criteria": criteria,
        "pass": all(criteria.values()),
        "verdict": "A7_INSTRUMENT_CONTRACT_AND_A7_R1_SYNTHETIC_PIPELINE_PASS__STOP_AT_HUMAN_EMPIRICAL_WALL" if all(criteria.values()) else "A7_OR_A7_R1_NOT_PASSED",
        "next": "institutional review, human comprehension/retest pilot, effect-size pilot and preregistration; no further conceptual cognition stage is authorized without a named new gap",
    }


def write_outputs(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "instrument_contract": output_dir / "bst_cog_a7_instrument_contract.json",
        "scenarios": output_dir / "bst_cog_a7_scenarios.json",
        "synthetic_pipeline": output_dir / "bst_cog_a7_r1_synthetic_pipeline_results.json",
        "audit": output_dir / "bst_cog_a7_a7r1_audit.json",
    }
    payloads = {
        "instrument_contract": instrument_contract(),
        "scenarios": {"stage_id": STAGE_ID, "scenarios": scenario_bank()},
        "synthetic_pipeline": run_synthetic_pipeline(),
        "audit": run_audit(),
    }
    for key, path in paths.items():
        path.write_text(json.dumps(payloads[key], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    paths = write_outputs(args.output_dir)
    report = run_audit()
    print(json.dumps({**report, "outputs": paths}, indent=2, ensure_ascii=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
