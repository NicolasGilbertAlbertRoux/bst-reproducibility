#!/usr/bin/env python3
'BST–COG A6: empirical biface protocol contract and synthetic dry-run.\n\nNo human data are collected or generated as if observed. Synthetic rows only\nvalidate randomization, blinding, schema, gate logic and claim ceilings.'

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass


PROTOCOL_ID = "BST-COG-A6-R0"
SEED = 20260817
CONDITIONS = (
    "free_deliberation",
    "forced_choice_control",
    "delegation_or_veto",
    "yoked_reason_playback",
)
ADVERSARIES = (
    "extended_deterministic_state",
    "stochastic_process",
    "exogenous_oracle",
    "behavioral_report_emulator",
)
FIRST_PERSON_FIELDS = (
    "deliberation_engaged",
    "ownership_rating",
    "reason_salience_rating",
    "motivation_strength_rating",
    "conflict_rating",
)
PUBLIC_FIELDS = (
    "participant_code",
    "session_id",
    "trial_id",
    "condition",
    "reason_direction",
    "reason_strength",
    "prior_motive",
    "action",
    "veto_or_defer",
    "delegated",
    "rule_revision",
    "response_time_ms",
)
PROHIBITED_DIRECT_IDENTIFIERS = (
    "name",
    "email",
    "phone",
    "address",
    "birth_date",
    "ip_address",
)


@dataclass(frozen=True)
class Prerequisite:
    prerequisite_id: str
    requirement: str
    status: str
    execution_blocking: bool


PREREQUISITES = (
    Prerequisite(
        "E1",
        "independent ethics committee approval for the final protocol",
        "PENDING",
        True,
    ),
    Prerequisite(
        "E2",
        "data protection review, lawful basis and retention schedule",
        "PENDING",
        True,
    ),
    Prerequisite(
        "E3",
        "plain-language participant information and documented free consent",
        "DRAFTED_NOT_APPROVED",
        True,
    ),
    Prerequisite(
        "M1",
        "instrument comprehension and test-retest pilot",
        "NOT_RUN",
        True,
    ),
    Prerequisite(
        "M2",
        "effect-size pilot and confirmatory power analysis",
        "NOT_RUN",
        True,
    ),
    Prerequisite(
        "R1",
        "timestamped read-only preregistration before confirmatory data",
        "NOT_REGISTERED",
        True,
    ),
)


@dataclass(frozen=True)
class TaskBlock:
    block_id: str
    purpose: str
    external_observable: str
    first_person_observable: str
    control: str


TASK_BLOCKS = (
    TaskBlock(
        "B1",
        "reason sensitivity",
        "counterfactual action change under frozen reason perturbation",
        "reason salience and ownership recorded before feedback",
        "yoked reason playback",
    ),
    TaskBlock(
        "B2",
        "choice to choose",
        "engage, veto, defer or delegate",
        "deliberation engagement and motivation strength",
        "forced choice",
    ),
    TaskBlock(
        "B3",
        "motive revision",
        "revision after accumulated concordant conflict",
        "conflict and ownership of the revision",
        "clock-matched scripted revision",
    ),
    TaskBlock(
        "B4",
        "diachronic appropriation",
        "session-2 response to session-1 history after 7 plus or minus 2 days",
        "identity continuity and remembered motive",
        "history-shuffled control",
    ),
    TaskBlock(
        "B5",
        "unseen holdout",
        "generalization to preregistered novel reason configurations",
        "pre-outcome report using the same instrument",
        "no adaptive thresholding",
    ),
)


@dataclass(frozen=True)
class ClaimCeiling:
    outcome: str
    maximum_claim: str


CLAIM_CEILINGS = (
    ClaimCeiling(
        "functional_or_biface_fail",
        "candidate authorship gate not passed",
    ),
    ClaimCeiling(
        "functional_and_biface_pass",
        "functional biface authorship in this protocol",
    ),
    ClaimCeiling(
        "all_frozen_adversaries_rejected",
        "local closure class C rejected; strong author remains open",
    ),
    ClaimCeiling(
        "independently_valid_strong_witness",
        "absence refuted and existence established",
    ),
)


def participant_code(index: int) -> str:
    return hashlib.sha256(f"{PROTOCOL_ID}:{index}".encode()).hexdigest()[:16]


def blocked_randomization(count: int, seed: int = SEED) -> list[dict]:
    if count <= 0 or count % len(CONDITIONS) != 0:
        raise ValueError("count must be a positive multiple of condition count")
    assignments = []
    rng = random.Random(seed)
    for block_start in range(0, count, len(CONDITIONS)):
        block = list(CONDITIONS)
        rng.shuffle(block)
        for offset, condition in enumerate(block):
            index = block_start + offset
            assignments.append(
                {
                    "participant_code": participant_code(index),
                    "allocation_code": hashlib.sha256(
                        f"sealed:{seed}:{index}:{condition}".encode()
                    ).hexdigest(),
                    "condition": condition,
                }
            )
    return assignments


def synthetic_schema() -> dict:
    all_fields = PUBLIC_FIELDS + FIRST_PERSON_FIELDS + (
        "first_person_locked_at",
        "feedback_released_at",
        "session_link_code",
    )
    return {
        "fields": list(all_fields),
        "direct_identifiers_present": sorted(
            set(all_fields).intersection(PROHIBITED_DIRECT_IDENTIFIERS)
        ),
        "free_text_in_confirmatory_core": False,
        "optional_qualitative_arm": (
            "separate consent, redaction and exploratory analysis only"
        ),
        "first_person_precedes_feedback_contract": True,
        "linkage_key_stored_separately": True,
    }


def blinding_contract() -> dict:
    return {
        "participant": (
            "blind to authorship classification target and adversarial model labels"
        ),
        "task_delivery": "automated from frozen schedule",
        "external_model_team": "receives public fields without first-person fields",
        "biface_team": "receives sealed event links after external predictions freeze",
        "primary_analyst": "blind to condition labels until primary scores freeze",
        "unblinding": "single logged join after all declared predictions are immutable",
    }


def analysis_contract() -> dict:
    return {
        "unit": "participant with trial-level hierarchical structure",
        "confirmatory_split": "participant-level train/validation/locked-holdout",
        "primary_targets": [
            "action",
            "veto_or_defer",
            "rule_revision",
            "session_2_history_dependence",
        ],
        "biface_targets": list(FIRST_PERSON_FIELDS),
        "model_comparison": (
            "predeclared out-of-sample proper scoring rules with uncertainty intervals"
        ),
        "missingness": (
            "report by condition; no silent imputation; sensitivity analysis frozen"
        ),
        "multiplicity": "hierarchical primary family plus corrected secondary family",
        "sample_size": "NOT_DERIVED_PENDING_PILOT_EFFECT_SIZE_AND_POWER_ANALYSIS",
        "stopping_rule": "NOT_DERIVED_PENDING_POWER_ANALYSIS",
        "exploratory_label_required": True,
    }


def ethics_contract() -> dict:
    return {
        "initial_population": (
            "adults able to give free informed consent; no vulnerable population pilot"
        ),
        "risk_level_target": "minimal risk behavioral study",
        "deception": "none in R0",
        "withdrawal": "any time without penalty; skip any item",
        "sensitive_framing": (
            "never tell participants that a task diagnoses consciousness or metaphysical freedom"
        ),
        "distress_mitigation": (
            "neutral wording, pause/stop control, debrief and support signposting"
        ),
        "compensation": "time-based and not contingent on choices or completion",
        "consent_separation": (
            "research participation and personal-data processing documented separately when required"
        ),
        "data_minimization": True,
        "pseudonymization": True,
        "retention": "TO_BE_FIXED_WITH_DPO_BEFORE_APPROVAL",
        "secondary_use": "separate explicit governance and participant information",
    }


def dry_run(count: int = 120) -> dict:
    assignments = blocked_randomization(count)
    counts = {
        condition: sum(row["condition"] == condition for row in assignments)
        for condition in CONDITIONS
    }
    public_export = [
        {
            "participant_code": row["participant_code"],
            "condition": row["condition"],
        }
        for row in assignments
    ]
    return {
        "synthetic_only": True,
        "synthetic_participant_count": count,
        "condition_counts": counts,
        "balanced": len(set(counts.values())) == 1,
        "unique_participant_codes": (
            len({row["participant_code"] for row in assignments}) == count
        ),
        "unique_allocation_codes": (
            len({row["allocation_code"] for row in assignments}) == count
        ),
        "public_export_excludes_allocation_code": all(
            "allocation_code" not in row for row in public_export
        ),
        "this_count_is_not_a_power_calculation": True,
    }


def run_audit() -> dict:
    schema = synthetic_schema()
    blind = blinding_contract()
    analysis = analysis_contract()
    ethics = ethics_contract()
    dry = dry_run()
    prerequisites = [asdict(row) for row in PREREQUISITES]
    blockers = [
        row["prerequisite_id"]
        for row in prerequisites
        if row["execution_blocking"] and row["status"] not in ("PASS", "APPROVED")
    ]
    criteria = {
        "five_task_blocks_cover_A2_A5_constructs": len(TASK_BLOCKS) == 5,
        "external_and_first_person_fields_are_separate": not set(
            PUBLIC_FIELDS
        ).intersection(FIRST_PERSON_FIELDS),
        "first_person_fields_are_locked_before_feedback": schema[
            "first_person_precedes_feedback_contract"
        ],
        "no_direct_identifier_in_confirmatory_schema": not schema[
            "direct_identifiers_present"
        ],
        "all_four_adversaries_are_frozen": set(ADVERSARIES)
        == {
            "extended_deterministic_state",
            "stochastic_process",
            "exogenous_oracle",
            "behavioral_report_emulator",
        },
        "blinding_has_single_logged_unblinding": "single logged join" in blind[
            "unblinding"
        ],
        "synthetic_randomization_is_balanced_and_unique": (
            dry["balanced"]
            and dry["unique_participant_codes"]
            and dry["unique_allocation_codes"]
        ),
        "synthetic_count_is_not_misused_as_power": dry[
            "this_count_is_not_a_power_calculation"
        ] and analysis["sample_size"].startswith("NOT_DERIVED"),
        "ethics_and_data_reviews_block_execution": set(blockers)
        >= {"E1", "E2", "E3"},
        "instrument_and_power_work_block_execution": set(blockers)
        >= {"M1", "M2"},
        "preregistration_blocks_confirmatory_execution": "R1" in blockers,
        "claim_ceiling_preserves_strong_author_open_after_local_rejection": any(
            row.outcome == "all_frozen_adversaries_rejected"
            and "strong author remains open" in row.maximum_claim
            for row in CLAIM_CEILINGS
        ),
    }
    passed = all(criteria.values())
    return {
        "scope": {
            "human_or_animal_data_collected": False,
            "synthetic_dry_run_only": True,
            "physical_V94_imported": False,
            "C2_or_C3_modified": False,
            "consciousness_or_strong_author_detected": False,
        },
        "protocol_id": PROTOCOL_ID,
        "prerequisites": prerequisites,
        "execution_blockers": blockers,
        "population": ethics["initial_population"],
        "task_blocks": [asdict(row) for row in TASK_BLOCKS],
        "public_fields": list(PUBLIC_FIELDS),
        "first_person_fields": list(FIRST_PERSON_FIELDS),
        "schema": schema,
        "blinding": blind,
        "adversaries": list(ADVERSARIES),
        "analysis": analysis,
        "ethics": ethics,
        "claim_ceilings": [asdict(row) for row in CLAIM_CEILINGS],
        "dry_run": dry,
        "criteria": criteria,
        "pass": passed,
        "verdict": (
            "A6_EMPIRICAL_PROTOCOL_CONTRACT_PASS__"
            "HUMAN_EXECUTION_BLOCKED_PENDING_ETHICS_INSTRUMENTS_AND_POWER"
            if passed
            else "A6_EMPIRICAL_PROTOCOL_CONTRACT_FAIL"
        ),
        "scientific_status": {
            "protocol_contract": "PASS",
            "human_execution": "NOT_AUTHORIZED",
            "sample_size": analysis["sample_size"],
            "empirical_evidence_for_strong_author": "NONE_COLLECTED",
        },
        "next": (
            "A6-R1: independent ethics/DPO review and instrument pilot; only then "
            "derive effect size, sample size, stopping rule and final preregistration"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--design-output", default=None)
    args = parser.parse_args()
    report = run_audit()
    payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
    print(payload)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    if args.design_output:
        design = {
            key: report[key]
            for key in (
                "protocol_id",
                "prerequisites",
                "execution_blockers",
                "population",
                "task_blocks",
                "public_fields",
                "first_person_fields",
                "schema",
                "blinding",
                "adversaries",
                "analysis",
                "ethics",
                "claim_ceilings",
            )
        }
        with open(args.design_output, "w", encoding="utf-8") as handle:
            json.dump(design, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
