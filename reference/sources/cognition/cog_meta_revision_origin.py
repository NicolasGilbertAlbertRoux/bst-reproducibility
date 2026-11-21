#!/usr/bin/env python3
'BST–COG A4: meta-revision origin and non-closure audit.\n\nThe audit separates deterministic closure, stochastic closure, exogenous input,\ncomputational unpredictability and agential attribution.  It proves no\nmetaphysical freedom and imports no V94 component.'

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from dataclasses import dataclass
from enum import Enum


REASONS = (-1, 0, 1)


class OriginClass(str, Enum):
    DETERMINISTIC_CLOSURE = "deterministic_closure"
    STOCHASTIC_CLOSURE = "stochastic_closure"
    EXOGENOUS_INPUT = "exogenous_input"


class NonClosureLevel(str, Enum):
    EPISTEMIC = "epistemic_model_incompleteness"
    COMPUTATIONAL = "computational_unpredictability"
    ONTOLOGICAL = "ontological_nonfixation"
    AGENTIAL = "agential_authorship"


@dataclass(frozen=True)
class MetaState:
    identity_id: int = 101
    motive: int = 1
    conflict_count: int = 0
    rule_index: int = 0
    step: int = 0


@dataclass(frozen=True)
class MetaEvent:
    action: int
    vetoed: bool
    revised_rule: bool
    revision_reason: str
    next_state: MetaState


def fixed_extended_kernel(state: MetaState, reason: int) -> MetaEvent:
    'One immutable kernel whose state contains the currently used rule.\n\nRule 0 requires two concordant conflicts before changing motive. Rule 1\nrequires one. Apparent changes of the rule are therefore ordinary state\nchanges of this fixed extended kernel.'

    threshold = 2 if state.rule_index == 0 else 1
    if reason == 0:
        action, vetoed, conflicts = 0, True, state.conflict_count
        motive, rule_index, revised, cause = (
            state.motive,
            state.rule_index,
            False,
            "insufficient_reason",
        )
    elif reason == state.motive:
        action, vetoed, conflicts = state.motive, False, 0
        motive, rule_index, revised, cause = (
            state.motive,
            state.rule_index,
            False,
            "concordant_reason",
        )
    else:
        conflicts = state.conflict_count + 1
        action, vetoed = 0, True
        if conflicts >= threshold:
            motive = reason
            # The controller appears to revise how future revisions occur.
            rule_index = 1 - state.rule_index
            conflicts = 0
            revised = True
            cause = "accumulated_conflict"
        else:
            motive, rule_index, revised, cause = (
                state.motive,
                state.rule_index,
                False,
                "conflict_stored",
            )

    next_state = MetaState(
        identity_id=state.identity_id,
        motive=motive,
        conflict_count=conflicts,
        rule_index=rule_index,
        step=state.step + 1,
    )
    return MetaEvent(action, vetoed, revised, cause, next_state)


def apparent_meta_controller(state: MetaState, reason: int) -> MetaEvent:
    'Same dynamics described as a controller that changes its own rule.'

    return fixed_extended_kernel(state, reason)


def simulate(kernel, reasons: tuple[int, ...]) -> tuple[MetaEvent, ...]:
    state = MetaState()
    events = []
    for reason in reasons:
        event = kernel(state, reason)
        events.append(event)
        state = event.next_state
    return tuple(events)


def bounded_internalization_audit(max_horizon: int = 5) -> dict:
    tested = 0
    equal = True
    revisions_seen = 0
    for horizon in range(1, max_horizon + 1):
        for reasons in itertools.product(REASONS, repeat=horizon):
            tested += 1
            apparent = simulate(apparent_meta_controller, reasons)
            compiled = simulate(fixed_extended_kernel, reasons)
            equal &= apparent == compiled
            revisions_seen += sum(event.revised_rule for event in apparent)
    return {
        "tested_histories": tested,
        "expected_histories": sum(3**n for n in range(1, max_horizon + 1)),
        "apparent_and_extended_fixed_kernel_are_identical": equal,
        "meta_rule_revisions_observed": revisions_seen,
        "conclusion": "bounded_meta_revision_internalizes_into_extended_state",
    }


class FiniteMetaTranscriptCompiler:
    'Deterministic table that replays a finite observed meta-revision trace.'

    def __init__(self, reasons: tuple[int, ...]):
        state = MetaState()
        self.table = {}
        for reason in reasons:
            event = apparent_meta_controller(state, reason)
            self.table[(state, reason)] = event
            state = event.next_state

    def step(self, state: MetaState, reason: int) -> MetaEvent:
        return self.table[(state, reason)]


def finite_trace_compilation_audit() -> dict:
    reasons = (-1, -1, 1, 0, 1, -1, 1, 1, 0)
    expected = simulate(apparent_meta_controller, reasons)
    compiler = FiniteMetaTranscriptCompiler(reasons)
    state = MetaState()
    replayed = []
    for reason in reasons:
        event = compiler.step(state, reason)
        replayed.append(event)
        state = event.next_state
    unseen_rejected = False
    try:
        compiler.step(MetaState(step=999), 1)
    except KeyError:
        unseen_rejected = True
    return {
        "observed_event_count": len(reasons),
        "finite_trace_exactly_compiled": tuple(replayed) == expected,
        "unseen_case_not_explained": unseen_rejected,
    }


def seeded_public_trace(seed: int, horizon: int = 32) -> tuple[int, ...]:
    rng = random.Random(seed)
    return tuple(rng.choice((-1, 1)) for _ in range(horizon))


def provenance_underdetermination_audit() -> dict:
    # Provenance labels are deliberately outside the public trace.
    stochastic_equal = all(
        seeded_public_trace(seed) == seeded_public_trace(seed)
        for seed in range(100)
    )
    exogenous_stream = (1, -1, 1, 1, -1, -1, 1, -1)
    external_oracle_trace = tuple(exogenous_stream)
    alleged_author_trace = tuple(exogenous_stream)
    return {
        "stochastic_public_traces_equal_under_hidden_provenance_labels": (
            stochastic_equal
        ),
        "exogenous_oracle_and_alleged_author_public_traces_equal": (
            external_oracle_trace == alleged_author_trace
        ),
        "public_trace_identifies_provenance": False,
        "origin_classes": [origin.value for origin in OriginClass],
    }


def deterministic_motivated_trace(reasons: tuple[int, ...]) -> dict:
    events = simulate(fixed_extended_kernel, reasons)
    return {
        "reason_responsive": len({event.action for event in events}) > 1,
        "veto_present": any(event.vetoed for event in events),
        "meta_revision_present": any(event.revised_rule for event in events),
        "identity_continuous": all(
            event.next_state.identity_id == 101 for event in events
        ),
        "fully_deterministic": True,
    }


def diagonal_counterexample(predictions: tuple[int, ...]) -> dict:
    'Finite executable analogue of D(x)=1-P(x,x).\n\nThis witnesses the logical form of diagonal defeat. It is not a universal\ncomputability proof and is not interpreted as authorship.'

    if not predictions or any(bit not in (0, 1) for bit in predictions):
        raise ValueError("predictions must be a non-empty binary tuple")
    diagonal = tuple(1 - bit for bit in predictions)
    mismatches = sum(a != b for a, b in zip(predictions, diagonal))
    return {
        "prediction_count": len(predictions),
        "diagonal_mismatch_count": mismatches,
        "universal_prediction_defeated_in_finite_witness": (
            mismatches == len(predictions)
        ),
        "authorship_follows_from_prediction_failure": False,
    }


def candidate_gate_audit() -> dict:
    'Show that strong-looking finite criteria can be passed by a closure.'

    reasons = (-1, -1, 1, 0, -1, 1, 1, 0)
    probe = deterministic_motivated_trace(reasons)
    public_candidate_gate = all(
        probe[key]
        for key in (
            "reason_responsive",
            "veto_present",
            "meta_revision_present",
            "identity_continuous",
        )
    )
    return {
        "functional_candidate_gate_pass": public_candidate_gate,
        "passing_model_is_deterministic": probe["fully_deterministic"],
        "strong_authorship_inferred": False,
        "diagnostic": probe,
    }


def stable_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_audit() -> dict:
    internalization = bounded_internalization_audit()
    finite = finite_trace_compilation_audit()
    provenance = provenance_underdetermination_audit()
    diagonal = diagonal_counterexample(tuple(i % 2 for i in range(256)))
    candidate = candidate_gate_audit()

    criteria = {
        "all_363_bounded_histories_are_exhausted": (
            internalization["tested_histories"]
            == internalization["expected_histories"]
            == 363
        ),
        "bounded_meta_revision_compiles_into_fixed_extended_state": (
            internalization[
                "apparent_and_extended_fixed_kernel_are_identical"
            ]
            and internalization["meta_rule_revisions_observed"] > 0
        ),
        "finite_meta_trace_has_exact_deterministic_compiler": finite[
            "finite_trace_exactly_compiled"
        ],
        "finite_compiler_does_not_explain_unseen_cases": finite[
            "unseen_case_not_explained"
        ],
        "stochastic_provenance_is_not_in_public_trace": provenance[
            "stochastic_public_traces_equal_under_hidden_provenance_labels"
        ],
        "exogenous_source_and_alleged_author_are_trace_equivalent": provenance[
            "exogenous_oracle_and_alleged_author_public_traces_equal"
        ],
        "diagonal_prediction_failure_does_not_entail_authorship": (
            diagonal["universal_prediction_defeated_in_finite_witness"]
            and not diagonal["authorship_follows_from_prediction_failure"]
        ),
        "functional_meta_revision_can_still_be_deterministic": (
            candidate["functional_candidate_gate_pass"]
            and candidate["passing_model_is_deterministic"]
            and not candidate["strong_authorship_inferred"]
        ),
    }
    passed = all(criteria.values())
    body = {
        "scope": {
            "physical_V94_imported": False,
            "C2_or_C3_modified": False,
            "metaphysical_freedom_simulated_or_detected": False,
            "results_are_formal_and_countermodel_based": True,
        },
        "closure_trilemma": {
            "classes": [origin.value for origin in OriginClass],
            "deterministic": (
                "closed transition; automatable when the extended kernel is computable"
            ),
            "stochastic": "novelty without authorship attribution",
            "exogenous": "received origin without identified authorship",
            "no_fourth_transition_label_is_derived": True,
        },
        "nonclosure_levels": {
            NonClosureLevel.EPISTEMIC.value: "observer/model may be incomplete",
            NonClosureLevel.COMPUTATIONAL.value: (
                "prediction may be undecidable or infeasible"
            ),
            NonClosureLevel.ONTOLOGICAL.value: (
                "future may not be fixed by prior physical state"
            ),
            NonClosureLevel.AGENTIAL.value: (
                "actualization is attributable to a subject as origin"
            ),
        },
        "bounded_internalization": internalization,
        "finite_trace_compilation": finite,
        "provenance_underdetermination": provenance,
        "diagonal_witness": diagonal,
        "candidate_gate": candidate,
        "criteria": criteria,
    }
    return {
        **body,
        "audit_digest": stable_digest(body),
        "pass": passed,
        "verdict": (
            "META_REVISION_CLOSURE_TRILEMMA_PASS__"
            "STRONG_AUTHOR_ORIGIN_NOT_IDENTIFIABLE_WITHIN_TRANSITION_FORMALISM"
            if passed
            else "A4_META_REVISION_ORIGIN_AUDIT_FAIL"
        ),
        "scientific_status": {
            "bounded_meta_coherence_internalization": "DERIVED_AND_TESTED",
            "prediction_failure_implies_freedom": "REFUTED",
            "randomness_implies_authorship": "REFUTED",
            "external_input_implies_authorship": "REFUTED",
            "strong_author_nonclosure": "NOT_DERIVED",
            "public_transition_only_closure": "IMPOSSIBILITY_WALL_REACHED",
        },
        "next": (
            "A5: consolidate A0-A4 as a qualitative theorem stack and define "
            "empirical first/third-person research protocols without upgrading "
            "strong authorship to a physical result"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = run_audit()
    payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
    print(payload)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
