#!/usr/bin/env python3
'BST–COG A3: biface authorship and first/third-person audit.\n\nThe executable part establishes epistemic distinguishability results.  It does\nnot simulate consciousness and imports no physical BST/V94 component.'

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from enum import Enum


REASONS = (-1, 0, 1)
MAX_EXHAUSTIVE_HORIZON = 7


class PhenomenalStatus(str, Enum):
    'Semantic label used to construct countermodels, never an observable.'

    OCCURRENCE = "occurrence"
    ABSENT = "absent"


@dataclass(frozen=True)
class PublicState:
    subject_id: int = 101
    motive: int = 1
    conflict_count: int = 0
    memory: tuple[int, ...] = ()
    step: int = 0


@dataclass(frozen=True)
class PublicEvent:
    event_id: int
    subject_id: int
    reason: int
    action: int
    vetoed: bool
    revised_motive: bool
    report: str


@dataclass(frozen=True)
class ExternalFace:
    event_id: int
    subject_id: int
    reason: int
    action: int
    report: str


@dataclass(frozen=True)
class InternalFace:
    'Formal first-person record; `occurrence` is stipulated, not detected.'

    event_id: int
    subject_id: int
    attended_reason: int
    owned_action: int
    occurrence: PhenomenalStatus


@dataclass(frozen=True)
class BifaceRecord:
    external: ExternalFace
    internal: InternalFace


def deterministic_policy(state: PublicState, reason: int) -> tuple[PublicEvent, PublicState]:
    'The A2-style functional-author policy, fully deterministic.'

    if reason == 0:
        action, vetoed, revised, motive, conflicts = (
            0,
            True,
            False,
            state.motive,
            state.conflict_count,
        )
        report = "Je suspends : raison insuffisante."
    elif reason == state.motive:
        action, vetoed, revised, motive, conflicts = (
            state.motive,
            False,
            False,
            state.motive,
            0,
        )
        report = "J'agis selon la motivation que je reconnais."
    elif state.conflict_count == 0:
        action, vetoed, revised, motive, conflicts = (
            0,
            True,
            False,
            state.motive,
            1,
        )
        report = "Je suspends et conserve ce conflit."
    else:
        action, vetoed, revised, motive, conflicts = (
            0,
            True,
            True,
            reason,
            0,
        )
        report = 'I revise my motivation from the accumulated reasons.'

    event = PublicEvent(
        event_id=state.step,
        subject_id=state.subject_id,
        reason=reason,
        action=action,
        vetoed=vetoed,
        revised_motive=revised,
        report=report,
    )
    next_state = PublicState(
        subject_id=state.subject_id,
        motive=motive,
        conflict_count=conflicts,
        memory=state.memory + (action,),
        step=state.step + 1,
    )
    return event, next_state


class BehavioralTwin:
    'Two semantic worlds with the same complete public transition law.'

    def __init__(self, status: PhenomenalStatus):
        self._status = status

    def run_public(self, reasons: tuple[int, ...]) -> tuple[PublicEvent, ...]:
        state = PublicState()
        events = []
        for reason in reasons:
            event, state = deterministic_policy(state, reason)
            events.append(event)
        return tuple(events)

    def oracle_status(self) -> PhenomenalStatus:
        'Meta-model access excluded from every external observation.'

        return self._status


def all_reason_histories(max_horizon: int = MAX_EXHAUSTIVE_HORIZON):
    for horizon in range(1, max_horizon + 1):
        yield from itertools.product(REASONS, repeat=horizon)


def external_equivalence_audit(max_horizon: int = MAX_EXHAUSTIVE_HORIZON) -> dict:
    with_occurrence = BehavioralTwin(PhenomenalStatus.OCCURRENCE)
    without_occurrence = BehavioralTwin(PhenomenalStatus.ABSENT)
    count = 0
    equal = True
    for history in all_reason_histories(max_horizon):
        count += 1
        if with_occurrence.run_public(history) != without_occurrence.run_public(history):
            equal = False
            break
    return {
        "tested_history_count": count,
        "maximum_horizon": max_horizon,
        "public_traces_identical": equal,
        "self_reports_identical": equal,
        "oracle_statuses_differ": (
            with_occurrence.oracle_status() != without_occurrence.oracle_status()
        ),
        "externally_compatible_statuses": sorted(
            status.value for status in PhenomenalStatus
        ),
    }


def biface_consistency(record: BifaceRecord) -> dict:
    checks = {
        "same_event": record.external.event_id == record.internal.event_id,
        "same_identity": record.external.subject_id == record.internal.subject_id,
        "same_reason": record.external.reason == record.internal.attended_reason,
        "same_owned_action": record.external.action == record.internal.owned_action,
    }
    return {"checks": checks, "consistent": all(checks.values())}


def sample_biface_records() -> dict:
    external = ExternalFace(
        event_id=17,
        subject_id=101,
        reason=-1,
        action=0,
        report="Je suspends et conserve ce conflit.",
    )
    coherent = BifaceRecord(
        external=external,
        internal=InternalFace(
            event_id=17,
            subject_id=101,
            attended_reason=-1,
            owned_action=0,
            occurrence=PhenomenalStatus.OCCURRENCE,
        ),
    )
    tampered = BifaceRecord(
        external=external,
        internal=InternalFace(
            event_id=17,
            subject_id=101,
            attended_reason=1,
            owned_action=0,
            occurrence=PhenomenalStatus.OCCURRENCE,
        ),
    )
    forged_but_consistent = BifaceRecord(
        external=external,
        internal=InternalFace(
            event_id=17,
            subject_id=101,
            attended_reason=-1,
            owned_action=0,
            occurrence=PhenomenalStatus.ABSENT,
        ),
    )
    return {
        "coherent": biface_consistency(coherent),
        "tampered": biface_consistency(tampered),
        "forged_but_consistent": biface_consistency(forged_but_consistent),
        "forged_oracle_status": forged_but_consistent.internal.occurrence.value,
    }


def cogito_type_audit() -> dict:
    'Separate a public proposition from a stipulated lived occurrence.\n\n`occurrence_entails_occurrence_for_subject` is an analytic/indexical rule of\nthe semantic model.  It is deliberately not exposed as a detector.'

    utterance = 'I saw this decision now.'
    public_worlds = {
        PhenomenalStatus.OCCURRENCE: utterance,
        PhenomenalStatus.ABSENT: utterance,
    }
    return {
        "same_utterance_is_compatible_with_both_statuses": (
            len(set(public_worlds.values())) == 1
        ),
        "utterance_entails_occurrence": False,
        "occurrence_entails_occurrence_for_subject": True,
        "occurrence_transfers_to_third_person_proof": False,
        "occurrence_explains_causal_mechanism": False,
    }


def run_audit() -> dict:
    twins = external_equivalence_audit()
    biface = sample_biface_records()
    cogito = cogito_type_audit()
    expected_histories = sum(len(REASONS) ** n for n in range(1, 8))

    criteria = {
        "all_3279_reason_histories_are_exhausted": (
            twins["tested_history_count"] == expected_histories == 3279
        ),
        "behavioral_twins_are_externally_indistinguishable": twins[
            "public_traces_identical"
        ],
        "self_report_does_not_distinguish_occurrence": twins[
            "self_reports_identical"
        ] and cogito["same_utterance_is_compatible_with_both_statuses"],
        "semantic_internal_statuses_really_differ": twins["oracle_statuses_differ"],
        "coherent_biface_record_passes": biface["coherent"]["consistent"],
        "tampered_face_binding_fails": not biface["tampered"]["consistent"],
        "formal_consistency_is_not_sufficient_for_phenomenology": (
            biface["forged_but_consistent"]["consistent"]
            and biface["forged_oracle_status"] == PhenomenalStatus.ABSENT.value
        ),
        "cogito_is_indexical_not_a_public_mechanism_proof": (
            cogito["occurrence_entails_occurrence_for_subject"]
            and not cogito["utterance_entails_occurrence"]
            and not cogito["occurrence_transfers_to_third_person_proof"]
            and not cogito["occurrence_explains_causal_mechanism"]
        ),
    }
    passed = all(criteria.values())

    return {
        "scope": {
            "physical_V94_imported": False,
            "consciousness_simulated_or_detected": False,
            "phenomenal_status_is_a_semantic_countermodel_label": True,
            "functional_authorship_inherited_from_A2": True,
            "C2_or_C3_modified": False,
        },
        "definitions": {
            "functional_author": (
                "A2: action, veto, reasons, causal memory, motive revision, "
                "identity continuity, nonaccidentality, reason-timed revision"
            ),
            "biface_author": (
                "functional author + first-person occurrence + event-level "
                "binding of identity, reason, action and memory"
            ),
            "strong_author": (
                "biface author + at least one meta-revision not fully closed "
                "by prior state, law, seed, or independent noise"
            ),
        },
        "external_equivalence": twins,
        "biface_binding": biface,
        "cogito": cogito,
        "criteria": criteria,
        "pass": passed,
        "verdict": (
            "BIFACE_AUTHORSHIP_ARCHITECTURE_REQUIRED__"
            "PHENOMENAL_BRIDGE_INDEXICAL_NOT_EXTERNALLY_CLOSED"
            if passed
            else "A3_BIFACE_AUTHORSHIP_AUDIT_FAIL"
        ),
        "scientific_status": {
            "functional_author": "CLOSED_IN_A2_SCOPE",
            "biface_architecture": "DERIVED_AS_NECESSARY_FOR_CROSS_FACE_ATTRIBUTION",
            "phenomenal_occurrence_for_self": "INDEXICALLY_CERTAIN_WHEN_GIVEN",
            "phenomenal_occurrence_for_others": "EXTERNALLY_UNDERDETERMINED",
            "strong_nonautomatable_author": "NOT_DERIVED",
        },
        "next": (
            "A4: audit whether meta-revision origin/nonclosure can be stated "
            "without circularly defining the strong author"
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
