#!/usr/bin/env python3
'BST–COG A5: theorem-stack and biface protocol audit.\n\nThis module checks the logic of falsifiability versus actual falsification,\nfreezes the A0–A4 dependency stack, and emits a preregisterable research\nprotocol. It does not detect consciousness or import any V94 result.'

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LogicalWorld:
    author_exists: bool
    absence_falsifiable: bool
    absence_actually_refuted_by_valid_witness: bool

    @property
    def admissible(self) -> bool:
        # A valid, actual counterexample to universal absence entails existence.
        return not self.absence_actually_refuted_by_valid_witness or self.author_exists


def logical_implication_audit() -> dict:
    worlds = [
        LogicalWorld(*bits)
        for bits in itertools.product((False, True), repeat=3)
    ]
    admissible = [world for world in worlds if world.admissible]
    falsifiable_without_existence = [
        world
        for world in admissible
        if world.absence_falsifiable and not world.author_exists
    ]
    actual_refutation_without_existence = [
        world
        for world in admissible
        if world.absence_actually_refuted_by_valid_witness
        and not world.author_exists
    ]
    return {
        "world_count": len(worlds),
        "admissible_world_count": len(admissible),
        "falsifiability_without_existence_countermodels": len(
            falsifiable_without_existence
        ),
        "actual_valid_refutation_without_existence_countermodels": len(
            actual_refutation_without_existence
        ),
        "falsifiability_implies_existence": False,
        "actual_valid_refutation_of_absence_implies_existence": True,
        "current_A0_A4_has_valid_strong_author_witness": False,
    }


THEOREM_STACK = (
    {
        "stage": "A0",
        "result": "covariant deterministic selector obstruction",
        "status": "DERIVED_CONDITIONALLY",
        "depends_on": (),
        "maximum_claim": "actualization/asymmetry is required when no branch is fixed",
    },
    {
        "stage": "A1",
        "result": "functional arbiter and persistence separation",
        "status": "DERIVED_AND_SIMULATED",
        "depends_on": ("A0",),
        "maximum_claim": "correction, veto and persistence do not imply freedom",
    },
    {
        "stage": "A2",
        "result": "functional authorship criterion",
        "status": "CLOSED_IN_MODEL_SCOPE",
        "depends_on": ("A0", "A1"),
        "maximum_claim": "functional authorship can be deterministic",
    },
    {
        "stage": "A3",
        "result": "biface authorship and external indistinguishability",
        "status": "DERIVED_EPISTEMIC_WALL",
        "depends_on": ("A2",),
        "maximum_claim": "first-person occurrence is not inferred from public report",
    },
    {
        "stage": "A4",
        "result": "meta-revision closure trilemma",
        "status": "DERIVED_TRANSITIONAL_IMPOSSIBILITY_WALL",
        "depends_on": ("A0", "A1", "A2", "A3"),
        "maximum_claim": "strong author origin is not identifiable from transitions alone",
    },
)


def validate_theorem_stack() -> dict:
    seen = set()
    order_valid = True
    unique = True
    stages = []
    for row in THEOREM_STACK:
        stage = row["stage"]
        unique &= stage not in seen
        order_valid &= set(row["depends_on"]).issubset(seen)
        seen.add(stage)
        stages.append(stage)
    return {
        "stage_order": stages,
        "unique_stages": unique,
        "dependencies_precede_consumers": order_valid,
        "strong_author_claim_present_as_closed_result": any(
            "strong author exists" in row["maximum_claim"].lower()
            for row in THEOREM_STACK
        ),
    }


@dataclass(frozen=True)
class ProtocolGate:
    gate_id: str
    face: str
    requirement: str
    failure_means: str
    pass_ceiling: str


PROTOCOL_GATES = (
    ProtocolGate(
        "P0",
        "preregistration",
        "freeze population, intervention family, closure class C and thresholds",
        "study is not confirmatory",
        "declared model comparison only",
    ),
    ProtocolGate(
        "P1",
        "external",
        "test action, veto, reason sensitivity and unseen holdouts",
        "functional-author candidate fails",
        "functional capacity",
    ),
    ProtocolGate(
        "P2",
        "diachronic",
        "test causal memory, motive revision and identity continuity",
        "appropriation candidate fails",
        "diachronic functional authorship",
    ),
    ProtocolGate(
        "P3",
        "first_person",
        "time-lock occurrence, salience, ownership and motive reports before outcome",
        "biface evidence is incomplete",
        "indexical testimony, not public proof of phenomenology",
    ),
    ProtocolGate(
        "P4",
        "binding",
        "blindly bind event, identity, reason, action and later memory across faces",
        "biface coherence fails",
        "cross-face coherence",
    ),
    ProtocolGate(
        "P5",
        "adversarial",
        "compare extended_deterministic_state, stochastic_process, "
        "exogenous_oracle and behavioral_report_emulator models",
        "strong-origin attribution is blocked",
        "relative rejection inside frozen class C",
    ),
    ProtocolGate(
        "P6",
        "claim",
        "apply the predeclared claim ceiling",
        "result is exploratory or invalid",
        "candidate evidence; never absolute nonclosure",
    ),
)


ADVERSARIES = (
    "extended_deterministic_state",
    "stochastic_process",
    "exogenous_oracle",
    "behavioral_report_emulator",
)


def validate_protocol() -> dict:
    ids = [gate.gate_id for gate in PROTOCOL_GATES]
    faces = {gate.face for gate in PROTOCOL_GATES}
    joined = " ".join(gate.requirement for gate in PROTOCOL_GATES)
    return {
        "gate_count": len(PROTOCOL_GATES),
        "unique_gate_ids": len(ids) == len(set(ids)),
        "has_preregistration": "P0" in ids,
        "has_external_face": "external" in faces,
        "has_first_person_face": "first_person" in faces,
        "has_cross_face_binding": "binding" in faces,
        "has_adversarial_gate": "adversarial" in faces,
        "all_four_A4_adversaries_named": all(token in joined for token in ADVERSARIES),
        "absolute_nonclosure_claim_forbidden": all(
            "never absolute nonclosure" in gate.pass_ceiling
            for gate in PROTOCOL_GATES
            if gate.gate_id == "P6"
        ),
    }


def classify_outcome(
    *,
    functional_pass: bool,
    biface_coherence_pass: bool,
    frozen_class_rejected: bool,
    valid_strong_witness: bool,
) -> dict:
    if valid_strong_witness:
        verdict = "ABSENCE_ACTUALLY_REFUTED__STRONG_AUTHOR_EXISTS"
        ceiling = "existence follows from an independently valid witness"
    elif functional_pass and biface_coherence_pass and frozen_class_rejected:
        verdict = "LOCAL_CLOSURE_CLASS_REJECTED__STRONG_AUTHOR_STILL_OPEN"
        ceiling = "candidate evidence relative to frozen class C"
    elif functional_pass and biface_coherence_pass:
        verdict = "BIFACE_FUNCTIONAL_CANDIDATE__NO_STRONG_ORIGIN_CLAIM"
        ceiling = "functional plus biface coherence"
    else:
        verdict = "AUTHOR_CANDIDATE_GATE_NOT_PASSED"
        ceiling = "no positive authorship claim"
    return {"verdict": verdict, "claim_ceiling": ceiling}


def outcome_lattice_audit() -> dict:
    outcomes = {
        "negative": classify_outcome(
            functional_pass=False,
            biface_coherence_pass=False,
            frozen_class_rejected=False,
            valid_strong_witness=False,
        ),
        "functional_biface": classify_outcome(
            functional_pass=True,
            biface_coherence_pass=True,
            frozen_class_rejected=False,
            valid_strong_witness=False,
        ),
        "local_refutation": classify_outcome(
            functional_pass=True,
            biface_coherence_pass=True,
            frozen_class_rejected=True,
            valid_strong_witness=False,
        ),
        "valid_strong_witness": classify_outcome(
            functional_pass=True,
            biface_coherence_pass=True,
            frozen_class_rejected=True,
            valid_strong_witness=True,
        ),
    }
    return {
        "outcomes": outcomes,
        "local_refutation_does_not_claim_strong_author": (
            "STRONG_AUTHOR_STILL_OPEN"
            in outcomes["local_refutation"]["verdict"]
        ),
        "valid_witness_refutes_absence": (
            outcomes["valid_strong_witness"]["verdict"]
            == "ABSENCE_ACTUALLY_REFUTED__STRONG_AUTHOR_EXISTS"
        ),
    }


def run_audit() -> dict:
    logic = logical_implication_audit()
    stack = validate_theorem_stack()
    protocol = validate_protocol()
    lattice = outcome_lattice_audit()
    criteria = {
        "falsifiability_does_not_entail_existence": (
            logic["falsifiability_without_existence_countermodels"] > 0
            and not logic["falsifiability_implies_existence"]
        ),
        "actual_valid_refutation_would_entail_existence": (
            logic["actual_valid_refutation_without_existence_countermodels"] == 0
            and logic["actual_valid_refutation_of_absence_implies_existence"]
        ),
        "A0_A4_do_not_contain_current_valid_strong_witness": not logic[
            "current_A0_A4_has_valid_strong_author_witness"
        ],
        "A0_A4_dependencies_are_valid": (
            stack["unique_stages"] and stack["dependencies_precede_consumers"]
        ),
        "A0_A4_do_not_close_strong_author_existence": not stack[
            "strong_author_claim_present_as_closed_result"
        ],
        "protocol_is_biface_preregistered_and_adversarial": all(
            protocol[key]
            for key in (
                "unique_gate_ids",
                "has_preregistration",
                "has_external_face",
                "has_first_person_face",
                "has_cross_face_binding",
                "has_adversarial_gate",
                "all_four_A4_adversaries_named",
            )
        ),
        "protocol_forbids_absolute_nonclosure_upgrade": protocol[
            "absolute_nonclosure_claim_forbidden"
        ],
        "outcome_lattice_separates_local_and_actual_refutation": (
            lattice["local_refutation_does_not_claim_strong_author"]
            and lattice["valid_witness_refutes_absence"]
        ),
    }
    passed = all(criteria.values())
    return {
        "scope": {
            "physical_V94_imported": False,
            "C2_or_C3_modified": False,
            "human_or_animal_experiment_performed": False,
            "strong_author_detected": False,
            "protocol_is_preregisterable_not_yet_executed": True,
        },
        "logic": logic,
        "theorem_stack": list(THEOREM_STACK),
        "stack_validation": stack,
        "protocol": [asdict(gate) for gate in PROTOCOL_GATES],
        "protocol_adversaries": list(ADVERSARIES),
        "protocol_validation": protocol,
        "outcome_lattice": lattice,
        "criteria": criteria,
        "pass": passed,
        "verdict": (
            "A0_A4_QUALITATIVE_STACK_CONSOLIDATED__"
            "ABSENCE_ONLY_LOCALLY_FALSIFIABLE__STRONG_AUTHOR_EXISTENCE_OPEN"
            if passed
            else "A5_THEOREM_STACK_PROTOCOL_AUDIT_FAIL"
        ),
        "scientific_status": {
            "absolute_absence_logically_falsifiable": True,
            "absolute_absence_currently_falsified": False,
            "restricted_closure_null_operationally_falsifiable": True,
            "strong_author_exists": "OPEN_NOT_DERIVED_NOT_REFUTED",
            "A0_A4_qualitative_stack": "CONSOLIDATED",
            "biface_protocol": "PREREGISTERABLE_NOT_EXECUTED",
        },
        "next": (
            "A6 or paper integration: operationalize instruments, ethics, "
            "sampling, blinded analysis and preregistered closure class C"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--protocol-output", default=None)
    args = parser.parse_args()
    report = run_audit()
    payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
    print(payload)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    if args.protocol_output:
        protocol_payload = {
            "status": "PREREGISTERABLE_NOT_EXECUTED",
            "gates": report["protocol"],
            "adversaries": report["protocol_adversaries"],
            "outcome_lattice": report["outcome_lattice"],
        }
        with open(args.protocol_output, "w", encoding="utf-8") as handle:
            json.dump(
                protocol_payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            handle.write("\n")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
