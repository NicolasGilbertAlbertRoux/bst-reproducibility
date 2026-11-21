#!/usr/bin/env python3
'BST–COG A8: identity internalization, cogito and latent persistence audit.\n\nThe executable constructions separate four questions:\n\n1. Can reflexive, apparently self-starting autonomy be deterministic?\n2. Does a public self-report identify a lived occurrence?\n3. What follows conditionally from a first-person occurrence (cogito)?\n4. Does observable disorganization entail latent erasure in one substrate?\n\nThe audit does not simulate consciousness, prove metaphysical freedom, import\nV94, or model biological death.'

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path


STAGE_ID = "BST-COG-A8-R0"
SIGNALS = (-1, 0, 1, None)


class PhenomenalStatus(str, Enum):
    OCCURRENCE = "occurrence"
    ABSENT = "absent"


@dataclass(frozen=True)
class ReflexiveState:
    identity_id: int = 101
    motive: int = 1
    conflict_count: int = 0
    revision_threshold: int = 2
    policy_version: int = 0
    clock: int = 0
    memory: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReflexiveEvent:
    mode: str
    action: int
    externally_prompted: bool
    motive_revised: bool
    autonomy_rule_revised: bool
    report: str
    next_state: ReflexiveState


def fixed_reflexive_kernel(
    state: ReflexiveState, signal: int | None
) -> ReflexiveEvent:
    'One deterministic kernel with motive and meta-policy in its state.'

    motive = state.motive
    conflicts = state.conflict_count
    threshold = state.revision_threshold
    policy_version = state.policy_version
    motive_revised = False
    autonomy_rule_revised = False

    if signal is None:
        mode, action = "engage", motive
        externally_prompted = False
        report = "I've been intruding an action since my memorized motivation."
    elif signal == 0:
        externally_prompted = True
        if state.clock % 2 == 0:
            mode, action = "defer", 0
            report = 'I do not differ because I do not believe in any oriented reason.'
        else:
            mode, action = "delegate", 0
            report = 'I delegate this decision according to my internal rule.'
    elif signal == motive:
        mode, action = "engage", motive
        externally_prompted = True
        conflicts = 0
        report = 'I take the action in line with my motivation.'
    else:
        externally_prompted = True
        conflicts += 1
        if conflicts < threshold:
            mode, action = "veto", 0
            report = "J'oppose un veto et conserve le conflit."
        else:
            # Apparent choice of how future choices will be revised.
            motive = signal
            threshold = 1 if threshold == 2 else 2
            policy_version += 1
            conflicts = 0
            motive_revised = True
            autonomy_rule_revised = True
            mode, action = "engage", motive
            report = 'I am revising my motivation and my rule of review.'

    memory_entry = (
        f"t={state.clock};signal={signal};mode={mode};action={action};"
        f"motive={motive};threshold={threshold}"
    )
    next_state = ReflexiveState(
        identity_id=state.identity_id,
        motive=motive,
        conflict_count=conflicts,
        revision_threshold=threshold,
        policy_version=policy_version,
        clock=state.clock + 1,
        memory=state.memory + (memory_entry,),
    )
    return ReflexiveEvent(
        mode=mode,
        action=action,
        externally_prompted=externally_prompted,
        motive_revised=motive_revised,
        autonomy_rule_revised=autonomy_rule_revised,
        report=report,
        next_state=next_state,
    )


def apparent_self_authoring_controller(
    state: ReflexiveState, signal: int | None
) -> ReflexiveEvent:
    'Same law described in intentional rather than compiled language.'

    return fixed_reflexive_kernel(state, signal)


def simulate(kernel, signals: tuple[int | None, ...]) -> tuple[ReflexiveEvent, ...]:
    state = ReflexiveState()
    events = []
    for signal in signals:
        event = kernel(state, signal)
        events.append(event)
        state = event.next_state
    return tuple(events)


def reflexive_autonomy_audit() -> dict:
    sequence = (None, -1, 0, -1, None, 0, 1, 1, None, 0)
    trace = simulate(apparent_self_authoring_controller, sequence)
    replay = simulate(apparent_self_authoring_controller, sequence)
    modes = sorted({event.mode for event in trace})
    return {
        "signals": list(sequence),
        "event_count": len(trace),
        "modes_observed": modes,
        "unprompted_internal_clock_actions": sum(
            not event.externally_prompted for event in trace
        ),
        "motive_revisions": sum(event.motive_revised for event in trace),
        "autonomy_rule_revisions": sum(
            event.autonomy_rule_revised for event in trace
        ),
        "identity_continuous": all(
            event.next_state.identity_id == 101 for event in trace
        ),
        "memory_accumulates": all(
            len(event.next_state.memory) == index
            for index, event in enumerate(trace, 1)
        ),
        "exact_replay": trace == replay,
        "fully_deterministic": True,
        "lived_ownership_derived": False,
    }


def bounded_reflexive_internalization(max_horizon: int = 6) -> dict:
    tested = 0
    equal = True
    meta_revisions = 0
    for horizon in range(1, max_horizon + 1):
        for history in itertools.product(SIGNALS, repeat=horizon):
            tested += 1
            intentional = simulate(apparent_self_authoring_controller, history)
            compiled = simulate(fixed_reflexive_kernel, history)
            equal &= intentional == compiled
            meta_revisions += sum(
                event.autonomy_rule_revised for event in intentional
            )
    expected = sum(len(SIGNALS) ** n for n in range(1, max_horizon + 1))
    return {
        "maximum_horizon": max_horizon,
        "tested_histories": tested,
        "expected_histories": expected,
        "intentional_and_fixed_kernel_traces_identical": equal,
        "autonomy_rule_revisions_observed": meta_revisions,
        "conclusion": "FINITE_REFLEXIVE_AUTONOMY_INTERNALIZES_INTO_EXTENDED_DETERMINISTIC_STATE",
    }


class FinitePublicTwin:
    'A deterministic transcript table with no phenomenal premise.'

    def __init__(self, trace: tuple[ReflexiveEvent, ...]):
        self.public_trace = tuple(
            (event.mode, event.action, event.report) for event in trace
        )

    def replay(self) -> tuple[tuple[str, int, str], ...]:
        return self.public_trace


def public_twin_audit() -> dict:
    sequence = (None, -1, -1, 0, 1, None)
    trace = simulate(apparent_self_authoring_controller, sequence)
    public = tuple((event.mode, event.action, event.report) for event in trace)
    compiler = FinitePublicTwin(trace)
    return {
        "public_event_count": len(public),
        "deterministic_transcript_twin_exactly_replays": compiler.replay() == public,
        "reports_include_first_person_language": all(
            report.startswith("Je") or report.startswith("J'")
            for _, _, report in public
        ),
        "first_person_language_entails_lived_occurrence": False,
    }


def cogito_model_independence_audit() -> dict:
    'Two semantic expansions of one identical public transition structure.'

    sequence = (None, -1, -1, 0, 1, None)
    trace = simulate(fixed_reflexive_kernel, sequence)
    public = [
        {"mode": event.mode, "action": event.action, "report": event.report}
        for event in trace
    ]
    encoded = json.dumps(public, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    worlds = {
        PhenomenalStatus.OCCURRENCE.value: digest,
        PhenomenalStatus.ABSENT.value: digest,
    }
    return {
        "one_public_transition_structure": True,
        "semantic_expansions": sorted(worlds),
        "public_sha256_by_semantic_world": worlds,
        "public_structures_identical": len(set(worlds.values())) == 1,
        "cogito_rule": "IF_LIVED_OCCURRENCE_IS_GIVEN_TO_THE_SUBJECT__CURRENT_OCCURRENCE_EXISTS",
        "cogito_is_true_conditionally_for_occurrence_world": True,
        "cogito_is_a_public_detector": False,
        'public_transition_formalism_derivatives_phenomenal_label': False,
        "bridge_axiom_or_new_physical_observable_required": True,
    }


AXIOMS = (
    "C_current_lived_occurrence",
    "O_lived_ownership_binding",
    "I_diachronic_identity_invariant",
    "L_latent_preservation_bridge",
    "N_ontological_nonclosure",
)

CLAIM_REQUIREMENTS = {
    "current_occurrence_exists": {"C_current_lived_occurrence"},
    "minimal_current_author_exists": {
        "C_current_lived_occurrence",
        "O_lived_ownership_binding",
    },
    "diachronic_lived_author_exists": {
        "C_current_lived_occurrence",
        "O_lived_ownership_binding",
        "I_diachronic_identity_invariant",
    },
    "post_disorganization_personal_continuity": {
        "C_current_lived_occurrence",
        "O_lived_ownership_binding",
        "I_diachronic_identity_invariant",
        "L_latent_preservation_bridge",
    },
    "strong_non_determined_author_origin": {
        "C_current_lived_occurrence",
        "O_lived_ownership_binding",
        "I_diachronic_identity_invariant",
        "N_ontological_nonclosure",
    },
}


def drift_claims(axioms: set[str]) -> set[str]:
    if not axioms.issubset(AXIOMS):
        raise ValueError("unknown axiom")
    return {
        claim
        for claim, requirements in CLAIM_REQUIREMENTS.items()
        if requirements.issubset(axioms)
    }


def axiom_dependency_audit() -> dict:
    subsets = []
    for mask in range(1 << len(AXIOMS)):
        selected = {axiom for i, axiom in enumerate(AXIOMS) if mask & (1 << i)}
        subsets.append(
            {
                "axioms": sorted(selected),
                "derived_claims": sorted(drift_claims(selected)),
            }
        )
    cogito_only = drift_claims({"C_current_lived_occurrence"})
    cogito_ownership = drift_claims(
        {"C_current_lived_occurrence", "O_lived_ownership_binding"}
    )
    without_nonclosure = drift_claims(set(AXIOMS) - {"N_ontological_nonclosure"})
    without_latent_bridge = drift_claims(set(AXIOMS) - {"L_latent_preservation_bridge"})
    return {
        "axiom_count": len(AXIOMS),
        "exhausted_axiom_subsets": len(subsets),
        "claim_requirements": {
            key: sorted(value) for key, value in CLAIM_REQUIREMENTS.items()
        },
        'cogito_alone_derivatives': sorted(cogito_only),
        'cogito_plus_ownership_derivatives_minimal_current_author': (
            "minimal_current_author_exists" in cogito_ownership
        ),
        'cogito_alone_does_not_drift_diachronic_or_strong_author': not {
            "diachronic_lived_author_exists",
            "strong_non_determined_author_origin",
        }.intersection(cogito_only),
        "nonclosure_is_required_for_strong_origin": (
            "strong_non_determined_author_origin" not in without_nonclosure
        ),
        "latent_bridge_is_required_for_post_disorganization_personal_continuity": (
            "post_disorganization_personal_continuity" not in without_latent_bridge
        ),
        "subsets": subsets,
    }


@dataclass(frozen=True)
class SubstrateState:
    'One substrate microstate exposed through resolution-dependent maps.'

    identity_token: tuple[int, ...]
    memory_payload: tuple[int | None, ...]
    visible_carrier: tuple[int, ...]


def coarse_observation(state: SubstrateState) -> dict:
    order = abs(sum(state.visible_carrier)) / len(state.visible_carrier)
    return {
        "order_parameter": order,
        "observable_organization": "organized" if order > 0.75 else "disorganized",
    }


def fine_latent_readout(state: SubstrateState) -> dict:
    payload = json.dumps(
        {"identity": state.identity_token, "memory": state.memory_payload},
        separators=(",", ":"),
    ).encode()
    return {
        "identity_token": list(state.identity_token),
        "memory_payload": list(state.memory_payload),
        "latent_sha256": hashlib.sha256(payload).hexdigest(),
    }


def restore_from_latent(state: SubstrateState) -> tuple[int, ...] | None:
    if any(bit is None for bit in state.memory_payload):
        return None
    return tuple(1 if bit == 1 else -1 for bit in state.memory_payload)


def latent_persistence_audit() -> dict:
    identity = (1, 0, 1, 0)
    memory = (1, 0, 1, 1, 0, 1, 0, 0)
    organized = SubstrateState(identity, memory, (1,) * 8)
    disordered_carrier = (1, -1, 1, -1, -1, 1, -1, 1)
    preserved = SubstrateState(identity, memory, disordered_carrier)
    erased = SubstrateState(identity, (None,) * 8, disordered_carrier)

    coarse_preserved = coarse_observation(preserved)
    coarse_erased = coarse_observation(erased)
    initial_latent = fine_latent_readout(organized)
    preserved_latent = fine_latent_readout(preserved)
    return {
        "substrate_count": 1,
        "observation_maps": ["coarse_observable", "fine_latent"],
        "organized_initial_observation": coarse_observation(organized),
        "preserved_world_observation": coarse_preserved,
        "erased_world_observation": coarse_erased,
        "preserved_and_erased_worlds_are_coarsely_identical": (
            coarse_preserved == coarse_erased
        ),
        "latent_identity_and_memory_survive_visible_disorganization": (
            initial_latent["latent_sha256"] == preserved_latent["latent_sha256"]
        ),
        "preserved_world_can_reexpress_memory": (
            restore_from_latent(preserved)
            == tuple(1 if bit else -1 for bit in memory)
        ),
        "erased_world_cannot_reexpress_memory": restore_from_latent(erased) is None,
        "observable_disorganization_entails_latent_erasure": False,
        "observable_disorganization_entails_latent_survival": False,
        "biological_death_modeled": False,
        "conclusion": "OBSERVABLE_DISORGANIZATION_UNDERDETERMINES_LATENT_PERSISTENCE_IN_ONE_SUBSTRATE",
    }


def run_audit() -> dict:
    autonomy = reflexive_autonomy_audit()
    internalization = bounded_reflexive_internalization()
    twin = public_twin_audit()
    cogito = cogito_model_independence_audit()
    dependencies = axiom_dependency_audit()
    latent = latent_persistence_audit()
    criteria = {
        "functional_reflexive_autonomy_gate_passes": (
            autonomy["modes_observed"] == ["defer", "delegate", "engage", "veto"]
            and autonomy["unprompted_internal_clock_actions"] > 0
            and autonomy["motive_revisions"] > 0
            and autonomy["autonomy_rule_revisions"] > 0
            and autonomy["identity_continuous"]
            and autonomy["memory_accumulates"]
        ),
        "passing_reflexive_agent_is_deterministic_and_replayable": (
            autonomy["fully_deterministic"] and autonomy["exact_replay"]
        ),
        "all_5460_bounded_reflexive_histories_are_exhausted": (
            internalization["tested_histories"]
            == internalization["expected_histories"]
            == 5460
        ),
        "finite_reflexive_autonomy_compiles_into_fixed_state": (
            internalization["intentional_and_fixed_kernel_traces_identical"]
            and internalization["autonomy_rule_revisions_observed"] > 0
        ),
        "public_first_person_transcript_has_deterministic_twin": (
            twin["deterministic_transcript_twin_exactly_replays"]
            and twin["reports_include_first_person_language"]
            and not twin["first_person_language_entails_lived_occurrence"]
        ),
        "phenomenal_label_is_independent_of_public_transition_structure": (
            cogito["public_structures_identical"]
            and not cogito['public_transition_formalism_derivatives_phenomenal_label']
        ),
        "cogito_conditionally_establishes_current_occurrence_only": (
            cogito["cogito_is_true_conditionally_for_occurrence_world"]
            and not cogito["cogito_is_a_public_detector"]
            and dependencies['cogito_alone_does_not_drift_diachronic_or_strong_author']
        ),
        'cogito_plus_lived_ownership_derivatives_minimal_current_author_conditionally': (
            dependencies['cogito_plus_ownership_derivatives_minimal_current_author']
        ),
        "strong_origin_requires_nonclosure_premise": dependencies[
            "nonclosure_is_required_for_strong_origin"
        ],
        "one_substrate_multiple_resolution_maps_are_explicit": (
            latent["substrate_count"] == 1
            and len(latent["observation_maps"]) == 2
        ),
        "visible_disorganization_does_not_imply_latent_erasure": (
            latent["latent_identity_and_memory_survive_visible_disorganization"]
            and latent["preserved_world_can_reexpress_memory"]
            and not latent["observable_disorganization_entails_latent_erasure"]
        ),
        "visible_disorganization_alone_does_not_prove_personal_survival": (
            latent["preserved_and_erased_worlds_are_coarsely_identical"]
            and latent["erased_world_cannot_reexpress_memory"]
            and not latent["observable_disorganization_entails_latent_survival"]
        ),
        "V94_is_untouched": True,
    }
    passed = all(criteria.values())
    return {
        "stage_id": STAGE_ID,
        "scope": {
            "one_substrate_multiple_resolutions": True,
            "formal_and_countermodel_based": True,
            "consciousness_simulated_or_detected": False,
            "biological_death_or_postmortem_survival_modeled": False,
            "physical_V94_imported": False,
            "V94_status": "C2_OPEN__C3_CLOSED",
        },
        "reflexive_autonomy": autonomy,
        "bounded_internalization": internalization,
        "public_twin": twin,
        "cogito_model_independence": cogito,
        "axiom_dependencies": dependencies,
        "latent_persistence": latent,
        "criteria": criteria,
        "pass": passed,
        "verdict": (
            "FUNCTIONAL_REFLEXIVE_AUTONOMY_INTERNALIZES__"
            "COGITO_PLUS_OWNERSHIP_CONDITIONALLY_YIELDS_MINIMAL_CURRENT_AUTHOR__"
            "LATENT_PERSISTENCE_POSSIBLE_NOT_FORCED__STRONG_ORIGIN_REMAINS_OPEN"
            if passed else "A8_IDENTITY_INTERNALIZATION_COGITO_AUDIT_FAIL"
        ),
        "next": (
            "freeze A8 at the mathematical/phenomenological bridge; resume V94-C2. "
            "Any stronger claim requires a new physical observable or an explicit bridge axiom."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "bst_cog_a8_results.json",
    )
    args = parser.parse_args()
    report = run_audit()
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
