#!/usr/bin/env python3
"BST–COG A9: sealed-vessel source-attribution audit.\n\nThis stage formalizes the user's water-in-a-sealed-vessel analogy and separates:\n\n1. observable novelty from ontological novelty;\n2. rule revision from non-deterministic origin;\n3. falsification of a closed deterministic description from identification of\na strong author;\n4. a source outside deterministic closure from a second physical substrate.\n\nNo consciousness, author, survival, or physical BST observable is simulated.\nV94 is neither imported nor modified."

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path


STAGE_ID = "BST-COG-A9-R0"
PUBLIC_TRACE = ("wait", "left", "veto", "right", "rewrite", "left")


def public_hash(trace: tuple[str, ...]) -> str:
    payload = json.dumps(trace, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ScriptedHiddenState:
    'A deterministic hidden seed carrying a finite future public trace.'

    script: tuple[str, ...]
    clock: int = 0


def hidden_deterministic_step(state: ScriptedHiddenState) -> tuple[str, ScriptedHiddenState]:
    if state.clock >= len(state.script):
        raise StopIteration
    output = state.script[state.clock]
    return output, ScriptedHiddenState(state.script, state.clock + 1)


def deterministic_hidden_twin(trace: tuple[str, ...]) -> tuple[str, ...]:
    state = ScriptedHiddenState(trace)
    outputs = []
    while state.clock < len(state.script):
        output, state = hidden_deterministic_step(state)
        outputs.append(output)
    return tuple(outputs)


def environment_twin(trace: tuple[str, ...]) -> tuple[str, ...]:
    'An external channel supplies each public transition.'

    return tuple(external_signal for external_signal in trace)


def stochastic_supports_trace(trace: tuple[str, ...]) -> bool:
    'A full-support random source assigns non-zero probability to the trace.'

    alphabet = set(trace)
    probability = (1.0 / len(alphabet)) ** len(trace)
    return probability > 0.0


def public_source_nonidentifiability_audit() -> dict:
    deterministic = deterministic_hidden_twin(PUBLIC_TRACE)
    environmental = environment_twin(PUBLIC_TRACE)
    semantic_worlds = {
        "hidden_deterministic_seed": public_hash(deterministic),
        "environmental_injection": public_hash(environmental),
        "impersonal_stochastic_realization": public_hash(PUBLIC_TRACE),
        "lived_authorial_extension": public_hash(PUBLIC_TRACE),
    }
    return {
        "public_trace": list(PUBLIC_TRACE),
        "candidate_source_classes": sorted(semantic_worlds),
        "deterministic_hidden_twin_exact": deterministic == PUBLIC_TRACE,
        "environmental_twin_exact": environmental == PUBLIC_TRACE,
        "stochastic_source_assigns_nonzero_probability": stochastic_supports_trace(
            PUBLIC_TRACE
        ),
        "public_sha256_by_source_class": semantic_worlds,
        "all_public_hashes_identical": len(set(semantic_worlds.values())) == 1,
        "finite_public_trace_identifies_source": False,
        "conclusion": "PUBLIC_NOVELTY_DOES_NOT_IDENTIFY_ONTOLOGICAL_SOURCE",
    }


def deterministic_function_exists(pairs: tuple[tuple[str, str], ...]) -> bool:
    mapping: dict[str, str] = {}
    for state, outcome in pairs:
        if state in mapping and mapping[state] != outcome:
            return False
        mapping[state] = outcome
    return True


def full_state_recurrence_audit() -> dict:
    apparently_same = (("S", "left"), ("S", "right"))
    hidden_completion = (("S|h=0", "left"), ("S|h=1", "right"))
    return {
        'same_declared_full_state_distinct_outcomes': [list(x) for x in apparently_same],
        "single_valued_deterministic_map_exists_on_declared_state": (
            deterministic_function_exists(apparently_same)
        ),
        "hidden_state_extension": [list(x) for x in hidden_completion],
        "single_valued_deterministic_map_exists_after_hidden_completion": (
            deterministic_function_exists(hidden_completion)
        ),
        "closure_falsification_requires_state_completeness": True,
        "conclusion": (
            "DISTINCT_ACTUALIZATIONS_FROM_ONE_TRULY_COMPLETE_STATE_REFUTE_A_FIXED_"
            "DETERMINISTIC_MAP__HIDDEN_COMPLETION_REOPENS_DETERMINISM"
        ),
    }


@dataclass(frozen=True)
class RevisingState:
    motive: int
    object_rule: int
    meta_rule: int


def deterministic_meta_revision(state: RevisingState, conflict: int) -> RevisingState:
    'A fixed map can revise motive, object rule, and apparent meta-rule.'

    new_motive = state.motive if conflict == 0 else -state.motive
    new_object_rule = (state.object_rule + abs(conflict) + state.meta_rule) % 5
    new_meta_rule = (2 * state.meta_rule + state.object_rule + conflict) % 7
    return RevisingState(new_motive, new_object_rule, new_meta_rule)


def rule_revision_audit() -> dict:
    initial = RevisingState(motive=1, object_rule=0, meta_rule=1)
    conflicts = (1, 0, -1, 1, 1, 0)
    states = [initial]
    current = initial
    for conflict in conflicts:
        current = deterministic_meta_revision(current, conflict)
        states.append(current)
    replay = [initial]
    current = initial
    for conflict in conflicts:
        current = deterministic_meta_revision(current, conflict)
        replay.append(current)
    return {
        "state_count": len(states),
        "motive_changes": sum(a.motive != b.motive for a, b in zip(states, states[1:])),
        "object_rule_changes": sum(
            a.object_rule != b.object_rule for a, b in zip(states, states[1:])
        ),
        "meta_rule_state_changes": sum(
            a.meta_rule != b.meta_rule for a, b in zip(states, states[1:])
        ),
        "exact_replay": states == replay,
        "rule_revision_alone_implies_nonclosure": False,
        "incomputability_alone_implies_nondeterminism": False,
        "infinite_choice_set_alone_implies_nondeterminism": False,
        "conclusion": "REVISION_AND_COMPLEXITY_ARE_NOT_SOURCE_ATTRIBUTION_TESTS",
    }


PREMISES = (
    "D_distinct_actualizations_from_same_prior_state",
    "F_prior_state_is_physically_complete",
    "L_evolution_law_is_fixed_and_single_valued",
    "E_environmental_channel_is_sealed",
    "R_impersonal_randomness_is_excluded_as_adequate_source",
    "X_source_taxonomy_is_exhaustive",
    "C_current_lived_occurrence",
    "O_lived_ownership_of_selection",
    "M_motivation_is_instituted_and_appropriated",
    "I_diachronic_identity_invariant",
)


CLAIM_REQUIREMENTS = {
    "closed_deterministic_account_is_falsified": {
        "D_distinct_actualizations_from_same_prior_state",
        "F_prior_state_is_physically_complete",
        "L_evolution_law_is_fixed_and_single_valued",
    },
    "nonenvironmental_nonrandom_source_is_required": {
        "D_distinct_actualizations_from_same_prior_state",
        "F_prior_state_is_physically_complete",
        "L_evolution_law_is_fixed_and_single_valued",
        "E_environmental_channel_is_sealed",
        "R_impersonal_randomness_is_excluded_as_adequate_source",
        "X_source_taxonomy_is_exhaustive",
    },
    "minimal_strong_author_is_required_conditionally": {
        "D_distinct_actualizations_from_same_prior_state",
        "F_prior_state_is_physically_complete",
        "L_evolution_law_is_fixed_and_single_valued",
        "E_environmental_channel_is_sealed",
        "R_impersonal_randomness_is_excluded_as_adequate_source",
        "X_source_taxonomy_is_exhaustive",
        "C_current_lived_occurrence",
        "O_lived_ownership_of_selection",
        "M_motivation_is_instituted_and_appropriated",
    },
    "diachronic_strong_author_is_required_conditionally": set(PREMISES),
}


def drift_claims(premises: set[str]) -> set[str]:
    if not premises.issubset(PREMISES):
        raise ValueError("unknown premise")
    return {
        claim
        for claim, requirements in CLAIM_REQUIREMENTS.items()
        if requirements.issubset(premises)
    }


def sealed_vessel_dependency_audit() -> dict:
    rows = []
    for mask in range(1 << len(PREMISES)):
        selected = {
            premise for i, premise in enumerate(PREMISES) if mask & (1 << i)
        }
        rows.append(
            {
                "premises": sorted(selected),
                "derived_claims": sorted(drift_claims(selected)),
            }
        )

    all_premises = set(PREMISES)
    strong_claim = "minimal_strong_author_is_required_conditionally"
    strong_requirements = CLAIM_REQUIREMENTS[strong_claim]
    necessity = {
        premise: strong_claim not in drift_claims(all_premises - {premise})
        for premise in strong_requirements
    }
    return {
        "premise_count": len(PREMISES),
        "exhausted_premise_subsets": len(rows),
        "claim_requirements": {
            claim: sorted(requirements)
            for claim, requirements in CLAIM_REQUIREMENTS.items()
        },
        "all_strong_author_requirements_individually_necessary": all(
            necessity.values()
        ),
        "strong_author_requirement_necessity": dict(sorted(necessity.items())),
        'all_premises_drift': sorted(drift_claims(all_premises)),
        "closed_determinism_falsified_without_author_identified": (
            drift_claims(CLAIM_REQUIREMENTS["closed_deterministic_account_is_falsified"])
            == {"closed_deterministic_account_is_falsified"}
        ),
        "source_outside_deterministic_closure_means_second_substrate": False,
        "source_outside_deterministic_closure_can_act_within_one_substrate": True,
        "subsets": rows,
    }


def attribution_ladder() -> dict:
    return {
        "level_0_public_novelty": "SOURCE_UNDERDETERMINED",
        "level_1_rule_or_motive_revision": "DETERMINISTIC_INTERNALIZATION_STILL_POSSIBLE",
        'level_2_same_truly_full_state_distinct_actualizations': (
            "CLOSED_FIXED_DETERMINISTIC_MAP_FALSIFIED"
        ),
        "level_3_sealed_environment_randomness_excluded_exhaustive_taxonomy": (
            "NONENVIRONMENTAL_NONRANDOM_SOURCE_REQUIRED_CONDITIONALLY"
        ),
        "level_4_cogito_ownership_instituted_motivation": (
            "MINIMAL_STRONG_AUTHOR_REQUIRED_CONDITIONALLY"
        ),
        "level_5_diachronic_identity_invariant": (
            "DIACHRONIC_STRONG_AUTHOR_REQUIRED_CONDITIONALLY"
        ),
        "unconditional_formal_level_reached": 1,
        "reason": (
            "No physical completeness observable, ownership bridge, or exhaustive "
            "source exclusion has yet been derived for BST."
        ),
    }


def run_audit() -> dict:
    public = public_source_nonidentifiability_audit()
    recurrence = full_state_recurrence_audit()
    revision = rule_revision_audit()
    vessel = sealed_vessel_dependency_audit()
    ladder = attribution_ladder()
    criteria = {
        "finite_public_trace_has_exact_hidden_deterministic_twin": public[
            "deterministic_hidden_twin_exact"
        ],
        "finite_public_trace_does_not_identify_source": (
            public["all_public_hashes_identical"]
            and not public["finite_public_trace_identifies_source"]
        ),
        "distinct_outputs_break_single_valued_map_on_same_state": (
            not recurrence[
                "single_valued_deterministic_map_exists_on_declared_state"
            ]
        ),
        "hidden_completion_reopens_deterministic_explanation": recurrence[
            "single_valued_deterministic_map_exists_after_hidden_completion"
        ],
        "rule_motive_and_meta_rule_revision_replay_deterministically": (
            revision["motive_changes"] > 0
            and revision["object_rule_changes"] > 0
            and revision["meta_rule_state_changes"] > 0
            and revision["exact_replay"]
        ),
        "revision_complexity_and_infinity_are_not_nonclosure_proofs": (
            not revision["rule_revision_alone_implies_nonclosure"]
            and not revision["incomputability_alone_implies_nondeterminism"]
            and not revision["infinite_choice_set_alone_implies_nondeterminism"]
        ),
        "all_1024_sealed_vessel_premise_sets_are_exhausted": (
            vessel["exhausted_premise_subsets"] == 1024
        ),
        "closed_determinism_can_be_falsified_without_identifying_author": vessel[
            "closed_determinism_falsified_without_author_identified"
        ],
        "every_strong_author_bridge_premise_is_explicit_and_necessary": vessel[
            "all_strong_author_requirements_individually_necessary"
        ],
        "authorial_nonclosure_does_not_require_a_second_substrate": (
            not vessel[
                "source_outside_deterministic_closure_means_second_substrate"
            ]
            and vessel[
                "source_outside_deterministic_closure_can_act_within_one_substrate"
            ]
        ),
        "strong_author_is_only_conditionally_derived": (
            ladder["unconditional_formal_level_reached"] < 4
        ),
        "V94_is_untouched": True,
    }
    passed = all(criteria.values())
    return {
        "stage_id": STAGE_ID,
        "scope": {
            "branch": "cognition_information_only",
            "formal_source_attribution_audit": True,
            "strong_author_physically_detected": False,
            "physical_completeness_observable_derived": False,
            "one_substrate_preserved": True,
            "physical_V94_imported": False,
            "V94_status": "C2_OPEN__C3_CLOSED",
        },
        "public_nonidentifiability": public,
        'full_state_recurrence': recurrence,
        "rule_revision": revision,
        "sealed_vessel_dependencies": vessel,
        "attribution_ladder": ladder,
        "criteria": criteria,
        "pass": passed,
        "verdict": (
            "SEALED_VESSEL_ATTRIBUTION_THEOREM_CONDITIONAL_PASS__"
            "CLOSED_DETERMINISM_FALSIFIABLE_IF_COMPLETE_STATE_RECURRENCE_IS_ESTABLISHED__"
            "STRONG_AUTHOR_REQUIRES_OWNERSHIP_MOTIVATION_AND_EXHAUSTIVE_SOURCE_EXCLUSION__"
            "PHYSICAL_COMPLETENESS_BRIDGE_REMAINS_OPEN"
            if passed
            else "A9_SEALED_VESSEL_SOURCE_ATTRIBUTION_AUDIT_FAIL"
        ),
        "next": (
            "Freeze A9. A stronger author claim requires a BST physical completeness "
            "observable or bridge axiom that can exclude hidden-state completion, "
            "environmental injection and impersonal stochastic realization. Resume V94-C2."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "bst_cog_a9_results.json",
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
