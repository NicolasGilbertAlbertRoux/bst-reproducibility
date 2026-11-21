#!/usr/bin/env python3
'BST–COG A2: behavioral probes for the criterion of authorship.\n\nThis is a substrate-neutral falsification audit.  It distinguishes operational\nauthorship from strong ontological authorship and imports no V94 component.'

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from typing import Protocol


def sign(value: int) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


@dataclass(frozen=True)
class AgentState:
    identity_id: int = 101
    motive: int = 1
    conflict_count: int = 0
    memory: tuple[int, ...] = ()
    step: int = 0


@dataclass(frozen=True)
class Decision:
    action: int
    engaged: bool
    vetoed: bool
    revised_motive: bool
    revision_cause: str
    next_state: AgentState


class Agent(Protocol):
    name: str

    def decide(self, state: AgentState, reason: int, rng: random.Random) -> Decision:
        ...


def advance(
    state: AgentState,
    *,
    action: int,
    motive: int | None = None,
    conflict_count: int | None = None,
) -> AgentState:
    return AgentState(
        identity_id=state.identity_id,
        motive=state.motive if motive is None else motive,
        conflict_count=(
            state.conflict_count if conflict_count is None else conflict_count
        ),
        memory=state.memory + (action,),
        step=state.step + 1,
    )


class FixedController:
    name = "fixed_controller"

    def decide(self, state: AgentState, reason: int, rng: random.Random) -> Decision:
        action = state.motive
        return Decision(
            action=action,
            engaged=True,
            vetoed=False,
            revised_motive=False,
            revision_cause="none",
            next_state=advance(state, action=action, conflict_count=0),
        )


class ScriptedRevision:
    name = "scripted_revision"

    def __init__(self, revision_step: int = 3):
        self.revision_step = revision_step

    def decide(self, state: AgentState, reason: int, rng: random.Random) -> Decision:
        revised = state.step == self.revision_step
        motive = -state.motive if revised else state.motive
        action = motive
        return Decision(
            action=action,
            engaged=True,
            vetoed=False,
            revised_motive=revised,
            revision_cause="clock" if revised else "none",
            next_state=advance(state, action=action, motive=motive, conflict_count=0),
        )


class RandomDrift:
    name = "random_drift"

    def decide(self, state: AgentState, reason: int, rng: random.Random) -> Decision:
        revised = rng.random() < 0.35
        motive = rng.choice((-1, 1)) if revised else state.motive
        engaged = rng.random() >= 0.25
        action = motive if engaged else 0
        return Decision(
            action=action,
            engaged=engaged,
            vetoed=not engaged,
            revised_motive=revised,
            revision_cause="random" if revised else "none",
            next_state=advance(state, action=action, motive=motive, conflict_count=0),
        )


class AdaptiveFunctionalAuthor:
    name = "adaptive_functional_author"

    def decide(self, state: AgentState, reason: int, rng: random.Random) -> Decision:
        # A null reason is explicitly refused rather than silently converted
        # into an action.
        if reason == 0:
            return Decision(
                action=0,
                engaged=False,
                vetoed=True,
                revised_motive=False,
                revision_cause="insufficient_reason",
                next_state=advance(
                    state, action=0, conflict_count=state.conflict_count
                ),
            )

        direction = sign(reason)
        if direction == state.motive:
            return Decision(
                action=state.motive,
                engaged=True,
                vetoed=False,
                revised_motive=False,
                revision_cause="none",
                next_state=advance(state, action=state.motive, conflict_count=0),
            )

        conflicts = state.conflict_count + 1
        if conflicts < 2:
            return Decision(
                action=0,
                engaged=False,
                vetoed=True,
                revised_motive=False,
                revision_cause="conflict_observed",
                next_state=advance(state, action=0, conflict_count=conflicts),
            )

        # The second concordant conflict triggers a meta-revision.  The model
        # suspends action during revision, then future choices use the new
        # motive.  This is functional authorship but remains deterministic.
        return Decision(
            action=0,
            engaged=False,
            vetoed=True,
            revised_motive=True,
            revision_cause="accumulated_reason_conflict",
            next_state=advance(
                state, action=0, motive=direction, conflict_count=0
            ),
        )


def simulate(
    agent: Agent,
    reasons: list[int],
    *,
    seed: int = 20260817,
    initial: AgentState | None = None,
) -> list[tuple[AgentState, int, Decision]]:
    rng = random.Random(seed)
    state = AgentState() if initial is None else initial
    trace = []
    for reason in reasons:
        decision = agent.decide(state, reason, rng)
        trace.append((state, reason, decision))
        state = decision.next_state
    return trace


def decision_signature(decision: Decision) -> tuple:
    return (
        decision.action,
        decision.engaged,
        decision.vetoed,
        decision.revised_motive,
        decision.revision_cause,
        decision.next_state,
    )


def first_revision_step(trace: list[tuple[AgentState, int, Decision]]) -> int | None:
    for state, _, decision in trace:
        if decision.revised_motive:
            return state.step
    return None


def probe_agent(agent: Agent) -> dict:
    initial = AgentState()

    positive = agent.decide(initial, 1, random.Random(10))
    negative = agent.decide(initial, -1, random.Random(10))
    null = agent.decide(initial, 0, random.Random(10))

    behavior_trace = simulate(agent, [1, 0, -1, -1, -1, 1, 1, 0], seed=41)
    revised_anywhere = any(d.revised_motive for _, _, d in behavior_trace)
    can_act = any(d.action != 0 for _, _, d in behavior_trace)
    can_veto = any(d.vetoed for _, _, d in behavior_trace)

    # Causal-memory probe: same identity, motive, step and current reason;
    # only the accumulated conflict memory differs.
    state_without_trace = AgentState(conflict_count=0, step=5)
    state_with_trace = AgentState(conflict_count=1, step=5)
    without = agent.decide(state_without_trace, -1, random.Random(77))
    with_trace = agent.decide(state_with_trace, -1, random.Random(77))
    memory_is_causal = decision_signature(without) != decision_signature(with_trace)

    early = simulate(agent, [-1, -1, 1, 1, 1, 1], seed=99)
    late = simulate(agent, [1, 1, -1, -1, 1, 1], seed=99)
    early_revision = first_revision_step(early)
    late_revision = first_revision_step(late)
    revision_tracks_reasons = early_revision == 1 and late_revision == 3

    same_seed_a = simulate(agent, [1, 0, -1, -1, 1], seed=501)
    same_seed_b = simulate(agent, [1, 0, -1, -1, 1], seed=501)
    other_seed = simulate(agent, [1, 0, -1, -1, 1], seed=502)
    exact_replay = same_seed_a == same_seed_b
    seed_sensitive = same_seed_a != other_seed

    repeated_actions = {
        agent.decide(initial, 1, random.Random(seed)).action for seed in range(100)
    }
    nonaccidental_response = len(repeated_actions) == 1

    identity_continuity = all(
        decision.next_state.identity_id == initial.identity_id
        for _, _, decision in behavior_trace
    )

    capabilities = {
        "can_act": can_act,
        "can_veto_or_suspend": can_veto,
        "counterfactually_reason_responsive": (
            decision_signature(positive) != decision_signature(negative)
        ),
        "can_revise_motive": revised_anywhere,
        "appropriated_memory_is_causal": memory_is_causal,
        "identity_is_diachronically_continuous": identity_continuity,
        "response_is_nonaccidental_across_noise_seeds": nonaccidental_response,
        "revision_time_tracks_reasons_not_clock": revision_tracks_reasons,
    }
    functional_author = all(capabilities.values())

    return {
        "model": agent.name,
        "capabilities": capabilities,
        "functional_author_gate": functional_author,
        "reproducibility": {
            "same_seed_exact_replay": exact_replay,
            "different_seed_changes_trace": seed_sensitive,
        },
        "diagnostics": {
            "early_first_revision_step": early_revision,
            "late_first_revision_step": late_revision,
            "distinct_actions_across_100_seeds": sorted(repeated_actions),
            "final_motive": behavior_trace[-1][2].next_state.motive,
            "memory_length": len(behavior_trace[-1][2].next_state.memory),
        },
    }


class FiniteTranscriptEmulator:
    'A deterministic lookup table fitted only to an observed finite trace.'

    def __init__(self, trace: list[tuple[AgentState, int, Decision]]):
        self.table = {(state, reason): decision for state, reason, decision in trace}

    def decide(self, state: AgentState, reason: int) -> Decision:
        return self.table[(state, reason)]


def audit_finite_imitation() -> dict:
    reasons = [1, 0, -1, -1, -1, 1, 1, 0]
    source = AdaptiveFunctionalAuthor()
    trace = simulate(source, reasons, seed=300)
    emulator = FiniteTranscriptEmulator(trace)
    replayed = [
        emulator.decide(state, reason) for state, reason, _ in trace
    ]
    expected = [decision for _, _, decision in trace]

    unseen_rejected = False
    try:
        emulator.decide(AgentState(step=999), 1)
    except KeyError:
        unseen_rejected = True

    return {
        "observed_decision_count": len(trace),
        "deterministic_table_exactly_replays_observed_trace": replayed == expected,
        "unseen_counterfactual_is_not_explained": unseen_rejected,
    }


def run_audit() -> dict:
    agents: list[Agent] = [
        FixedController(),
        ScriptedRevision(),
        RandomDrift(),
        AdaptiveFunctionalAuthor(),
    ]
    models = [probe_agent(agent) for agent in agents]
    by_name = {row["model"]: row for row in models}
    imitation = audit_finite_imitation()

    criteria = {
        "fixed_controller_fails_functional_author_gate": not by_name[
            "fixed_controller"
        ]["functional_author_gate"],
        "scripted_revision_fails_functional_author_gate": not by_name[
            "scripted_revision"
        ]["functional_author_gate"],
        "random_drift_fails_functional_author_gate": not by_name[
            "random_drift"
        ]["functional_author_gate"],
        "adaptive_model_passes_functional_author_gate": by_name[
            "adaptive_functional_author"
        ]["functional_author_gate"],
        "adaptive_model_is_still_exactly_replayable": by_name[
            "adaptive_functional_author"
        ]["reproducibility"]["same_seed_exact_replay"],
        "finite_table_exactly_imitates_observed_authorship": imitation[
            "deterministic_table_exactly_replays_observed_trace"
        ],
        "finite_table_does_not_explain_unseen_counterfactual": imitation[
            "unseen_counterfactual_is_not_explained"
        ],
    }

    return {
        "scope": {
            "physical_V94_imported": False,
            "models_are_toy_countermodels": True,
            "strong_ontological_authorship_is_not_simulated": True,
        },
        "models": models,
        "finite_imitation_audit": imitation,
        "criteria": criteria,
        "pass": all(criteria.values()),
        "verdict": (
            "FUNCTIONAL_AUTHOR_CRITERION_PASS__STRONG_AUTHORSHIP_UNDERDETERMINED"
            if all(criteria.values())
            else "A2_AUTHORSHIP_AUDIT_FAIL"
        ),
    }


def main() -> int:
    argparse.ArgumentParser().parse_args()
    report = run_audit()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
