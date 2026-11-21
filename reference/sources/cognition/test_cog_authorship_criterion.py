#!/usr/bin/env python3

import random
import unittest

import bst_cog_a2_authorship_criterion as a2


class AuthorshipCriterionTests(unittest.TestCase):
    def test_adaptive_author_uses_conflict_memory_causally(self):
        agent = a2.AdaptiveFunctionalAuthor()
        no_history = a2.AgentState(conflict_count=0, step=5)
        with_history = a2.AgentState(conflict_count=1, step=5)
        a = agent.decide(no_history, -1, random.Random(1))
        b = agent.decide(with_history, -1, random.Random(1))
        self.assertFalse(a.revised_motive)
        self.assertTrue(b.revised_motive)

    def test_adaptive_author_can_veto(self):
        decision = a2.AdaptiveFunctionalAuthor().decide(
            a2.AgentState(), 0, random.Random(1)
        )
        self.assertTrue(decision.vetoed)
        self.assertFalse(decision.engaged)
        self.assertEqual(decision.action, 0)

    def test_revision_tracks_reasons_not_clock(self):
        agent = a2.AdaptiveFunctionalAuthor()
        early = a2.simulate(agent, [-1, -1, 1, 1, 1, 1])
        late = a2.simulate(agent, [1, 1, -1, -1, 1, 1])
        self.assertEqual(a2.first_revision_step(early), 1)
        self.assertEqual(a2.first_revision_step(late), 3)

    def test_scripted_revision_tracks_clock_not_reasons(self):
        agent = a2.ScriptedRevision(revision_step=3)
        early = a2.simulate(agent, [-1, -1, 1, 1, 1, 1])
        late = a2.simulate(agent, [1, 1, -1, -1, 1, 1])
        self.assertEqual(a2.first_revision_step(early), 3)
        self.assertEqual(a2.first_revision_step(late), 3)

    def test_random_drift_is_seed_sensitive(self):
        agent = a2.RandomDrift()
        reasons = [1, 0, -1, -1, 1]
        self.assertNotEqual(
            a2.simulate(agent, reasons, seed=501),
            a2.simulate(agent, reasons, seed=502),
        )

    def test_only_adaptive_model_passes_functional_gate(self):
        agents = [
            a2.FixedController(),
            a2.ScriptedRevision(),
            a2.RandomDrift(),
            a2.AdaptiveFunctionalAuthor(),
        ]
        results = {agent.name: a2.probe_agent(agent) for agent in agents}
        self.assertFalse(results["fixed_controller"]["functional_author_gate"])
        self.assertFalse(results["scripted_revision"]["functional_author_gate"])
        self.assertFalse(results["random_drift"]["functional_author_gate"])
        self.assertTrue(
            results["adaptive_functional_author"]["functional_author_gate"]
        )

    def test_finite_table_imitates_observed_trace_only(self):
        result = a2.audit_finite_imitation()
        self.assertTrue(result["deterministic_table_exactly_replays_observed_trace"])
        self.assertTrue(result["unseen_counterfactual_is_not_explained"])

    def test_full_audit_passes(self):
        report = a2.run_audit()
        self.assertTrue(report["pass"])
        self.assertTrue(all(report["criteria"].values()))


if __name__ == "__main__":
    unittest.main()
