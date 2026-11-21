#!/usr/bin/env python3

import unittest

import bst_cog_a9_sealed_vessel_source_attribution as a9


class SealedVesselSourceAttributionTests(unittest.TestCase):
    def test_finite_public_trace_has_deterministic_hidden_twin(self):
        result = a9.public_source_nonidentifiability_audit()
        self.assertTrue(result["deterministic_hidden_twin_exact"])
        self.assertTrue(result["all_public_hashes_identical"])
        self.assertFalse(result["finite_public_trace_identifies_source"])

    def test_environment_and_stochastic_classes_are_not_excluded_by_trace(self):
        result = a9.public_source_nonidentifiability_audit()
        self.assertTrue(result["environmental_twin_exact"])
        self.assertTrue(result["stochastic_source_assigns_nonzero_probability"])

    def test_same_full_state_cannot_have_two_outputs_under_a_function(self):
        result = a9.full_state_recurrence_audit()
        self.assertFalse(result["single_valued_deterministic_map_exists_on_declared_state"])

    def test_hidden_completion_reopens_determinism(self):
        result = a9.full_state_recurrence_audit()
        self.assertTrue(result["single_valued_deterministic_map_exists_after_hidden_completion"])
        self.assertTrue(result["closure_falsification_requires_state_completeness"])

    def test_rule_and_motive_revision_can_be_deterministic(self):
        result = a9.rule_revision_audit()
        self.assertGreater(result["motive_changes"], 0)
        self.assertGreater(result["object_rule_changes"], 0)
        self.assertGreater(result["meta_rule_state_changes"], 0)
        self.assertTrue(result["exact_replay"])

    def test_incomputability_and_infinity_do_not_alone_prove_nonclosure(self):
        result = a9.rule_revision_audit()
        self.assertFalse(result["incomputability_alone_implies_nondeterminism"])
        self.assertFalse(result["infinite_choice_set_alone_implies_nondeterminism"])

    def test_all_premise_subsets_are_exhausted(self):
        result = a9.sealed_vessel_dependency_audit()
        self.assertEqual(result["premise_count"], 10)
        self.assertEqual(result["exhausted_premise_subsets"], 1024)

    def test_determinism_falsification_does_not_identify_author(self):
        result = a9.sealed_vessel_dependency_audit()
        self.assertTrue(result["closed_determinism_falsified_without_author_identified"])

    def test_all_strong_author_premises_are_individually_necessary(self):
        result = a9.sealed_vessel_dependency_audit()
        self.assertTrue(result["all_strong_author_requirements_individually_necessary"])

    def test_one_substrate_is_preserved(self):
        result = a9.sealed_vessel_dependency_audit()
        self.assertFalse(result["source_outside_deterministic_closure_means_second_substrate"])
        self.assertTrue(result["source_outside_deterministic_closure_can_act_within_one_substrate"])

    def test_unconditional_ladder_stops_before_strong_author(self):
        result = a9.attribution_ladder()
        self.assertEqual(result["unconditional_formal_level_reached"], 1)

    def test_full_audit_passes_without_touching_v94(self):
        result = a9.run_audit()
        self.assertTrue(result["pass"])
        self.assertTrue(all(result["criteria"].values()))
        self.assertFalse(result["scope"]["physical_V94_imported"])
        self.assertEqual(result["scope"]["V94_status"], "C2_OPEN__C3_CLOSED")


if __name__ == "__main__":
    unittest.main()
