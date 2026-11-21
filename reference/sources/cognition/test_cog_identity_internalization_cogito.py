#!/usr/bin/env python3

import unittest

import bst_cog_a8_identity_internalization_cogito as a8


class IdentityInternalizationCogitoTests(unittest.TestCase):
    def test_reflexive_agent_passes_functional_gate(self):
        result = a8.reflexive_autonomy_audit()
        self.assertEqual(result["modes_observed"], ["defer", "delegate", "engage", "veto"])
        self.assertGreater(result["unprompted_internal_clock_actions"], 0)
        self.assertGreater(result["motive_revisions"], 0)
        self.assertGreater(result["autonomy_rule_revisions"], 0)

    def test_reflexive_agent_is_deterministic(self):
        result = a8.reflexive_autonomy_audit()
        self.assertTrue(result["fully_deterministic"])
        self.assertTrue(result["exact_replay"])
        self.assertFalse(result["lived_ownership_derived"])

    def test_all_bounded_histories_compile(self):
        result = a8.bounded_reflexive_internalization()
        self.assertEqual(result["tested_histories"], 5460)
        self.assertTrue(result["intentional_and_fixed_kernel_traces_identical"])
        self.assertGreater(result["autonomy_rule_revisions_observed"], 0)

    def test_first_person_public_trace_has_deterministic_twin(self):
        result = a8.public_twin_audit()
        self.assertTrue(result["deterministic_transcript_twin_exactly_replays"])
        self.assertTrue(result["reports_include_first_person_language"])
        self.assertFalse(result["first_person_language_entails_lived_occurrence"])

    def test_phenomenal_status_is_not_derived_from_public_structure(self):
        result = a8.cogito_model_independence_audit()
        self.assertTrue(result["public_structures_identical"])
        self.assertFalse(result['public_transition_formalism_derivatives_phenomenal_label'])
        self.assertTrue(result["bridge_axiom_or_new_physical_observable_required"])

    def test_cogito_alone_has_a_strict_ceiling(self):
        result = a8.axiom_dependency_audit()
        self.assertEqual(result['cogito_alone_derivatives'], ["current_occurrence_exists"])
        self.assertTrue(result['cogito_alone_does_not_drift_diachronic_or_strong_author'])

    def test_cogito_plus_ownership_conditionally_derivatives_minimal_author(self):
        result = a8.axiom_dependency_audit()
        self.assertTrue(result['cogito_plus_ownership_derivatives_minimal_current_author'])

    def test_nonclosure_is_required_for_strong_origin(self):
        self.assertTrue(a8.axiom_dependency_audit()["nonclosure_is_required_for_strong_origin"])

    def test_one_substrate_has_multiple_observation_maps(self):
        result = a8.latent_persistence_audit()
        self.assertEqual(result["substrate_count"], 1)
        self.assertEqual(len(result["observation_maps"]), 2)

    def test_visible_disorganization_does_not_imply_latent_erasure(self):
        result = a8.latent_persistence_audit()
        self.assertTrue(result["latent_identity_and_memory_survive_visible_disorganization"])
        self.assertTrue(result["preserved_world_can_reexpress_memory"])
        self.assertFalse(result["observable_disorganization_entails_latent_erasure"])

    def test_visible_disorganization_alone_does_not_prove_survival(self):
        result = a8.latent_persistence_audit()
        self.assertTrue(result["preserved_and_erased_worlds_are_coarsely_identical"])
        self.assertTrue(result["erased_world_cannot_reexpress_memory"])
        self.assertFalse(result["observable_disorganization_entails_latent_survival"])

    def test_biological_death_is_not_claimed(self):
        self.assertFalse(a8.latent_persistence_audit()["biological_death_modeled"])

    def test_full_audit_passes_without_touching_V94(self):
        result = a8.run_audit()
        self.assertTrue(result["pass"])
        self.assertTrue(all(result["criteria"].values()))
        self.assertFalse(result["scope"]["physical_V94_imported"])
        self.assertEqual(result["scope"]["V94_status"], "C2_OPEN__C3_CLOSED")


if __name__ == "__main__":
    unittest.main()
