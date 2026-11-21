#!/usr/bin/env python3

import unittest

import bst_cog_a4_meta_revision_origin as a4


class MetaRevisionOriginTests(unittest.TestCase):
    def test_bounded_meta_revision_internalizes_exactly(self):
        audit = a4.bounded_internalization_audit()
        self.assertEqual(audit["tested_histories"], 363)
        self.assertTrue(
            audit["apparent_and_extended_fixed_kernel_are_identical"]
        )
        self.assertGreater(audit["meta_rule_revisions_observed"], 0)

    def test_finite_meta_trace_compiles_exactly(self):
        audit = a4.finite_trace_compilation_audit()
        self.assertTrue(audit["finite_trace_exactly_compiled"])
        self.assertTrue(audit["unseen_case_not_explained"])

    def test_stochastic_provenance_is_hidden(self):
        audit = a4.provenance_underdetermination_audit()
        self.assertTrue(
            audit[
                "stochastic_public_traces_equal_under_hidden_provenance_labels"
            ]
        )
        self.assertFalse(audit["public_trace_identifies_provenance"])

    def test_exogenous_oracle_does_not_identify_author(self):
        audit = a4.provenance_underdetermination_audit()
        self.assertTrue(
            audit["exogenous_oracle_and_alleged_author_public_traces_equal"]
        )

    def test_diagonal_witness_defeats_prediction_not_automation(self):
        audit = a4.diagonal_counterexample((0, 1, 1, 0, 1))
        self.assertEqual(audit["diagonal_mismatch_count"], 5)
        self.assertTrue(
            audit["universal_prediction_defeated_in_finite_witness"]
        )
        self.assertFalse(audit["authorship_follows_from_prediction_failure"])

    def test_motivated_meta_revision_can_be_deterministic(self):
        audit = a4.candidate_gate_audit()
        self.assertTrue(audit["functional_candidate_gate_pass"])
        self.assertTrue(audit["passing_model_is_deterministic"])
        self.assertFalse(audit["strong_authorship_inferred"])

    def test_origin_trilemma_is_exactly_three_transition_classes(self):
        audit = a4.provenance_underdetermination_audit()
        self.assertEqual(
            audit["origin_classes"],
            [
                "deterministic_closure",
                "stochastic_closure",
                "exogenous_input",
            ],
        )

    def test_nonclosure_levels_remain_distinct(self):
        report = a4.run_audit()
        self.assertEqual(len(report["nonclosure_levels"]), 4)
        self.assertNotEqual(
            report["nonclosure_levels"]["computational_unpredictability"],
            report["nonclosure_levels"]["agential_authorship"],
        )

    def test_full_audit_passes_without_deriving_strong_author(self):
        report = a4.run_audit()
        self.assertTrue(report["pass"])
        self.assertTrue(all(report["criteria"].values()))
        self.assertEqual(
            report["scientific_status"]["strong_author_nonclosure"],
            "NOT_DERIVED",
        )


if __name__ == "__main__":
    unittest.main()
