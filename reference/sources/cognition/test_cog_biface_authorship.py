#!/usr/bin/env python3

import unittest

import bst_cog_a3_biface_authorship as a3


class BifaceAuthorshipTests(unittest.TestCase):
    def test_exhaustive_external_equivalence(self):
        audit = a3.external_equivalence_audit()
        self.assertEqual(audit["tested_history_count"], 3279)
        self.assertTrue(audit["public_traces_identical"])

    def test_self_reports_do_not_reveal_status(self):
        audit = a3.external_equivalence_audit(max_horizon=5)
        self.assertTrue(audit["self_reports_identical"])
        self.assertTrue(audit["oracle_statuses_differ"])

    def test_cogito_occurrence_is_not_an_utterance(self):
        audit = a3.cogito_type_audit()
        self.assertTrue(audit["same_utterance_is_compatible_with_both_statuses"])
        self.assertFalse(audit["utterance_entails_occurrence"])
        self.assertTrue(audit["occurrence_entails_occurrence_for_subject"])

    def test_cogito_does_not_transfer_publicly(self):
        audit = a3.cogito_type_audit()
        self.assertFalse(audit["occurrence_transfers_to_third_person_proof"])
        self.assertFalse(audit["occurrence_explains_causal_mechanism"])

    def test_coherent_biface_record_passes(self):
        records = a3.sample_biface_records()
        self.assertTrue(records["coherent"]["consistent"])

    def test_tampered_binding_fails(self):
        records = a3.sample_biface_records()
        self.assertFalse(records["tampered"]["consistent"])
        self.assertFalse(records["tampered"]["checks"]["same_reason"])

    def test_consistent_record_does_not_prove_phenomenology(self):
        records = a3.sample_biface_records()
        self.assertTrue(records["forged_but_consistent"]["consistent"])
        self.assertEqual(records["forged_oracle_status"], "absent")

    def test_full_audit_passes_without_closing_strong_authorship(self):
        report = a3.run_audit()
        self.assertTrue(report["pass"])
        self.assertTrue(all(report["criteria"].values()))
        self.assertEqual(
            report["scientific_status"]["strong_nonautomatable_author"],
            "NOT_DERIVED",
        )


if __name__ == "__main__":
    unittest.main()
