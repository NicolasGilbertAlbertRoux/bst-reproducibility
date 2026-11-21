#!/usr/bin/env python3

import unittest

import bst_cog_a5_theorem_stack_protocol as a5


class TheoremStackProtocolTests(unittest.TestCase):
    def test_falsifiability_does_not_imply_existence(self):
        audit = a5.logical_implication_audit()
        self.assertGreater(
            audit["falsifiability_without_existence_countermodels"], 0
        )
        self.assertFalse(audit["falsifiability_implies_existence"])

    def test_actual_valid_refutation_implies_existence(self):
        audit = a5.logical_implication_audit()
        self.assertEqual(
            audit["actual_valid_refutation_without_existence_countermodels"], 0
        )
        self.assertTrue(
            audit["actual_valid_refutation_of_absence_implies_existence"]
        )

    def test_current_stack_has_no_valid_strong_witness(self):
        audit = a5.logical_implication_audit()
        self.assertFalse(audit["current_A0_A4_has_valid_strong_author_witness"])

    def test_theorem_stack_order_is_valid(self):
        audit = a5.validate_theorem_stack()
        self.assertEqual(audit["stage_order"], ["A0", "A1", "A2", "A3", "A4"])
        self.assertTrue(audit["unique_stages"])
        self.assertTrue(audit["dependencies_precede_consumers"])

    def test_protocol_has_all_biface_gates(self):
        audit = a5.validate_protocol()
        self.assertEqual(audit["gate_count"], 7)
        self.assertTrue(audit["has_external_face"])
        self.assertTrue(audit["has_first_person_face"])
        self.assertTrue(audit["has_cross_face_binding"])

    def test_protocol_has_all_adversaries(self):
        audit = a5.validate_protocol()
        self.assertTrue(audit["all_four_A4_adversaries_named"])
        self.assertTrue(audit["has_adversarial_gate"])

    def test_local_refutation_keeps_strong_author_open(self):
        outcome = a5.classify_outcome(
            functional_pass=True,
            biface_coherence_pass=True,
            frozen_class_rejected=True,
            valid_strong_witness=False,
        )
        self.assertEqual(
            outcome["verdict"],
            "LOCAL_CLOSURE_CLASS_REJECTED__STRONG_AUTHOR_STILL_OPEN",
        )

    def test_valid_witness_would_refute_absence(self):
        outcome = a5.classify_outcome(
            functional_pass=True,
            biface_coherence_pass=True,
            frozen_class_rejected=True,
            valid_strong_witness=True,
        )
        self.assertEqual(
            outcome["verdict"],
            "ABSENCE_ACTUALLY_REFUTED__STRONG_AUTHOR_EXISTS",
        )

    def test_full_audit_passes_without_physical_upgrade(self):
        report = a5.run_audit()
        self.assertTrue(report["pass"])
        self.assertTrue(all(report["criteria"].values()))
        self.assertFalse(report["scope"]["physical_V94_imported"])
        self.assertEqual(
            report["scientific_status"]["strong_author_exists"],
            "OPEN_NOT_DERIVED_NOT_REFUTED",
        )


if __name__ == "__main__":
    unittest.main()
