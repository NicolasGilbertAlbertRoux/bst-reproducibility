#!/usr/bin/env python3

import unittest

import bst_cog_a6_empirical_protocol as a6


class EmpiricalProtocolTests(unittest.TestCase):
    def test_randomization_is_balanced(self):
        audit = a6.dry_run(120)
        self.assertTrue(audit["balanced"])
        self.assertEqual(set(audit["condition_counts"].values()), {30})

    def test_randomization_codes_are_unique(self):
        audit = a6.dry_run(120)
        self.assertTrue(audit["unique_participant_codes"])
        self.assertTrue(audit["unique_allocation_codes"])

    def test_public_export_hides_allocation_code(self):
        self.assertTrue(a6.dry_run()["public_export_excludes_allocation_code"])

    def test_confirmatory_schema_has_no_direct_identifiers(self):
        schema = a6.synthetic_schema()
        self.assertEqual(schema["direct_identifiers_present"], [])
        self.assertFalse(schema["free_text_in_confirmatory_core"])

    def test_biface_fields_are_disjoint(self):
        self.assertFalse(set(a6.PUBLIC_FIELDS).intersection(a6.FIRST_PERSON_FIELDS))
        self.assertTrue(
            a6.synthetic_schema()["first_person_precedes_feedback_contract"]
        )

    def test_all_four_adversaries_are_frozen(self):
        self.assertEqual(len(a6.ADVERSARIES), 4)
        self.assertIn("behavioral_report_emulator", a6.ADVERSARIES)

    def test_ethics_and_method_prerequisites_block_execution(self):
        report = a6.run_audit()
        self.assertTrue({"E1", "E2", "E3", "M1", "M2", "R1"}.issubset(
            report["execution_blockers"]
        ))
        self.assertEqual(report["scientific_status"]["human_execution"], "NOT_AUTHORIZED")

    def test_synthetic_n_is_not_claimed_as_powered_sample(self):
        report = a6.run_audit()
        self.assertTrue(report["dry_run"]["this_count_is_not_a_power_calculation"])
        self.assertTrue(report["analysis"]["sample_size"].startswith("NOT_DERIVED"))

    def test_local_rejection_keeps_strong_author_open(self):
        ceilings = {row.outcome: row.maximum_claim for row in a6.CLAIM_CEILINGS}
        self.assertIn(
            "strong author remains open",
            ceilings["all_frozen_adversaries_rejected"],
        )

    def test_full_protocol_contract_passes_without_human_data(self):
        report = a6.run_audit()
        self.assertTrue(report["pass"])
        self.assertTrue(all(report["criteria"].values()))
        self.assertFalse(report["scope"]["human_or_animal_data_collected"])
        self.assertFalse(report["scope"]["physical_V94_imported"])


if __name__ == "__main__":
    unittest.main()
