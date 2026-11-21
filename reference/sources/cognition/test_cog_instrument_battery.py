#!/usr/bin/env python3

import unittest

import bst_cog_a7_instrument_battery as a7


class InstrumentBatteryTests(unittest.TestCase):
    def test_scenario_bank_has_four_scenarios_per_block(self):
        rows = a7.scenario_bank()
        self.assertEqual(len(rows), 20)
        self.assertEqual(
            {block: sum(row["block_id"] == block for row in rows) for block in "B1 B2 B3 B4 B5".split()},
            {"B1": 4, "B2": 4, "B3": 4, "B4": 4, "B5": 4},
        )

    def test_scenarios_use_only_neutral_whitelisted_topics(self):
        self.assertTrue(all(row["topic"] in a7.NEUTRAL_TOPICS for row in a7.scenario_bank()))
        self.assertTrue(all(not row["sensitive_content"] for row in a7.scenario_bank()))

    def test_choice_to_choose_modes_are_exact(self):
        for row in a7.scenario_bank():
            if row["block_id"] == "B2":
                self.assertEqual(tuple(row["response_options"]), a7.CHOICE_TO_CHOOSE_MODES)

    def test_first_person_instrument_is_frozen_and_pre_feedback(self):
        contract = a7.instrument_contract()
        self.assertEqual(tuple(item.field for item in a7.ITEMS), a7.FIRST_PERSON_FIELDS)
        self.assertEqual((contract["scale"]["minimum"], contract["scale"]["maximum"]), (1, 7))
        self.assertTrue(contract["administration"]["all_first_person_items_locked_before_feedback"])
        self.assertFalse(contract["administration"]["confirmatory_core_free_text"])

    def test_comprehension_gate_is_full(self):
        gate = a7.instrument_contract()["comprehension_gate"]
        self.assertEqual(len(gate["items"]), 3)
        self.assertIn("3_of_3", gate["pass_rule"])
        self.assertIn("exclude", gate["failure_rule"])

    def test_deterministic_world_is_recovered(self):
        result = a7.run_synthetic_pipeline(2048)
        self.assertTrue(result["recovery"]["extended_deterministic_state"])

    def test_stochastic_world_is_recovered(self):
        result = a7.run_synthetic_pipeline(2048)
        self.assertTrue(result["recovery"]["stochastic_process"])

    def test_exogenous_world_is_recovered(self):
        result = a7.run_synthetic_pipeline(2048)
        self.assertTrue(result["recovery"]["exogenous_oracle"])

    def test_report_emulator_is_recovered(self):
        result = a7.run_synthetic_pipeline(2048)
        self.assertTrue(result["recovery"]["behavioral_report_emulator"])

    def test_semantic_twins_are_publicly_identical_and_score_tied(self):
        result = a7.run_synthetic_pipeline(1024)
        twin = result["semantic_twin_audit"]
        self.assertTrue(twin["public_records_equal"])
        self.assertTrue(twin["scores_equal"])
        self.assertFalse(twin["hidden_labels_available_to_pipeline"])

    def test_no_human_or_strong_author_claim_is_made(self):
        contract = a7.instrument_contract()
        pipeline = a7.run_synthetic_pipeline(1024)
        self.assertEqual(contract["human_execution"], "NOT_AUTHORIZED")
        self.assertIn("STRONG_AUTHOR_NOT_IDENTIFIED", pipeline["claim_ceiling"])

    def test_physics_branch_remains_untouched(self):
        self.assertEqual(
            a7.instrument_contract()["physical_branch"],
            "NO_V94_IMPORT__C2_OPEN__C3_CLOSED",
        )

    def test_full_audit_passes(self):
        result = a7.run_audit()
        self.assertTrue(result["pass"])
        self.assertTrue(all(result["criteria"].values()))
        self.assertIn("STOP_AT_HUMAN_EMPIRICAL_WALL", result["verdict"])


if __name__ == "__main__":
    unittest.main()
