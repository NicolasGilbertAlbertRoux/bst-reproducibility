#!/usr/bin/env python3

import unittest

import bst_cog_a1_arbiter_models as a1


class ArbiterModelTests(unittest.TestCase):
    def test_exact_symmetry_stays_exact_in_open_loop(self):
        trajectory = a1.unstable_spiral(initial=0j)
        self.assertTrue(all(value == 0j for value in trajectory))

    def test_nonzero_asymmetry_is_amplified(self):
        trajectory = a1.unstable_spiral(initial=1e-6 + 0j)
        self.assertGreater(abs(trajectory[-1]), 0.1)

    def test_same_hidden_seed_replays_exactly(self):
        a = a1.lcg_branches(8, 10_000, 12345)
        b = a1.lcg_branches(8, 10_000, 12345)
        c = a1.lcg_branches(8, 10_000, 12346)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_iid_has_chance_level_persistence(self):
        sequence = a1.iid_branches(8, 100_000, 12345)
        self.assertAlmostEqual(a1.persistence_rate(sequence), 1 / 8, delta=0.01)

    def test_stochastic_memory_is_counterexample_to_universal_chaos_claim(self):
        sequence = a1.sticky_random_branches(8, 100_000, 12345, 0.99)
        self.assertGreater(a1.persistence_rate(sequence), 0.98)

    def test_deterministic_corrector_restores_after_shock(self):
        states, _ = a1.corrective_spiral()
        delay = a1.first_return_delay(states, 0j, 80, 1e-3)
        self.assertIsNotNone(delay)
        self.assertLess(abs(states[-1]), 1e-12)

    def test_deterministic_corrector_can_retarget(self):
        states, targets = a1.corrective_spiral(switch_step=160)
        self.assertEqual(targets[-1], 2 + 1j)
        self.assertLess(abs(states[-1] - targets[-1]), 1e-12)

    def test_full_audit_passes(self):
        report = a1.run_audit()
        self.assertTrue(report["pass"])
        self.assertTrue(all(report["criteria"].values()))


if __name__ == "__main__":
    unittest.main()
