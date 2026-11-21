#!/usr/bin/env python3

import itertools
import unittest

import bst_hidden_asymmetry_audit as audit


class SelectorTheoremTests(unittest.TestCase):
    def test_group_orders(self):
        for n in range(2, 8):
            self.assertEqual(len(audit.cyclic_group(n)), n)
            # On two vertices the abstract D2 action is not faithful: only
            # two distinct permutations act on the branch set.
            self.assertEqual(len(audit.dihedral_group(n)), 2 if n == 2 else 2 * n)
            expected_symmetric = 1
            for k in range(2, n + 1):
                expected_symmetric *= k
            self.assertEqual(len(audit.symmetric_group(n)), expected_symmetric)

    def test_uniform_transitive_states_have_no_selector(self):
        for n in range(2, 8):
            for constructor in (
                audit.cyclic_group,
                audit.dihedral_group,
                audit.symmetric_group,
            ):
                group = constructor(n)
                self.assertEqual(
                    audit.admissible_selector_branches(group, (0,) * n), ()
                )

    def test_stabilizer_criterion_exhaustively(self):
        for n in range(2, 8):
            for constructor in (
                audit.cyclic_group,
                audit.dihedral_group,
                audit.symmetric_group,
            ):
                group = constructor(n)
                for colors in itertools.product(range(2), repeat=n):
                    expected = audit.fixed_branches(audit.stabilizer(group, colors))
                    observed = audit.admissible_selector_branches(group, colors)
                    self.assertEqual(observed, expected)

    def test_unique_marker_exposes_hidden_asymmetry(self):
        for n in range(3, 8):
            colors = (1,) + (0,) * (n - 1)
            group = audit.symmetric_group(n)
            self.assertEqual(audit.admissible_selector_branches(group, colors), (0,))

    def test_hidden_asymmetry_need_not_make_selector_unique(self):
        # A marked vertex of a square leaves both it and its opposite fixed
        # under the residual reflection symmetry.
        group = audit.dihedral_group(4)
        colors = (1, 0, 0, 0)
        self.assertEqual(audit.admissible_selector_branches(group, colors), (0, 2))

    def test_index_zero_tiebreak_is_not_covariant(self):
        for n in range(2, 8):
            self.assertFalse(
                audit.canonical_index_selector_is_covariant(audit.cyclic_group(n))
            )

    def test_declared_audit_passes(self):
        report = audit.run_audit(max_n=7, alphabet_size=2)
        self.assertTrue(all(report["criteria"].values()))


if __name__ == "__main__":
    unittest.main()
