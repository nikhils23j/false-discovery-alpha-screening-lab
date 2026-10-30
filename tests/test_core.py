import unittest

import numpy as np

from alpha_screen.core import (
    BLOCK_DAYS, FACTOR_COUNT, TRAIN_DAYS, benjamini_hochberg,
    block_permutation_indices, daily_rank_ic, evaluate, synthetic_market,
)


class ScreeningTests(unittest.TestCase):
    def test_market_is_reproducible_and_temporally_separated(self):
        x1, y1, _ = synthetic_market(42)
        x2, y2, _ = synthetic_market(42)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
        self.assertEqual(x1.shape, (1200, 12, FACTOR_COUNT))
        train_ic = daily_rank_ic(x1[:TRAIN_DAYS], y1[:TRAIN_DAYS]).mean(axis=0)
        altered = y1.copy()
        altered[TRAIN_DAYS:] *= -100
        np.testing.assert_array_equal(train_ic, daily_rank_ic(x1[:TRAIN_DAYS], altered[:TRAIN_DAYS]).mean(axis=0))

    def test_block_shuffle_keeps_members_and_within_block_order(self):
        indices = block_permutation_indices(500, BLOCK_DAYS, np.random.default_rng(7))
        np.testing.assert_array_equal(np.sort(indices), np.arange(500))
        self.assertFalse(np.array_equal(indices, np.arange(500)))
        np.testing.assert_array_equal(np.diff(indices.reshape(-1, BLOCK_DAYS), axis=1), np.ones((25, 19), dtype=int))

    def test_bh_step_up_rule(self):
        np.testing.assert_array_equal(
            benjamini_hochberg(np.array([0.001, 0.02, 0.03, 0.9]), q=0.05),
            np.array([True, True, True, False]),
        )
        with self.assertRaises(ValueError):
            benjamini_hochberg(np.array([1.2]))

    def test_signal_and_null_controls(self):
        signal = evaluate(seed=42, permutations=499)
        self.assertEqual(signal["summary"]["detected_injected_signals"], 5)
        self.assertEqual(signal["summary"]["false_discoveries"], 0)
        self.assertEqual(signal["summary"]["naive_top10_decoys"], 5)
        null = evaluate(seed=42, permutations=499, signal_scale=0)
        self.assertEqual(null["summary"]["bh_selected_count"], 0)


if __name__ == "__main__":
    unittest.main()
