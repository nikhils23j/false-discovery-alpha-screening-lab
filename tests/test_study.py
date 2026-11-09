import math
import unittest

import numpy as np

from alpha_screen import stats
from alpha_screen.grammar import build_grammar, cs_zscore, delta, evaluate_all, ts_corr, ts_mean, ts_zscore
from alpha_screen.lab import (
    block_cross_matrix, block_permutation_null, build_panel, candidate_family, run_trial,
)
from alpha_screen.market import MarketConfig, simulate
from alpha_screen.study import aggregate

SMALL = MarketConfig(assets=12, days=400, train_days=200)


class MultipleTestingByHand(unittest.TestCase):
    # p sorted: .001 .008 .039 .041 .042 .060 .074 .205 ; m = 8, q = .05
    P = np.array([0.041, 0.001, 0.074, 0.039, 0.205, 0.008, 0.042, 0.060])

    def test_bh_hand_computed(self):
        # thresholds i*q/m = .00625 .0125 .01875 .025 .03125 .0375 .04375 .05
        # largest i with p_(i) <= threshold is i=2 (.008 <= .0125); p_(3)=.039 > .01875 ...
        # but step-up checks every i: i=7 .074>.04375, i=6 .060>.0375, i=5 .042>.03125 -> k=2
        expected = np.zeros(8, bool)
        expected[[1, 5]] = True
        np.testing.assert_array_equal(stats.benjamini_hochberg(self.P, 0.05), expected)

    def test_bh_step_up_rescues_a_larger_p(self):
        # p_(1)=.03 fails its own threshold .0125 but p_(4)=.05 <= .05, so all four are rejected
        p = np.array([0.04, 0.03, 0.05, 0.045])
        self.assertTrue(stats.benjamini_hochberg(p, 0.05).all())
        self.assertFalse(stats.bonferroni(p, 0.05).any())

    def test_bh_ties_none_and_validation(self):
        self.assertTrue(stats.benjamini_hochberg(np.array([0.02, 0.02, 0.02]), 0.05).all())
        self.assertFalse(stats.benjamini_hochberg(np.array([0.5, 0.9]), 0.05).any())
        with self.assertRaises(ValueError):
            stats.benjamini_hochberg(np.array([1.2]))
        with self.assertRaises(ValueError):
            stats.benjamini_hochberg(np.array([0.1]), q=0)

    def test_by_holm_bonferroni_hand_computed(self):
        # H_8 = 2.717857; BY thresholds i*.05/(8*2.717857) = .0022996*i -> only .001 passes (i=1), .008 > .0046
        by = stats.benjamini_yekutieli(self.P, 0.05)
        self.assertEqual(np.flatnonzero(by).tolist(), [1])
        # Holm: .001 <= .05/8=.00625 yes; .008 <= .05/7=.00714 no -> stop
        self.assertEqual(np.flatnonzero(stats.holm(self.P, 0.05)).tolist(), [1])
        self.assertEqual(np.flatnonzero(stats.bonferroni(self.P, 0.05)).tolist(), [1])
        # Holm beats Bonferroni: .01 <= .025 then .04 <= .05/1 -> both; Bonferroni (.025) keeps only .01
        p = np.array([0.01, 0.04])
        self.assertTrue(stats.holm(p, 0.05).all())
        self.assertEqual(stats.bonferroni(p, 0.05).tolist(), [True, False])

    def test_procedure_nesting(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            p = np.concatenate([rng.uniform(size=30), rng.uniform(0, 0.01, size=rng.integers(0, 10))])
            bh, by = stats.benjamini_hochberg(p), stats.benjamini_yekutieli(p)
            ho, bo = stats.holm(p), stats.bonferroni(p)
            self.assertTrue(np.all(by <= bh) and np.all(bo <= ho) and np.all(ho <= bh))


class PermutationAndRomanoWolf(unittest.TestCase):
    def test_permutation_p_formula(self):
        null = np.array([[0.1, 5.0], [0.3, 6.0], [0.5, 7.0]])
        np.testing.assert_allclose(stats.permutation_p_values(np.array([0.3, 10.0]), null), [3 / 4, 1 / 4])

    def test_romano_wolf_single_hypothesis_matches_permutation_p(self):
        rng = np.random.default_rng(1)
        null = rng.normal(size=(999, 1))
        obs = np.array([1.7])
        _, adj = stats.romano_wolf(obs, null)
        self.assertAlmostEqual(adj[0], stats.permutation_p_values(obs, null)[0])

    def test_romano_wolf_adjusted_never_below_raw(self):
        rng = np.random.default_rng(2)
        null = rng.normal(size=(1999, 15))
        obs = np.concatenate([[4.0, 3.5, 3.0], rng.normal(size=12)])
        rej, adj = stats.romano_wolf(obs, null)
        raw = stats.permutation_p_values((obs - null.mean(0)) / null.std(0), (null - null.mean(0)) / null.std(0))
        self.assertTrue(np.all(adj >= raw - 1e-12))
        self.assertTrue(rej[:3].all())

    def test_fast_block_permutation_equals_direct_reindexing(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(100, 6, 4))
        y = rng.normal(size=(100, 6))
        cross = block_cross_matrix(x, y, 20)
        np.testing.assert_allclose(np.trace(cross, axis1=0, axis2=1), np.einsum("daf,da->f", x, y) / 600)
        perm = np.array([3, 0, 4, 1, 2])
        idx = (perm[:, None] * 20 + np.arange(20)).ravel()
        direct = np.einsum("daf,da->f", x, y[idx]) / 600
        fast = sum(cross[b, perm[b]] for b in range(5))
        np.testing.assert_allclose(fast, direct)
        null = block_permutation_null(cross, 50, np.random.default_rng(4))
        self.assertEqual(null.shape, (50, 4))


class SerialDependenceTools(unittest.TestCase):
    def test_newey_west_lag0_is_iid_se(self):
        x = np.random.default_rng(5).normal(size=(300, 3))
        np.testing.assert_allclose(stats.newey_west_se(x, 0), x.std(axis=0) / math.sqrt(300))

    def test_newey_west_widens_for_persistent_series(self):
        rng = np.random.default_rng(6)
        e = rng.normal(size=5000)
        ar = np.empty(5000)
        ar[0] = e[0]
        for t in range(1, 5000):
            ar[t] = 0.9 * ar[t - 1] + e[t]
        self.assertGreater(stats.newey_west_se(ar[:, None], 30)[0], 2 * stats.newey_west_se(ar[:, None], 0)[0])

    def test_deflated_sharpe_falls_with_more_trials(self):
        rng = np.random.default_rng(7)
        pnl = rng.normal(0.05, 1, size=750)
        trials = rng.normal(0, 0.04, size=200)
        self.assertGreater(stats.deflated_sharpe_ratio(pnl, trials, 5), stats.deflated_sharpe_ratio(pnl, trials, 500))

    def test_wilson_and_effective_tests(self):
        lo, hi = stats.wilson_interval(5, 100)
        self.assertAlmostEqual(lo, 0.02154, places=4)
        self.assertAlmostEqual(hi, 0.11175, places=4)
        self.assertAlmostEqual(stats.effective_number_of_tests(np.eye(7)), 7.0)
        self.assertAlmostEqual(stats.effective_number_of_tests(np.ones((7, 7))), 1.0)


class GrammarIsCausalAndCorrect(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8)
        self.x = rng.normal(size=(80, 5))
        self.y = rng.normal(size=(80, 5))

    def test_operators_match_naive_loops(self):
        t = 50
        np.testing.assert_allclose(ts_mean(self.x, 10)[t], self.x[t - 9:t + 1].mean(axis=0))
        np.testing.assert_allclose(delta(self.x, 5)[t], self.x[t] - self.x[t - 5])
        w = self.x[t - 19:t + 1]
        np.testing.assert_allclose(ts_zscore(self.x, 20)[t], (self.x[t] - w.mean(0)) / w.std(0), rtol=1e-8)
        wy = self.y[t - 19:t + 1]
        expect = [np.corrcoef(w[:, a], wy[:, a])[0, 1] for a in range(5)]
        np.testing.assert_allclose(ts_corr(self.x, self.y, 20)[t], expect, rtol=1e-8)
        np.testing.assert_allclose(cs_zscore(self.x).mean(axis=1), 0, atol=1e-12)
        self.assertTrue(np.isnan(ts_mean(self.x, 10)[8]).all())

    def test_every_expression_is_causal(self):
        exprs = build_grammar(4, {0})
        prims = [np.random.default_rng(i).normal(size=(120, 6)) for i in range(4)]
        sectors = np.arange(6) % 2
        base = evaluate_all(exprs, prims, sectors)
        future = [p.copy() for p in prims]
        for p in future:
            p[90:] += 100.0
        moved = evaluate_all(exprs, future, sectors)
        np.testing.assert_array_equal(base[:90], moved[:90])

    def test_family_size_and_ambiguous_exclusions(self):
        exprs, uses_signal = candidate_family()
        names = [e.name for e in exprs]
        self.assertEqual(len(exprs), 217)
        self.assertEqual(int(uses_signal.sum()), 53)
        self.assertNotIn("sub(x0, x1)", names)
        self.assertFalse(any(n.startswith("ts_corr(x0") or n.startswith("ts_corr(x1") for n in names))


class NoLookAheadAndGroundTruth(unittest.TestCase):
    def test_decoys_do_not_depend_on_signal_strength(self):
        a, ra, _ = simulate(11, SMALL, signal_scale=0.0)
        b, rb, _ = simulate(11, SMALL, signal_scale=2.0)
        for p in range(len(a)):
            np.testing.assert_array_equal(a[p], b[p])
        self.assertFalse(np.allclose(ra, rb))

    def test_holdout_multiplier_only_changes_holdout_returns(self):
        _, r1, _ = simulate(12, SMALL, holdout_multiplier=1.0)
        _, r0, _ = simulate(12, SMALL, holdout_multiplier=0.0)
        cut = SMALL.burn_in + SMALL.train_days
        np.testing.assert_array_equal(r1[:cut], r0[:cut])
        self.assertFalse(np.allclose(r1[cut:], r0[cut:]))

    def test_training_decisions_ignore_holdout(self):
        # Changing holdout-period signal strength cannot change training-window choices
        a = run_trial(13, SMALL, holdout_multiplier=1.0, permutations=199, detail=True)
        b = run_trial(13, SMALL, holdout_multiplier=0.0, permutations=199, detail=True)
        for ca, cb in zip(a["candidates"], b["candidates"]):
            self.assertEqual(ca["direction"], cb["direction"])
            self.assertEqual(ca["in_sample_annual_sharpe"], cb["in_sample_annual_sharpe"])
        self.assertEqual(a["naive_top1"]["annual_sharpe_in_sample"], b["naive_top1"]["annual_sharpe_in_sample"])
        self.assertEqual(b["m_true"], 0)  # dead holdout => every discovery counts as false

    def test_fast_rank_matches_original_rank(self):
        from alpha_screen.core import rank_standardize as slow
        from alpha_screen.lab import rank_standardize as fast
        a = np.random.default_rng(15).normal(size=(30, 9, 5))
        np.testing.assert_allclose(fast(a), slow(a))

    def test_panel_has_no_nans_and_trial_aggregates(self):
        x, y, yr = build_panel(14, SMALL, 1.0)
        self.assertEqual(x.shape[:2], (SMALL.days, SMALL.assets))
        self.assertFalse(np.isnan(x).any())
        trials = [run_trial(s, SMALL, permutations=199) for s in (1, 2)]
        agg = aggregate(trials)
        self.assertEqual(agg["trials"], 2)
        self.assertIn("bh_block_perm", agg["methods"])


if __name__ == "__main__":
    unittest.main()
