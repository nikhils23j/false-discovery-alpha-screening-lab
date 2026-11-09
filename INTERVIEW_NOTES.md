# Interview notes: False-Discovery Alpha Screening Lab

All numbers here come from `results/study.json` (full profile, fixed seeds) and `results/seed42.json` / `results/null-seed42.json`.

## 1. What the project is

It's a simulation lab for checking whether multiple-testing corrections really protect you when you screen a lot of trading-signal formulas. I built a synthetic stock market where I know which formulas are real. I generate 217 candidate "alphas" from a small formula grammar and test them on a held-out period. Then I measure how often each selection method picks fakes (empirical FDR/FWER) and how many real ones it finds (power), over hundreds of simulated markets.

## 2. How it works

**Pipeline:** market generator → formula grammar → train/holdout split → p-values → multiple-testing procedure → scoring against ground truth → Monte-Carlo aggregation.

- **Market (`market.py`)**
  - 40 assets in 4 sectors, 1,200 days, plus burn-in.
  - Ten observable AR(1) primitives per asset, with persistence from 0 to 0.99. Some share a sector component.
  - Only x0 (slow, φ = 0.97) and x1 (fast, φ = 0.3) enter next-day returns.
  - Returns also include a market factor, sector shocks, persistent sector and asset drifts (AR 0.97) and Student-t noise.
  - *Why:* I wanted the dependence that breaks naive statistics (persistent features plus persistent return components → autocorrelated daily IC) while keeping the ground truth exact.
  - Primitives and returns use separate `SeedSequence` streams, so decoys are independent of returns by construction. A test verifies this.
- **Grammar (`grammar.py`)**
  - Operators: `ts_mean`, `delta`, `ts_zscore`, `sector_neutral`, `add`/`sub` of cross-sectional z-scores, and `ts_corr`.
  - That gives 217 formulas: 53 touch a signal primitive, 164 use only decoys.
  - I excluded `sub(x0, x1)` (could cancel) and `ts_corr` with a signal primitive (zero IC by symmetry), so "true" and "null" are unambiguous.
  - Every operator is causal, and a test perturbs future data to prove it.
  - Lots of near-duplicates: in seed 42, 66 pairs have |corr| > 0.8, and the Li–Ji effective number of tests is about 83 out of 217.
- **Train/holdout (`lab.py`)**
  - Days 1–700 are used only to choose each formula's sign, and for the naive baseline (top-10 by |in-sample Sharpe|).
  - Days 701–1200 give each formula's mean daily cross-sectional rank IC.
  - A test checks that changing the holdout regime can't change any training decision.
- **Three kinds of p-value**
  - (a) Block permutation of holdout returns, with 20-day and 50-day blocks and 9,999 draws.
  - (b) Newey–West HAC t-test with 20 lags.
  - (c) Plain iid t-test.
  - Speed trick: precompute `S[b, c] = Σ x(block b)·y(block c)`. Then any block permutation's statistic is `Σ_b S[b, π(b)]`, so the permutations are cheap gathers. A unit test checks it against direct reindexing.
- **Procedures (`stats.py`)**
  - BH and BY step-up, Bonferroni, Holm step-down.
  - Romano–Wolf stepdown on studentised max-statistics, using the same permutation draws.
  - BH, BY and Holm are checked against hand-computed examples.
- **Extras**
  - Holdout long/short rank-weighted book with turnover and net Sharpe at 5 bps per unit traded.
  - Deflated Sharpe ratio for the naive winner.
  - Regime switch: holdout signal ×0.5 or ×0.
- **Study (`study.py`)**
  - Main scenario: 200 markets. Null-only: 100. Strength sweep: 7 scales × 30. Regime: 2 × 50.
  - Runs in parallel with `multiprocessing`.
  - Writes means with 95% CIs (normal CI for FDR/power, Wilson for FWER), plus environment, seeds and command.

## 3. Key results (measured) and what they don't show

Main scenario, 200 markets, q = 0.05:

- **BH + 20-day block-permutation p:** FDR **0.072** [0.054, 0.090], power **0.338**, and at least one false pick in 44% of markets.
- **BH + 50-day blocks:** FDR 0.058 [0.041, 0.076], power 0.267.
- **BH + iid t-test:** FDR **0.152** [0.132, 0.173]. **BH + Newey–West:** 0.106.
- **Uncorrected p ≤ 0.05:** FDR 0.240. At least one false pick in 99.5% of markets.
- **Familywise methods:** Bonferroni/Holm FDR 0.030, power 0.130. Romano–Wolf power 0.167. BY FDR 0.015, power 0.156.
- **Null calibration:** the share of truly-null formulas with p ≤ 0.05 was:
  - iid t: 0.096
  - Newey–West: 0.075
  - 20-day blocks: 0.066
  - 50-day blocks: 0.058
- **Null-only (100 markets):** the naive in-sample winner had a median annualised Sharpe of **2.0** with no signal at all. BH + iid t made at least one false discovery in 43% of markets, against 9% for BH + block permutation.
- **Strength sweep:** the naive top-10 was 62% decoys at scale 0.25 and 39% at 0.5, but 3–4% at scale 1. BH power rose from 0.01 to 0.88 across scales 0.25 → 2.0.
- **Costs:** 63.6% of 3,920 BH discoveries kept a positive net Sharpe at 5 bps. In seed 42 the fast signal x1 went from 2.00 gross to −0.91 net.
- **Seed 42 alone:** BH found 0 of 53 true formulas, which is a good example of why one seed proves nothing.
- **Original 50-formula experiment, seed 42:** 5/5 signals found, 0/45 decoys. The naive top-10 had 5 decoys. The null control picked 0/50.

**What they do NOT show:**

- Anything about real markets or live alpha.
- That BH controls FDR in general.
- That a 5 bps cost model is realistic.
- That my generator's dependence looks like real data.

They also show that my *best* p-value method was still slightly anti-conservative, so "BH held at 5%" would be a false claim.

## 4. Two-minute answer (STAR)

**Situation.** I'd been reading about quant research and kept seeing the same warning: if you try enough formulas, one will backtest beautifully by luck. I could generate dozens of formulas in an afternoon, and the textbook fix is a false-discovery-rate correction like Benjamini–Hochberg. But real market data has a lot of dependence, with persistent signals and formulas that are near-copies of each other, and I wasn't sure the guarantee survives that.

**Task.** I gave myself a question I could actually answer: in a market where I know the truth, does BH really keep false discoveries near 5%, and how does it compare with stricter and looser methods? Because nothing tells you the truth in real data, I had to build a simulator first.

**Action.** I wrote a synthetic market with persistent features, sector structure and slow drifts, where only two hidden features actually predict returns. On top of it I built a small formula grammar that generates 217 candidates, including lots of near-duplicate decoys. I picked each formula's direction on the first 700 days and tested on the last 500. I compared three ways of getting p-values (a block permutation, Newey–West, and a plain t-test) and six selection rules, then ran the whole thing over 610 simulated markets. I built it as a personal project with AI pair-programming help. The parts I made sure I understood and tested myself were the statistics: hand-checked BH and Holm cases, a no-look-ahead test, and a test that my fast permutation trick matches the slow version.

**Result.** The most interesting finding was that BH is only as good as its p-values. With plain t-tests, the false discovery rate came out at about 15%, three times the target. With block-permutation p-values it was about 7%, and longer blocks brought it to about 5.8%, but power dropped from 34% to 27%. So even my best method was slightly over-optimistic, and I report that rather than claiming it hit 5%. I also found that a strategy can be statistically real and still lose money after costs: a fast signal went from a Sharpe of 2.0 to −0.9. And one seed on its own found zero true signals, which made me much more careful about single backtests.

## 5. Likely follow-up questions

1. **Why does BH fail with iid t-tests if BH is a valid procedure?**
   BH assumes valid p-values. Persistent features and persistent return components make the daily IC autocorrelated, so the iid standard error is too small and nulls look significant. In my null-only runs about 10% of null p-values were below 0.05 instead of 5%.

2. **Why not just use Bonferroni?**
   In my runs it was safe (FDR 0.030) but found 13% of true formulas versus BH's 34%. With 217 heavily correlated formulas (about 83 effective tests) Bonferroni is very pessimistic. Whether that tradeoff is right depends on how expensive a false positive is.

3. **What does Romano–Wolf add?**
   It controls the familywise error using the joint permutation distribution of all statistics, so correlation between formulas is taken into account. Here it got slightly more power than Holm (0.167 vs 0.130). Its null-only FWER was 0.11 [0.063, 0.186], which inherits the same slight p-value miscalibration.

4. **How do you know there's no look-ahead?**
   There are three tests. Each grammar operator's past values don't change when future inputs change. Training decisions are identical when only the holdout regime changes. And the decoy primitives don't change when signal strength changes.

5. **Why is the block permutation still miscalibrated?**
   Some features have AR 0.99, and dependence lasting longer than 20 days crosses block boundaries. The permutation then treats blocks as more exchangeable than they are. 50-day blocks reduced the null rejection rate to 0.058 but cost power. I haven't tried a stationary bootstrap or subsampling yet.

6. **Is this realistic?**
   The structure is realistic-ish, but the numbers aren't calibrated to any real market. The signal sizes are chosen so the sweep covers both easy and hard cases. I wouldn't quote any of these rates as facts about real data.

7. **What was the hardest bug or decision?**
   Defining ground truth. Formulas like `sub(x0, x1)` or `ts_corr(x0, x5)` could have zero predictive power while still being related to the signal, so I excluded them rather than guess. The second was speed: I made permutations cheap with the block-matrix trick.

8. **What did the AI help with, and what did you do?**
   AI pair-programming sped up writing boilerplate, the plotting code and the first drafts of functions. I'm responsible for the experiment design, I checked the statistics against hand calculations, and I can explain every number in the README. If you want, I can walk through BH on a small example.

## 6. Things I should be able to explain cold

- [ ] BH step-up on a small example by hand, and why "step-up" means a larger p-value can be rescued
- [ ] FDR vs FWER, and why under the global null FDR equals FWER
- [ ] Why BY divides by the harmonic number, and the cost of that
- [ ] Holm step-down vs Bonferroni
- [ ] Romano–Wolf stepdown: studentised max-statistic over the remaining hypotheses, and monotone adjusted p-values
- [ ] Permutation p-value `(1 + #{null ≥ obs}) / (B + 1)`, and why the +1
- [ ] Why serial correlation inflates t-stats, what Newey–West does (Bartlett weights), and why it still over-rejected here
- [ ] Block permutation, and the block-length tradeoff (calibration vs power: 0.066 → 0.058 null rate, 0.338 → 0.267 power)
- [ ] The `S[b, π(b)]` trick for fast block permutations
- [ ] Rank IC: cross-sectional Spearman correlation per day, then averaged
- [ ] Why the sign is chosen on train only, and what look-ahead would look like
- [ ] Deflated Sharpe ratio: expected max Sharpe across N trials, and skew/kurtosis adjustment
- [ ] Effective number of tests (Li–Ji) and why near-duplicates matter
- [ ] Turnover and net Sharpe: why the fast x1 signal dies after costs
- [ ] Wilson interval vs normal interval for a rate near 0 or 1
- [ ] The exact limits: synthetic only, one generator, permutation floor 1/10,000, toy costs
