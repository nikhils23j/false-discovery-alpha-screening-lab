# False-Discovery Alpha Screening Lab

An AI-assisted portfolio exercise in testing cross-sectional predictive signals. The implementation and measurements were produced by Codex for Nikhil Jain on 30 October 2026. It uses a seeded synthetic market, not employer data or a live trading strategy.

## Research question

When 50 candidate expressions are screened, which survive a held-out test that accounts for multiple comparisons? The data generator injects five known predictive factors into next-day returns and leaves 45 independent decoys. This makes detection and false discoveries observable, unlike a real market where ground truth is unknown.

## Design

1. Generate 1,200 daily cross-sections for 12 assets. Each of the 50 candidate expressions is a bounded transform of a generated primitive. Five factors influence the next-day return; common and asset-specific noise are added.
2. Use the first 700 days only to choose each candidate's expected direction. A naive baseline selects the 10 largest absolute in-sample long/short Sharpe ratios.
3. On the final 500 days, compute mean daily cross-sectional rank information coefficient (IC). Reorder 20-day blocks of held-out returns 499 times to obtain a one-sided null p-value for each predeclared candidate. This preserves within-block return order while breaking most factor/return alignment.
4. Apply the Benjamini-Hochberg step-up procedure at `q=0.05` across all 50 p-values. Compare selected names with injected ground truth and the naive top-10 list.

This is a bounded methodology demonstration. The synthetic generator is intentionally simple; its five true signals are stronger and more stationary than plausible production alphas. Benjamini-Hochberg controls expected false-discovery rate only under its statistical assumptions. One seeded result is not a guarantee of a 5% realized error rate, and no result establishes investment profitability.

## Run

Python 3.10+ and NumPy are required. From this folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m alpha_screen --seed 42 --permutations 499 --output results/seed42.json
python -m alpha_screen --seed 42 --permutations 499 --signal-scale 0 --output results/null-seed42.json
```

The two JSON files include the workload, all candidate ICs and p-values, selected expressions, and summary counts. `results/seed42.json` and `results/null-seed42.json` contain the recorded runs. No network access or paid data is needed after NumPy is installed.

## Evaluation target and observed result

The proposal target was to detect at least four of five injected signals and no more than two false discoveries among 45 decoys, compared with naive in-sample top-Sharpe selection. The observed seed-42 run detected **5/5 injected signals with 0/45 false discoveries**. The naive top-10 list contained five injected signals and five decoys. In the null-only control, BH selected **0/50** candidates. Four unit tests cover reproducibility, train/test separation, block shuffling, BH selection, signal detection, and the null control.

These observations depend on the stated seed, signal strength, generator, 12 assets, 1,200 days, 20-day blocks, and 499 permutations. They are detection results on generated data, not market returns, trading P&L, a live alpha, or a broad power study. A real application would need point-in-time data, corporate-action handling, universe rules, costs, regime checks, and a truly untouched final evaluation set.
