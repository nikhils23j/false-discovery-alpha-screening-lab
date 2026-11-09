"""Draw the README figures from the saved results (no simulation is re-run).

    python scripts/make_figures.py                       # reads results/study.json, writes docs/img/*.png
    python scripts/make_figures.py --input results/study.json --outdir docs/img

Every plotted value comes from results/study.json; nothing is typed in by hand. The chart code itself
lives in alpha_screen/plots.py so `python -m alpha_screen.plots` draws exactly the same PNGs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alpha_screen import plots  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=ROOT / "results" / "study.json")
    ap.add_argument("--outdir", type=Path, default=ROOT / "docs" / "img")
    args = ap.parse_args()
    study = json.loads(args.input.read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)
    for fn in (plots.fdr_power, plots.power_curve, plots.calibration, plots.showcase):
        fn(study, args.outdir)
    for p in sorted(args.outdir.glob("*.png")):
        print(f"wrote {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
