import argparse
import json
from pathlib import Path

from .core import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate candidate alphas on a seeded synthetic market")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--signal-scale", type=float, default=1.0, help="Set to 0 for a null-only control")
    parser.add_argument("--output", type=Path, default=Path("results/seed42.json"))
    args = parser.parse_args()
    result = evaluate(seed=args.seed, permutations=args.permutations, signal_scale=args.signal_scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
