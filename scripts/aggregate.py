"""Build the analysis from the run registry: curves.csv, r_star.csv/.md, and figures.

    python scripts/aggregate.py --registry results/runs.jsonl --out results --figures figures
"""
from __future__ import annotations

import argparse

from lorathresh.analysis import analyze
from lorathresh.registry import Registry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="results/runs.jsonl")
    ap.add_argument("--out", default="results")
    ap.add_argument("--figures", default="figures")
    args = ap.parse_args()

    result = analyze(Registry(args.registry).rows(), args.out, args.figures)
    if result["curves"].empty:
        print("No complete settings yet: each setting needs a base run and at least one full-FT run.")
        return
    print(result["table"])
    print("wrote:", ", ".join(str(p) for p in result["figures"]))


if __name__ == "__main__":
    main()
