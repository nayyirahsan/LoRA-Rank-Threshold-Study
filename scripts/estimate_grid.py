"""Estimate GPU hours for a grid before spending Kaggle quota.

    python scripts/estimate_grid.py configs/grid_sql.yaml
    python scripts/estimate_grid.py configs/grid_facts.yaml --tps facts:full=1400 --tps facts:lora=1600

Training tokens per run = examples x mean tokens per example x epochs. Mean lengths were measured
on the real tokenizer (engineering log #8): facts 21.4 tokens per training example (4 templates per
fact), SQL 67. Throughput defaults are the lr_cal measurements on a T4 in fp16. Override them with
--tps as newer runs measure faster settings. Eval time and the in-kernel oracle sweep (one eval per
oracle rank for every full-FT run) are included. Wall clock divides by the number of GPUs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_grid import expand  # noqa: E402

MEAN_TOKENS = {"facts": 21.4, "sql": 67.0}
EXAMPLES_PER_UNIT = {"facts": 4, "sql": 1}  # facts: n counts facts, 4 training templates each
# tokens/s on one T4, fp16, measured in lr_cal (with micro-batch 8, before the token-cap fix)
DEFAULT_TPS = {("facts", "full"): 485.0, ("facts", "lora"): 370.0, ("sql", "full"): 1264.0, ("sql", "lora"): 1175.0}
# seconds per eval example, measured in lr_cal (facts ~16s/500, SQL ~10s/100 with 96-token generations)
EVAL_SECONDS_PER_EXAMPLE = {"facts": 0.032, "sql": 0.1}
ORACLE_RANKS = 11
LOAD_OVERHEAD_SECONDS = 90  # model load, data prep, checkpoint save


def run_seconds(cfg, tps: dict) -> float:
    if cfg.method == "base":
        train = 0.0
    else:
        tokens = cfg.n * EXAMPLES_PER_UNIT[cfg.task] * MEAN_TOKENS[cfg.task] * cfg.epochs
        method = "lora" if cfg.method in ("lora", "qlora") else cfg.method
        train = tokens / tps[(cfg.task, method)]
    return train + cfg.n_eval * EVAL_SECONDS_PER_EXAMPLE[cfg.task] + LOAD_OVERHEAD_SECONDS


def oracle_seconds(cfg) -> float:
    return ORACLE_RANKS * cfg.n_eval * EVAL_SECONDS_PER_EXAMPLE[cfg.task] + 120 if cfg.method == "full" else 0.0


def estimate(configs, tps: dict, gpus: int = 2) -> dict:
    per_run = [(c, run_seconds(c, tps), oracle_seconds(c)) for c in configs]
    gpu_seconds = sum(t + o for _, t, o in per_run)
    longest = max((t for _, t, _ in per_run), default=0.0)
    return {
        "runs": len(per_run),
        "gpu_hours": gpu_seconds / 3600,
        # a shard can't split a run, so the wall clock is at least the longest single run
        "wall_hours": max(gpu_seconds / gpus, longest) / 3600,
        "longest_run_hours": longest / 3600,
        "per_run": per_run,
    }


def parse_tps(values: list[str]) -> dict:
    tps = dict(DEFAULT_TPS)
    for v in values:
        key, rate = v.split("=")
        task, method = key.split(":")
        tps[(task, method)] = float(rate)
    return tps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--tps", action="append", default=[], help="task:method=tokens_per_second")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    configs = expand(yaml.safe_load(Path(args.grid).read_text()))
    est = estimate(configs, parse_tps(args.tps), args.gpus)
    if args.verbose:
        for c, t, o in est["per_run"]:
            print(f"  {c.task:5} n={c.n:<6} {c.method:5} r={c.rank:<4} ep={c.epochs:<4g} train+eval {t / 60:6.1f} min  oracle {o / 60:5.1f} min")
    print(f"{est['runs']} runs: {est['gpu_hours']:.1f} GPU-hours, ~{est['wall_hours']:.1f} h wall on {args.gpus} GPU(s), "
          f"longest run {est['longest_run_hours']:.1f} h")


if __name__ == "__main__":
    main()
