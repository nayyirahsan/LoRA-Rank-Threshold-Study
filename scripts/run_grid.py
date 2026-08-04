"""Expand a YAML grid into run configs and execute them in order, skipping finished runs.

    python scripts/run_grid.py configs/grid_sql.yaml --dry-run
    python scripts/run_grid.py configs/grid_facts.yaml --shard 0/2   # GPU 0 on Kaggle 2xT4
    CUDA_VISIBLE_DEVICES=1 python scripts/run_grid.py configs/grid_facts.yaml --shard 1/2

Grid format: `defaults` apply to every block. In a block, a list value becomes a sweep axis.
`rank` only applies to lora/qlora. `lr` is a number, a list, or a mapping by method. For
lora/qlora the mapping can go one level deeper, by rank:

    lr: {full: 3e-5, lora: {1: 1e-3, 64: 3e-4}}

Runs are ordered base -> full -> lora -> qlora within each (task, n). Full-FT checkpoints come
first because gap closure and the oracle both depend on them.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import sys
import traceback
from dataclasses import asdict

import yaml

from lorathresh.registry import Registry, run_id
from lorathresh.train import METHODS, RunConfig, run


def _as_list(v) -> list:
    return v if isinstance(v, list) else [v]


def resolve_lr(spec, method: str, rank: int) -> list[float]:
    if method == "base":
        return [0.0]
    if isinstance(spec, dict):
        if method not in spec:
            raise KeyError(f"lr has no entry for method {method!r}")
        spec = spec[method]
        if isinstance(spec, dict):
            if rank not in spec:
                raise KeyError(f"lr for {method!r} has no entry for rank {rank}")
            spec = spec[rank]
    return [float(x) for x in _as_list(spec)]


def expand(grid: dict) -> list[RunConfig]:
    defaults = grid.get("defaults", {})
    configs: dict[str, RunConfig] = {}
    for block in grid["blocks"]:
        spec = {**defaults, **block}
        lr_spec = spec.pop("lr", 0.0)
        methods = _as_list(spec.pop("method"))
        ranks = _as_list(spec.pop("rank", 0))
        axes = {k: _as_list(v) for k, v in spec.items()}
        for combo in itertools.product(*axes.values()):
            fixed = dict(zip(axes, combo))
            for method in methods:
                for rank in ranks if method in ("lora", "qlora") else [0]:
                    for lr in resolve_lr(lr_spec, method, rank):
                        cfg = RunConfig(**fixed, method=method, rank=rank, lr=lr)
                        configs.setdefault(run_id(asdict(cfg)), cfg)  # canonicalization dedupes base/full
    return sorted(
        configs.values(),
        key=lambda c: (c.task, c.n, METHODS.index(c.method), c.rank, c.scaling, c.lr, c.seed),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="run at most this many unfinished configs")
    ap.add_argument("--shard", default="0/1", help="i/k: run every k-th config starting at i")
    ap.add_argument("--registry", default="results/runs.jsonl")
    ap.add_argument("--output-dir", default="checkpoints")
    ap.add_argument("--sql-path", default=None)
    ap.add_argument("--push-repo", default=None)
    ap.add_argument("--micro-batch-size", type=int, default=None, help="split batches to fit memory (e.g. 8 for full FT on T4)")
    ap.add_argument("--max-consecutive-failures", type=int, default=3,
                    help="abort the shard after this many failed configs in a row (a systematic error)")
    args = ap.parse_args()

    with open(args.grid) as f:
        configs = expand(yaml.safe_load(f))
    i, k = map(int, args.shard.split("/"))
    configs = configs[i::k]
    registry = Registry(args.registry)
    if args.push_repo:
        registry.pull_from_hub(args.push_repo)  # a fresh session resumes where the last one stopped
    todo = [c for c in configs if not registry.is_done(asdict(c))]
    print(f"{len(configs)} configs in shard {args.shard}, {len(configs) - len(todo)} done, {len(todo)} to run")
    if args.limit is not None:
        todo = todo[: args.limit]
    if args.dry_run:
        for c in todo:
            print(f"  {run_id(asdict(c))}  {c.task:5} n={c.n:<6} {c.method:5} r={c.rank:<4} {c.scaling:8} lr={c.lr:.0e} seed={c.seed}")
        return 0
    # One failing config must not take the rest of the shard with it. In the first lr_cal run,
    # a LoRA import error ended a shard that still had queued runs. Repeated failures in a row
    # usually mean a systematic error, so stop rather than burn GPU time on a doomed queue.
    failed, consecutive = [], 0
    for n_done, cfg in enumerate(todo, 1):
        rid = run_id(asdict(cfg))
        print(f"\n=== [{n_done}/{len(todo)}] {asdict(cfg)}", flush=True)
        try:
            run(cfg, args.registry, args.output_dir, args.sql_path, args.push_repo, micro_batch_size=args.micro_batch_size)
            consecutive = 0
        except Exception:
            traceback.print_exc()
            failed.append(rid)
            consecutive += 1
            print(f"!!! run {rid} failed ({consecutive} in a row)", flush=True)
            _free_gpu_memory()
            if consecutive >= args.max_consecutive_failures:
                print(f"!!! {consecutive} consecutive failures: aborting shard", flush=True)
                break
    if failed:
        print(f"{len(failed)} run(s) failed: {failed}", flush=True)
        return 1
    return 0


def _free_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


if __name__ == "__main__":
    sys.exit(main())
