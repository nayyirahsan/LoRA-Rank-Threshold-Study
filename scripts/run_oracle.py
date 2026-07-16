"""Oracle rank sweep for one full-FT checkpoint (hypothesis H2).

    python scripts/run_oracle.py --ft-run-id 3f2a9c1b04de --ranks 0,1,2,4,8,16,32,64,128,256,512,1024

Records two kinds of registry rows:
  kind=spectrum  per-module energy ranks and the fraction of dW energy captured at each rank (H2a)
  kind=oracle    task metrics for W_base + SVD_r(dW) at each rank (H2b)

Evaluation uses the same task, eval split and precision as the FT run, so an oracle row can be
compared directly with a LoRA row at the same (task, n, rank).
"""
from __future__ import annotations

import argparse
import statistics
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from lorathresh.oracle import DeltaSVD, energy_captured
from lorathresh.registry import Registry
from lorathresh.train import RunConfig, device_and_amp, evaluate, load_task


def find_run(registry: Registry, rid: str) -> dict:
    for row in registry.rows():
        if row["run_id"] == rid:
            return row
    raise SystemExit(f"run {rid} not found in {registry.path}")


def resolve_checkpoint(ft_row: dict, hub_repo: str | None) -> str:
    """The local checkpoint if it's still on disk, otherwise download it from the Hub (fresh session)."""
    local = Path(ft_row["metrics"]["checkpoint"])
    if local.exists():
        return str(local)
    if not hub_repo:
        raise SystemExit(f"checkpoint {local} is missing locally; pass --push-repo to download it from the Hub")
    from huggingface_hub import snapshot_download

    root = snapshot_download(hub_repo, repo_type="model", allow_patterns=[f"{ft_row['run_id']}/*"])
    return str(Path(root) / ft_row["run_id"])


def spectrum_summary(oracle: DeltaSVD, ranks: list[int]) -> dict:
    by_type: dict[str, list[int]] = {}
    for name, r in oracle.energy_ranks(0.9).items():
        by_type.setdefault(name.split(".")[-1], []).append(r)
    total = {name: float(s.double().square().sum()) for name, s in oracle.singular_values.items()}
    grand = sum(total.values())
    return {
        "energy_rank_90_median": statistics.median(r for rs in by_type.values() for r in rs),
        "energy_rank_90_median_by_module": {k: statistics.median(v) for k, v in by_type.items()},
        "energy_rank_90_by_module": oracle.energy_ranks(0.9),
        # Two aggregates: mean per module (every matrix counts equally) and energy-weighted (large updates count more).
        "energy_captured_mean": {
            r: statistics.mean(energy_captured(s, r) for s in oracle.singular_values.values()) for r in ranks
        },
        "energy_captured_weighted": {
            r: sum(energy_captured(s, r) * total[n] for n, s in oracle.singular_values.items()) / grand for r in ranks
        },
        "delta_frobenius_norm": grand ** 0.5,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-run-id", required=True)
    ap.add_argument("--ranks", default="0,1,2,4,8,16,32,64,128,256,512,1024")
    ap.add_argument("--registry", default="results/runs.jsonl")
    ap.add_argument("--sql-path", default=None)
    ap.add_argument("--svd-device", default="cpu", help="cuda speeds up the SVDs if the GPU has room")
    ap.add_argument("--push-repo", default=None, help="Hub repo holding checkpoints and the registry mirror")
    args = ap.parse_args()

    registry = Registry(args.registry)
    if args.push_repo:
        registry.pull_from_hub(args.push_repo)
    ft_row = find_run(registry, args.ft_run_id)
    cfg = RunConfig(**ft_row["config"])
    if cfg.method != "full":
        raise SystemExit(f"run {args.ft_run_id} is {cfg.method}, not full FT")
    ranks = [int(r) for r in args.ranks.split(",")]

    base = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.float32)
    tuned = AutoModelForCausalLM.from_pretrained(resolve_checkpoint(ft_row, args.push_repo), dtype=torch.float32)
    oracle = DeltaSVD(base, tuned, device=args.svd_device)
    del tuned

    spectrum_key = {"kind": "spectrum", "ft_run_id": args.ft_run_id}
    if not registry.is_done(spectrum_key):
        summary = spectrum_summary(oracle, ranks)
        registry.record(spectrum_key, summary)
        if args.push_repo:
            registry.push_to_hub(args.push_repo)
        print("median 90%-energy rank:", summary["energy_rank_90_median"], summary["energy_rank_90_median_by_module"])

    device, amp_dtype = device_and_amp()
    work = base.to(device=device, dtype=amp_dtype or torch.float32)
    # Non-target parameters (embeddings, norms) stay at base, as they do under LoRA.
    _, eval_examples = load_task(cfg, args.sql_path)
    for r in ranks:
        key = {"kind": "oracle", "ft_run_id": args.ft_run_id, "rank": r, "ft_config": asdict(cfg)}
        if registry.is_done(key):
            print(f"skip oracle r={r}")
            continue
        oracle.apply(work, r)
        metrics = evaluate(work, _tokenizer(cfg), eval_examples, cfg, device, amp_dtype)
        registry.record(key, metrics)
        if args.push_repo:
            registry.push_to_hub(args.push_repo)
        print(f"oracle r={r}: {metrics}")


def _tokenizer(cfg: RunConfig):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


if __name__ == "__main__":
    main()
