"""Generate the README's Results section from a run registry.

    python scripts/write_results.py --registry results/final/runs.jsonl --out results/final --figures figures

Runs the analysis (curves, r*, energy prediction, figures), computes each hypothesis check from the
numbers, and replaces the text between the README's results markers. Every verdict is computed
from the registry, never typed in, so the section can't claim more than the data supports.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import pandas as pd

from lorathresh.analysis import SETTING, THRESHOLD, _in_setting, analyze
from lorathresh.registry import Registry

START, END = "<!-- results:start -->", "<!-- results:end -->"
MARGIN = 0.05  # oracle-vs-LoRA gap-closure difference that counts as a real separation


def _rank(value) -> int | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else int(value)


def _ceiling(row) -> bool:
    """r* at the lowest rank tried is an upper bound: the true threshold may be lower. The first real-data
    run of this script reported "facts H1 fails: r* grows 4 -> 4" when 4 was the smallest rank tested."""
    r = _rank(row.r_star)
    return r is not None and r <= int(row.get("min_rank_tested", 1))


def _fmt(row) -> str:
    return f"≤ {_rank(row.r_star)}" if _ceiling(row) else f"= {_rank(row.r_star)}"


def h1_sql(rstar: pd.DataFrame) -> tuple[str, str]:
    rows = rstar[(rstar.task == "sql") & (rstar.arm == "lora")]
    if rows.empty:
        return "not tested", "no SQL LoRA results"
    row = rows.iloc[0]
    r = _rank(row.r_star)
    if r is None:
        return "fails", f"LoRA never reached {THRESHOLD:.0%} of full FT (max rank {int(row.max_rank_tested)})"
    return ("supported" if r <= 8 else "fails"), f"r* {_fmt(row)} (prediction: ≤ 8)"


def h1_facts(rstar: pd.DataFrame) -> tuple[str, str]:
    rows = rstar[(rstar.task == "facts") & (rstar.arm == "lora")].sort_values("n")
    if len(rows) < 2:
        return "not tested", "needs at least two facts sizes"
    lo, hi = rows.iloc[0], rows.iloc[-1]
    n_ratio = hi.n / lo.n
    r_lo, r_hi = _rank(lo.r_star), _rank(hi.r_star)
    need = n_ratio / 2  # the plan predicted >= 8x growth per 16x more facts
    if r_lo is None:
        return "inconclusive", f"threshold not reached even at N={lo.n:,}"
    if r_hi is None:
        bound = hi.max_rank_tested * 2 / r_lo  # true r_lo <= r_lo only makes growth larger
        verdict = "supported" if bound >= need else "inconclusive"
        return verdict, (f"r* {_fmt(lo)} at N={lo.n:,}; not reached by rank {int(hi.max_rank_tested)} at N={hi.n:,}, "
                         f"so growth ≥ {bound:.0f}× for {n_ratio:.0f}× more facts (prediction: ≥ {need:.0f}×)")
    if _ceiling(lo) and _ceiling(hi):
        return "inconclusive", (
            f"LoRA reaches {THRESHOLD:.0%} of full FT at the smallest tested rank ({int(hi.min_rank_tested)}) for every N "
            f"from {lo.n:,} to {hi.n:,}: capacity never binds in the tested range, so growth can't be measured")
    growth = r_hi / r_lo
    text = (f"r* {_fmt(lo)} at N={lo.n:,} and {_fmt(hi)} at N={hi.n:,} ({n_ratio:.0f}× more facts); "
            f"growth {'≥ ' if _ceiling(lo) else ''}{growth:.0f}×, prediction: ≥ {need:.0f}×")
    if growth >= need:
        return "supported", text
    # With a ceiling at small N, the measured growth is only a lower bound, so it can't refute the prediction.
    return ("inconclusive" if _ceiling(lo) else "fails"), text


def h2a(rstar: pd.DataFrame, energy: dict) -> tuple[str, str]:
    checks = []
    for key, e in energy.items():
        rs = _in_setting(rstar, key)
        lora = rs[rs.arm == "lora"]
        r = _rank(lora.iloc[0].r_star) if not lora.empty else None
        if r:
            q = e["energy_rank_90"] / r
            checks.append((key, r, e["energy_rank_90"], 0.5 <= q <= 2))
    if not checks:
        return "not tested", "needs full-FT spectra and a reached LoRA r*"
    hits = sum(ok for *_, ok in checks)
    detail = "; ".join(f"{k[0]} N={k[1]:,}: 90%-energy rank {e:.0f} vs r* {r}" for k, r, e, _ in checks)
    verdict = "supported" if hits == len(checks) else ("partly supported" if hits else "fails")
    return verdict, f"{hits}/{len(checks)} settings within 2×. {detail}"


def _settings(curves: pd.DataFrame) -> list[tuple]:
    keys = {tuple(r) for r in curves[SETTING].itertuples(index=False, name=None)}
    return sorted(keys, key=lambda k: (k[0] != "facts", k[1]))


def lora_minus_oracle(curves: pd.DataFrame, key: tuple) -> tuple[float | None, list[int]]:
    """Mean G(LoRA) - G(oracle) over the ranks both arms measured in one setting."""
    df = _in_setting(curves, key)
    lora = df[(df.arm == "lora") & (df["rank"] >= 1)].set_index("rank").G
    orc = df[(df.arm == "oracle") & (df["rank"] >= 1)].set_index("rank").G
    shared = sorted(lora.index.intersection(orc.index))
    if not shared:
        return None, []
    return float((lora[shared] - orc[shared]).mean()), [int(r) for r in shared]


def h2b(curves: pd.DataFrame) -> list[tuple[str, str]]:
    out = []
    for key in _settings(curves):
        diff, shared = lora_minus_oracle(curves, key)
        if diff is None:
            continue
        if diff > MARGIN:
            reading = "LoRA ≫ oracle: compact rank-r solutions exist that full FT didn't pick"
        elif diff < -MARGIN:
            reading = "oracle ≫ LoRA: an optimization/parameterization gap"
        else:
            reading = "oracle ≈ LoRA: consistent with a capacity limit"
        label = f"Facts N={key[1]:,}" if key[0] == "facts" else f"SQL n={key[1]:,}"
        out.append((label, f"mean G(LoRA) − G(oracle) = {diff:+.2f} over ranks {', '.join(map(str, shared))} → {reading}"))
    return out


def _rank_phrase(row) -> str:
    r = _rank(row.r_star)
    return f"rank {r} (the smallest tested)" if _ceiling(row) else f"rank {r}"


def resume_bullet(rstar: pd.DataFrame, curves: pd.DataFrame, n_runs: int | None) -> str:
    """One resume line built only from measured values. Thresholds met at the smallest tested rank are
    stated as such, and the oracle clause follows the sign and size of the measured gap."""
    findings = []
    sql = rstar[(rstar.task == "sql") & (rstar.arm == "lora")]
    if not sql.empty:
        row = sql.iloc[0]
        findings.append(f"{_rank_phrase(row)} on text-to-SQL" if _rank(row.r_star) is not None
                        else f"no tested rank (up to {int(row.max_rank_tested)}) on text-to-SQL")
    facts = rstar[(rstar.task == "facts") & (rstar.arm == "lora")].sort_values("n")
    if not facts.empty:
        row = facts.iloc[-1]
        findings.append(f"{_rank_phrase(row)} for {int(row.n):,} injected facts" if _rank(row.r_star) is not None
                        else f"no tested rank (up to {int(row.max_rank_tested)}) for {int(row.n):,} injected facts")
    size = f"{n_runs}-run " if n_runs else ""
    bullet = (f"Designed and ran a {size}LoRA rank × data-size study on Qwen3-0.6B against full fine-tuning "
              f"on free Kaggle T4s; found LoRA recovers ≥{THRESHOLD:.0%} of full fine-tuning's gain at "
              + " and ".join(findings or ["the tested ranks"]))
    diffs = [d for d, _ in (lora_minus_oracle(curves, k) for k in _settings(curves)) if d is not None]
    if diffs:
        mean = sum(diffs) / len(diffs)
        if mean > MARGIN:
            bullet += (f", and that trained LoRA beats SVD-truncated full-FT updates of equal rank by {mean:.2f} mean "
                       "gap closure: low-rank solutions exist that full fine-tuning doesn't find")
        elif mean < -MARGIN:
            bullet += (f", and diagnosed an optimization gap: SVD-truncated full-FT updates beat trained LoRA of "
                       f"equal rank by {-mean:.2f} mean gap closure")
        else:
            bullet += (", and that SVD-truncated full-FT updates match trained LoRA at equal rank, "
                       "consistent with a capacity limit")
    return bullet + "."


def render(result: dict, fig_dir: Path, readme_dir: Path, n_runs: int | None = None) -> str:
    rstar, curves, energy = result["r_star"], result["curves"], result.get("energy", {})
    seeds = int(curves.n_seeds.min()) if not curves.empty else 0
    lines = [
        "## Results",
        "",
        f"_Generated by `scripts/write_results.py` from the run registry. Minimum seeds per point: {seeds}. "
        "Every verdict below is computed from the numbers._",
        "",
        "| Hypothesis | Verdict | Evidence |",
        "|---|---|---|",
    ]
    for name, (verdict, evidence) in (("H1, skill task (SQL)", h1_sql(rstar)), ("H1, knowledge task (facts)", h1_facts(rstar)),
                                      ("H2a, energy rank predicts r*", h2a(rstar, energy))):
        lines.append(f"| {name} | **{verdict}** | {evidence} |")
    lines += ["", "**H2b, oracle truncation vs trained LoRA:**", ""]
    lines += [f"- {label}: {text}" for label, text in h2b(curves)] or ["- not tested"]
    lines += ["", "**Resume bullet** (generated from the numbers above):", "",
              f"> {resume_bullet(rstar, curves, n_runs)}"]
    lines += ["", "### Thresholds", "", result["table"].strip(), ""]
    for name, caption in (("gap_closure.png", "Gap closure vs rank: trained LoRA and the SVD oracle, per setting"),
                          ("spectrum.png", "Share of the full-FT update's energy in its top-r directions"),
                          ("ablation.png", "Ablations against standard LoRA")):
        path = fig_dir / name
        if path.exists():
            lines += [f"![{caption}]({os.path.relpath(path, readme_dir)})", ""]
    return "\n".join(lines).rstrip() + "\n"


def replace_section(readme: str, section: str) -> str:
    if START not in readme or END not in readme:
        raise SystemExit(f"README is missing the {START} / {END} markers")
    head, rest = readme.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{section}{END}{tail}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--readme", default="README.md")
    args = ap.parse_args()

    rows = Registry(args.registry).rows()
    result = analyze(rows, args.out, args.figures)
    if result["curves"].empty:
        raise SystemExit("no complete settings in the registry yet")
    n_runs = sum(1 for r in rows if r["config"].get("kind", "train") == "train")
    readme = Path(args.readme)
    section = render(result, Path(args.figures), readme.parent, n_runs)
    readme.write_text(replace_section(readme.read_text(), section))
    print(section)


if __name__ == "__main__":
    main()
