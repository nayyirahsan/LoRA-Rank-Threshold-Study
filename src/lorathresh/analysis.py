"""From registry rows to the findings: gap-closure curves, r*, the energy-rank prediction, and figures.

Definitions (the README states the same):
  setting    (task, n, epochs, n_eval, max_steps). Runs are compared only within one setting, so
             LR-calibration rows (different n_eval) never mix with grid rows.
  G(r)       (score_arm - score_base) / (score_FT - score_base), using seed means.
             Undefined (NaN) when full FT gains < MIN_FT_GAIN over base.
  LR choice  within each (setting, arm, rank), the LR with the best mean score. Full FT gets the
             same choice, so both sides of G carry the same mild optimism from selecting on eval.
  CI         percentile bootstrap over seeds, resampling the arm's and full FT's seeds separately.
  r*         smallest rank with G >= THRESHOLD (mean), and a stricter version needing the CI's lower
             bound >= THRESHOLD. Read from measured ranks only, never interpolated; oracle curves
             can be non-monotone (docs/engineering_log.md #5).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY_METRIC = {"facts": "acc", "sql": "exec_match"}
THRESHOLD = 0.95
ENERGY_FRACTION_LABEL = "90%"
MIN_FT_GAIN = 0.02
SETTING = ["task", "n", "epochs", "n_eval", "max_steps"]
ADAPTER_ARMS = ("lora", "rslora", "qlora")

TRAIN_COLS = ["run_id", *SETTING, "arm", "rank", "lr", "seed", "score"]
ORACLE_COLS = ["ft_run_id", *SETTING, "lr", "seed", "rank", "score"]

# Light-mode chart tokens from the dataviz reference palette. Validated with validate_palette.js:
# lora/oracle all-pairs, rslora/qlora all-pairs, and the ordinal ramp (see docs/engineering_log.md #10).
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ARM_COLOR = {"lora": "#2a78d6", "oracle": "#eb6834", "rslora": "#1baf7a", "qlora": "#eda100"}
CONTEXT_GRAY = AXIS
ORDINAL_BLUE = ["#6da7ec", "#2a78d6", "#184f95"]  # steps 300 / 450 / 600
ARM_LABEL = {
    "lora": "LoRA (α/r)",
    "oracle": "Oracle: SVD-truncated full-FT update",
    "rslora": "rsLoRA (α/√r)",
    "qlora": "QLoRA (4-bit base)",
}


# ---------------------------------------------------------------- loading

def _arm(cfg: dict) -> str:
    if cfg["method"] == "qlora":
        return "qlora"
    if cfg["method"] == "lora":
        return "rslora" if cfg["scaling"] == "rslora" else "lora"
    return cfg["method"]


def _setting_of(cfg: dict) -> dict:
    return {
        "task": cfg["task"],
        "n": cfg["n"],
        "epochs": cfg["epochs"],
        "n_eval": cfg["n_eval"],
        "max_steps": -1 if cfg.get("max_steps") is None else cfg["max_steps"],  # NaN keys vanish in groupby
    }


def load(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    train, oracle, spectrum = [], [], []
    for row in rows:
        cfg, metrics = row["config"], row["metrics"]
        kind = cfg.get("kind", "train")
        if kind == "train":
            train.append({
                "run_id": row["run_id"], **_setting_of(cfg), "arm": _arm(cfg), "rank": cfg["rank"],
                "lr": cfg["lr"], "seed": cfg["seed"], "score": metrics[PRIMARY_METRIC[cfg["task"]]],
            })
        elif kind == "oracle":
            ft = cfg["ft_config"]
            oracle.append({
                "ft_run_id": cfg["ft_run_id"], **_setting_of(ft), "lr": ft["lr"], "seed": ft["seed"],
                "rank": cfg["rank"], "score": metrics[PRIMARY_METRIC[ft["task"]]],
            })
        elif kind == "spectrum":
            spectrum.append({"ft_run_id": cfg["ft_run_id"], **metrics})
    return pd.DataFrame(train, columns=TRAIN_COLS), pd.DataFrame(oracle, columns=ORACLE_COLS), spectrum


def _in_setting(df: pd.DataFrame, key: tuple) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    for col, value in zip(SETTING, key):
        mask &= (df[col] == value).to_numpy()
    return df[mask]


# ---------------------------------------------------------------- statistics

def references(train: pd.DataFrame) -> dict[tuple, dict]:
    """Per setting: base score, the chosen full-FT LR, and full-FT scores per seed at that LR."""
    refs: dict[tuple, dict] = {}
    base_rows = train[train.arm == "base"]
    for key, full in train[train.arm == "full"].groupby(SETTING):
        task, n, _, n_eval, _ = key
        base = base_rows[(base_rows.task == task) & (base_rows.n == n) & (base_rows.n_eval == n_eval)]
        if base.empty:
            continue
        ft_lr = full.groupby("lr")["score"].mean().idxmax()
        refs[key] = {
            "base": float(base.score.mean()),
            "ft_lr": float(ft_lr),
            "ft_scores": full[full.lr == ft_lr].score.to_numpy(dtype=float),
        }
    return refs


def gap_closure(arm_mean: float, base: float, ft_mean: float) -> float:
    gain = ft_mean - base
    return float("nan") if gain < MIN_FT_GAIN else (arm_mean - base) / gain


def bootstrap_ci(arm_scores, ft_scores, base: float, reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    arm, ft = np.asarray(arm_scores, dtype=float), np.asarray(ft_scores, dtype=float)
    point = gap_closure(arm.mean(), base, ft.mean())
    if math.isnan(point):
        return point, point, point
    rng = np.random.default_rng(seed)
    arm_means = rng.choice(arm, size=(reps, len(arm))).mean(axis=1)
    ft_means = rng.choice(ft, size=(reps, len(ft))).mean(axis=1)
    gains = ft_means - base
    g = np.where(gains >= MIN_FT_GAIN, (arm_means - base) / np.maximum(gains, 1e-12), np.nan)
    lo, hi = np.nanpercentile(g, [2.5, 97.5])
    return point, float(lo), float(hi)


def _curve_row(key: tuple, arm: str, rank: int, lr: float, scores: np.ndarray, ref: dict) -> dict:
    g, lo, hi = bootstrap_ci(scores, ref["ft_scores"], ref["base"])
    return {
        **dict(zip(SETTING, key)), "arm": arm, "rank": int(rank), "lr": float(lr), "n_seeds": len(scores),
        "score_mean": float(np.mean(scores)), "base": ref["base"], "ft_mean": float(ref["ft_scores"].mean()),
        "G": g, "G_lo": lo, "G_hi": hi,
    }


def curves(train: pd.DataFrame, oracle: pd.DataFrame, refs: dict[tuple, dict]) -> pd.DataFrame:
    rows = []
    for key, ref in refs.items():
        adapters = _in_setting(train, key)
        for (arm, rank), g in adapters[adapters.arm.isin(ADAPTER_ARMS)].groupby(["arm", "rank"]):
            best_lr = g.groupby("lr")["score"].mean().idxmax()
            rows.append(_curve_row(key, arm, rank, best_lr, g[g.lr == best_lr].score.to_numpy(dtype=float), ref))
        orc = _in_setting(oracle, key)
        orc = orc[orc.lr == ref["ft_lr"]]  # only truncations of the full-FT runs used as the reference
        for rank, g in orc.groupby("rank"):
            rows.append(_curve_row(key, "oracle", rank, ref["ft_lr"], g.score.to_numpy(dtype=float), ref))
    df = pd.DataFrame(rows)
    return df.sort_values([*SETTING, "arm", "rank"]).reset_index(drop=True) if not df.empty else df


def r_star(curve_df: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    rows = []
    for key, g in curve_df[curve_df["rank"] >= 1].groupby([*SETTING, "arm"]):
        g = g.sort_values("rank")
        hit, strict = g[g.G >= threshold], g[g.G_lo >= threshold]
        rows.append({
            **dict(zip([*SETTING, "arm"], key)),
            "r_star": int(hit["rank"].iloc[0]) if len(hit) else None,
            "r_star_ci": int(strict["rank"].iloc[0]) if len(strict) else None,
            "max_rank_tested": int(g["rank"].max()),
            "min_seeds": int(g.n_seeds.min()),
        })
    return pd.DataFrame(rows)


def energy_by_setting(spectrum: list[dict], train: pd.DataFrame, refs: dict[tuple, dict]) -> dict[tuple, dict]:
    """Per setting, over the reference full-FT runs: median 90%-energy rank and mean energy-captured curve."""
    full = train[train.arm == "full"].set_index("run_id")
    collected: dict[tuple, list[dict]] = {}
    for s in spectrum:
        if s["ft_run_id"] not in full.index:
            continue
        run = full.loc[s["ft_run_id"]]
        key = tuple(run[col].item() if hasattr(run[col], "item") else run[col] for col in SETTING)
        if key in refs and float(run.lr) == refs[key]["ft_lr"]:
            collected.setdefault(key, []).append(s)
    out = {}
    for key, specs in collected.items():
        ranks = sorted({int(r) for s in specs for r in s["energy_captured_weighted"]})
        out[key] = {
            "energy_rank_90": float(np.median([s["energy_rank_90_median"] for s in specs])),
            "captured": {r: float(np.mean([s["energy_captured_weighted"][str(r)] if str(r) in s["energy_captured_weighted"]
                                           else s["energy_captured_weighted"][r] for s in specs])) for r in ranks},
            "n_ft_runs": len(specs),
        }
    return out


# ---------------------------------------------------------------- table view

def _fmt_rank(r) -> str:
    return "not reached" if r is None or (isinstance(r, float) and math.isnan(r)) else str(int(r))


def rstar_table(curve_df: pd.DataFrame, rstar_df: pd.DataFrame, energy: dict[tuple, dict]) -> str:
    lines = [
        f"| Task | N | Base | Full FT | r* LoRA | r* LoRA (CI-strict) | r* oracle | {ENERGY_FRACTION_LABEL}-energy rank of ΔW | Prediction / r* LoRA | Seeds |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in _ordered_settings(curve_df):
        rs = _in_setting(rstar_df, key).set_index("arm")
        ref = _in_setting(curve_df, key).iloc[0]
        lora = rs.loc["lora"] if "lora" in rs.index else None
        orc = rs.loc["oracle"] if "oracle" in rs.index else None
        e = energy.get(key)
        ratio = ""
        if e and lora is not None and lora.r_star is not None and not pd.isna(lora.r_star):
            q = e["energy_rank_90"] / lora.r_star
            ratio = f"{q:.2f}× ({'within' if 0.5 <= q <= 2 else 'outside'} 2×)"
        lines.append(
            f"| {key[0]} | {key[1]:,} | {ref.base:.3f} | {ref.ft_mean:.3f} "
            f"| {_fmt_rank(lora.r_star) if lora is not None else '—'} "
            f"| {_fmt_rank(lora.r_star_ci) if lora is not None else '—'} "
            f"| {_fmt_rank(orc.r_star) if orc is not None else '—'} "
            f"| {e['energy_rank_90']:.0f} | {ratio} | {int(lora.min_seeds) if lora is not None else '—'} |"
            if e else
            f"| {key[0]} | {key[1]:,} | {ref.base:.3f} | {ref.ft_mean:.3f} "
            f"| {_fmt_rank(lora.r_star) if lora is not None else '—'} "
            f"| {_fmt_rank(lora.r_star_ci) if lora is not None else '—'} "
            f"| {_fmt_rank(orc.r_star) if orc is not None else '—'} | — | — | {int(lora.min_seeds) if lora is not None else '—'} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- figures

def _ordered_settings(df: pd.DataFrame) -> list[tuple]:
    keys = {tuple(r) for r in df[SETTING].itertuples(index=False, name=None)}
    return sorted(keys, key=lambda k: (k[0] != "facts", k[1], k[2], k[3], k[4]))


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        # A short key gets split into dashes by the marker's surface-colored ring; lengthen it so it reads solid.
        "legend.handlelength": 3.0,
        "legend.handletextpad": 0.6,
    })
    return plt


def _title(key: tuple) -> str:
    return f"Facts, N = {key[1]:,}" if key[0] == "facts" else f"Text-to-SQL, {key[1]:,} examples"


def _rank_axis(ax, ranks) -> None:
    ranks = sorted({int(r) for r in ranks if r >= 1})
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    # Past 8 ticks the labels run together in a narrow panel ("128256512"). Keep every tick, label every other one.
    thin = len(ranks) > 8
    ax.set_xticklabels([str(r) if not thin or i % 2 == 0 else "" for i, r in enumerate(ranks)])
    ax.minorticks_off()
    ax.set_xlabel("Rank r")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _threshold(ax, y: float, label: str, in_legend: bool = False) -> None:
    """Solid hairline at y (dashed reads as a projection).

    in_legend=True names it in the legend instead of in the plot. Use that when data can sit near the
    line at any rank, as gap-closure curves can (SQL is near 0.9 at r=1). The in-plot label is only
    safe for monotone curves that stay below the line at low rank, like cumulative spectrum energy.
    """
    import matplotlib.transforms as mtransforms

    ax.axhline(y, color=MUTED, linewidth=0.8, zorder=1, label=label if in_legend else None)
    if not in_legend:
        # Just below the line: above it, the label is squeezed against the 1.0 gridline.
        ax.annotate(label, xy=(0.02, y), xycoords=mtransforms.blended_transform_factory(ax.transAxes, ax.transData),
                    xytext=(0, -3), textcoords="offset points", color=INK_2, fontsize=8, va="top", ha="left")


def _line(ax, df: pd.DataFrame, color: str, label: str, band: bool = True) -> None:
    df = df[df["rank"] >= 1].sort_values("rank")
    if df.empty:
        return
    if band:
        ax.fill_between(df["rank"], df.G_lo, df.G_hi, color=color, alpha=0.10, linewidth=0)
    ax.plot(df["rank"], df.G, color=color, linewidth=1.5, marker="o", markersize=5.5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, solid_joinstyle="round", solid_capstyle="round",
            label=label, zorder=3)


def _figure_legend(fig, axes) -> None:
    """One legend above the panels: entries deduplicated across panels, reference lines last."""
    entries: dict[str, object] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(label, handle)
    labels = sorted(entries, key=lambda l: l.startswith("95%"))
    fig.legend([entries[l] for l in labels], labels, loc="upper left", ncol=len(labels), bbox_to_anchor=(0.0, 1.08))


def _g_limits(df: pd.DataFrame) -> tuple[float, float]:
    lo = np.nanmin(df[["G", "G_lo"]].to_numpy()) if len(df) else 0.0
    hi = np.nanmax(df[["G", "G_hi"]].to_numpy()) if len(df) else 1.0
    return min(-0.05, lo - 0.05), max(1.1, hi + 0.05)


def plot_gap_closure(curve_df: pd.DataFrame, path: str | Path) -> None:
    """H1 + H2(b): LoRA and the oracle, one panel per setting."""
    plt = _plt()
    keys = _ordered_settings(curve_df)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.3 * len(keys), 3.3), sharey=True, squeeze=False)
    for ax, key in zip(axes[0], keys):
        df = _in_setting(curve_df, key)
        for arm in ("lora", "oracle"):
            _line(ax, df[df.arm == arm], ARM_COLOR[arm], ARM_LABEL[arm])
        _threshold(ax, THRESHOLD, "95% of full FT", in_legend=True)
        _rank_axis(ax, df["rank"])
        ax.set_title(_title(key))
    axes[0][0].set_ylabel("Gap closure (share of full FT's gain)")
    axes[0][0].set_ylim(*_g_limits(curve_df[curve_df.arm.isin(["lora", "oracle"])]))
    _figure_legend(fig, axes[0])
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(curve_df: pd.DataFrame, path: str | Path) -> bool:
    """Emphasis form: each ablated arm in its own color, standard LoRA as gray context."""
    panels = [(key, arm) for key in _ordered_settings(curve_df) for arm in ("rslora", "qlora")
              if not _in_setting(curve_df, key)[lambda d: d.arm == arm].empty]
    if not panels:
        return False
    plt = _plt()
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 3.3), sharey=True, squeeze=False)
    for ax, (key, arm) in zip(axes[0], panels):
        df = _in_setting(curve_df, key)
        _line(ax, df[df.arm == "lora"], CONTEXT_GRAY, ARM_LABEL["lora"], band=False)
        _line(ax, df[df.arm == arm], ARM_COLOR[arm], ARM_LABEL[arm])
        _threshold(ax, THRESHOLD, "95% of full FT", in_legend=True)
        _rank_axis(ax, df[df.arm.isin(["lora", arm])]["rank"])
        ax.set_title(f"{ARM_LABEL[arm].split(' ')[0]} · {_title(key)}")
    axes[0][0].set_ylabel("Gap closure (share of full FT's gain)")
    axes[0][0].set_ylim(*_g_limits(curve_df))
    # One legend above all panels: a per-panel legend inside the plot covered the QLoRA line.
    # Entities keep one color across panels, so a shared legend is valid.
    _figure_legend(fig, axes[0])
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_spectrum(energy: dict[tuple, dict], path: str | Path) -> bool:
    """H2(a): share of ‖ΔW‖² captured by the top-r directions. Facts sizes form an ordinal ramp."""
    if not energy:
        return False
    plt = _plt()
    groups = {task: sorted(k for k in energy if k[0] == task) for task in ("facts", "sql")}
    groups = {t: ks for t, ks in groups.items() if ks}
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.3), sharey=True, squeeze=False)
    for ax, (task, keys) in zip(axes[0], groups.items()):
        ramp = {1: [ORDINAL_BLUE[1]], 2: [ORDINAL_BLUE[0], ORDINAL_BLUE[2]]}.get(len(keys), ORDINAL_BLUE)
        all_ranks = []
        for color, key in zip(ramp, keys):
            captured = energy[key]["captured"]
            ranks = [r for r in sorted(captured) if r >= 1]
            all_ranks += ranks
            label = f"N = {key[1]:,}" if task == "facts" else f"{key[1]:,} examples"
            ax.plot(ranks, [captured[r] for r in ranks], color=color, linewidth=1.5, marker="o", markersize=5.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        _threshold(ax, 0.9, "90% of ‖ΔW‖²")
        _rank_axis(ax, all_ranks)
        ax.set_title("Facts: full-FT update spectrum" if task == "facts" else "Text-to-SQL: full-FT update spectrum")
        if len(keys) >= 2:
            ax.legend(loc="lower right", fontsize=8)
    axes[0][0].set_ylabel("Share of ‖ΔW‖² in top-r directions")
    axes[0][0].set_ylim(0, 1.05)
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------- entry point

def analyze(rows: list[dict], out_dir: str | Path, fig_dir: str | Path) -> dict:
    out_dir, fig_dir = Path(out_dir), Path(fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    train, oracle, spectrum = load(rows)
    refs = references(train)
    curve_df = curves(train, oracle, refs)
    if curve_df.empty:
        return {"curves": curve_df, "r_star": pd.DataFrame(), "table": "", "figures": []}
    rstar_df = r_star(curve_df)
    energy = energy_by_setting(spectrum, train, refs)
    table = rstar_table(curve_df, rstar_df, energy)
    curve_df.to_csv(out_dir / "curves.csv", index=False)
    rstar_df.to_csv(out_dir / "r_star.csv", index=False)
    (out_dir / "r_star.md").write_text(table)
    figures = [fig_dir / "gap_closure.png"]
    plot_gap_closure(curve_df, figures[0])
    if plot_ablation(curve_df, fig_dir / "ablation.png"):
        figures.append(fig_dir / "ablation.png")
    if plot_spectrum(energy, fig_dir / "spectrum.png"):
        figures.append(fig_dir / "spectrum.png")
    return {"curves": curve_df, "r_star": rstar_df, "table": table, "figures": figures, "energy": energy}
