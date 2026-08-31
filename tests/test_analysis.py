import math
from dataclasses import asdict

import numpy as np
import pytest

from lorathresh import analysis as A
from lorathresh.registry import run_id
from lorathresh.train import RunConfig


def _train_row(score: float, **kw) -> dict:
    cfg = asdict(RunConfig(**{"task": "facts", "n": 1000, "epochs": 10.0, "n_eval": 1000, **kw}))
    return {"run_id": run_id(cfg), "config": cfg, "metrics": {"acc": score}}


def _world(ft_scores=(0.9, 0.9), lora=None, ft_lr=3e-5) -> list[dict]:
    lora = lora or {1: [0.3, 0.3], 4: [0.6, 0.6], 16: [0.88, 0.89], 64: [0.9, 0.9]}
    rows = [_train_row(0.1, method="base")]
    rows += [_train_row(s, method="full", lr=ft_lr, seed=i) for i, s in enumerate(ft_scores)]
    for rank, scores in lora.items():
        rows += [_train_row(s, method="lora", rank=rank, lr=3e-4, seed=i) for i, s in enumerate(scores)]
    return rows


def _oracle_rows(ft_row: dict, scores: dict[int, float]) -> list[dict]:
    rows = []
    for rank, s in scores.items():
        key = {"kind": "oracle", "ft_run_id": ft_row["run_id"], "rank": rank, "ft_config": ft_row["config"]}
        rows.append({"run_id": run_id(key), "config": key, "metrics": {"acc": s}})
    return rows


def _analyze(rows):
    train, oracle, spectrum = A.load(rows)
    refs = A.references(train)
    curves = A.curves(train, oracle, refs)
    return train, refs, curves, A.r_star(curves), spectrum


def test_gap_closure_and_r_star():
    _, refs, curves, rstar, _ = _analyze(_world())
    lora = curves[curves.arm == "lora"].set_index("rank")
    assert lora.loc[1, "G"] == pytest.approx(0.25)
    assert lora.loc[16, "G"] == pytest.approx((0.885 - 0.1) / 0.8)
    row = rstar[rstar.arm == "lora"].iloc[0]
    assert row.r_star == 16 and row.max_rank_tested == 64 and row.min_rank_tested == 1


def test_r_star_not_reached_is_none():
    _, _, _, rstar, _ = _analyze(_world(lora={1: [0.2, 0.2], 4: [0.5, 0.5]}))
    assert rstar[rstar.arm == "lora"].iloc[0].r_star is None


def test_best_lr_is_selected_per_rank():
    rows = _world() + [_train_row(0.2, method="lora", rank=4, lr=1e-3, seed=0)]
    _, _, curves, _, _ = _analyze(rows)
    r4 = curves[(curves.arm == "lora") & (curves["rank"] == 4)].iloc[0]
    assert r4.lr == pytest.approx(3e-4) and r4.score_mean == pytest.approx(0.6)


def test_gap_closure_undefined_when_full_ft_does_not_improve():
    assert math.isnan(A.gap_closure(0.5, base=0.10, ft_mean=0.11))
    _, _, curves, _, _ = _analyze(_world(ft_scores=(0.105, 0.11)))
    assert curves.G.isna().all()


def test_bootstrap_ci_brackets_point_and_collapses_without_variance():
    g, lo, hi = A.bootstrap_ci([0.5, 0.7, 0.6], [0.9, 0.8, 0.85], base=0.1)
    assert lo < g < hi
    g, lo, hi = A.bootstrap_ci([0.6, 0.6], [0.9, 0.9], base=0.1)
    assert lo == pytest.approx(g) and hi == pytest.approx(g)


def test_settings_with_different_eval_size_do_not_mix():
    cal = [_train_row(0.1, method="base", n_eval=500), _train_row(0.5, method="full", lr=1e-5, n_eval=500)]
    _, refs, _, _, _ = _analyze(_world() + cal)
    assert len(refs) == 2
    grid_key = ("facts", 1000, 10.0, 1000, -1)
    assert refs[grid_key]["ft_scores"].tolist() == [0.9, 0.9]


def test_oracle_curve_uses_reference_ft_runs_only():
    rows = _world()
    ft_rows = [r for r in rows if r["config"]["method"] == "full"]
    stale_ft = _train_row(0.4, method="full", lr=1e-4, seed=0)  # worse LR: not the reference
    rows += [stale_ft] + _oracle_rows(ft_rows[0], {0: 0.1, 4: 0.2, 64: 0.88}) + _oracle_rows(stale_ft, {4: 0.9})
    _, _, curves, rstar, _ = _analyze(rows)
    orc = curves[curves.arm == "oracle"].set_index("rank")
    assert orc.loc[4, "score_mean"] == pytest.approx(0.2)
    assert rstar[rstar.arm == "oracle"].iloc[0].r_star == 64


def test_energy_prediction_and_table():
    rows = _world()
    ft = next(r for r in rows if r["config"]["method"] == "full")
    spec_key = {"kind": "spectrum", "ft_run_id": ft["run_id"]}
    rows.append({"run_id": run_id(spec_key), "config": spec_key, "metrics": {
        "energy_rank_90_median": 24.0, "energy_captured_weighted": {"1": 0.2, "16": 0.85, "64": 0.97}}})
    train, refs, curves, rstar, spectrum = _analyze(rows)
    energy = A.energy_by_setting(spectrum, train, refs)
    (e,) = energy.values()
    assert e["energy_rank_90"] == 24.0 and e["captured"][16] == pytest.approx(0.85)
    table = A.rstar_table(curves, rstar, energy)
    assert "| facts | 1,000 |" in table and "1.50× (within 2×)" in table


def test_analyze_writes_outputs_and_figures(tmp_path):
    rows = _world()
    ft = next(r for r in rows if r["config"]["method"] == "full")
    rows += _oracle_rows(ft, {1: 0.1, 4: 0.3, 16: 0.7, 64: 0.9})
    rows += [_train_row(s, method="lora", scaling="rslora", rank=r, lr=3e-4) for r, s in ((4, 0.7), (64, 0.9))]
    result = A.analyze(rows, tmp_path / "out", tmp_path / "fig")
    assert (tmp_path / "out" / "curves.csv").exists() and (tmp_path / "out" / "r_star.md").exists()
    assert [p.name for p in result["figures"]] == ["gap_closure.png", "ablation.png"]
    assert all(p.stat().st_size > 10_000 for p in result["figures"])


def test_empty_registry():
    result = A.analyze([], "/tmp/unused-out", "/tmp/unused-fig")
    assert result["curves"].empty


def test_load_handles_numpy_free_rows():
    train, oracle, spectrum = A.load([])
    assert list(train.columns) == A.TRAIN_COLS and oracle.empty and spectrum == []
    assert np.isnan(A.gap_closure(0.0, 0.0, 0.0))
