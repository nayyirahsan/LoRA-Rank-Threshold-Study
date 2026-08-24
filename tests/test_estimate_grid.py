import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("estimate_grid", ROOT / "scripts" / "estimate_grid.py")
eg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eg)

from lorathresh.train import RunConfig  # noqa: E402


def test_training_seconds_match_hand_calculation():
    cfg = RunConfig(task="facts", n=1000, method="full", lr=1e-5, epochs=10, n_eval=500)
    # 1000 facts * 4 templates * 21.4 tokens * 10 epochs = 856,000 tokens at 485 tok/s
    expected_train = 856_000 / 485
    got = eg.run_seconds(cfg, eg.DEFAULT_TPS)
    assert got == pytest.approx(expected_train + 500 * 0.032 + 90)
    # lr_cal measured 1766-1775 s of pure training for exactly this config
    assert 1600 < expected_train < 1900


def test_base_runs_cost_only_eval_and_oracle_only_for_full_ft():
    base = RunConfig(task="sql", n=2000, method="base", n_eval=500)
    assert eg.run_seconds(base, eg.DEFAULT_TPS) == pytest.approx(500 * 0.1 + 90)
    assert eg.oracle_seconds(base) == 0.0
    full = RunConfig(task="sql", n=2000, method="full", lr=1e-5, epochs=2, n_eval=500)
    assert eg.oracle_seconds(full) == pytest.approx(11 * 500 * 0.1 + 120)


def test_wall_clock_bounded_below_by_longest_run():
    long = RunConfig(task="facts", n=16000, method="full", lr=1e-5, epochs=10, n_eval=1000)
    short = RunConfig(task="sql", n=100, method="base", n_eval=10)
    est = eg.estimate([long, short], eg.DEFAULT_TPS, gpus=2)
    assert est["wall_hours"] == pytest.approx(est["longest_run_hours"])


def test_tps_override_and_shipped_configs_estimate():
    tps = eg.parse_tps(["facts:lora=1600"])
    assert tps[("facts", "lora")] == 1600 and tps[("sql", "full")] == eg.DEFAULT_TPS[("sql", "full")]
    import yaml

    for name in ("grid_sql.yaml", "grid_facts.yaml", "lr_cal.yaml"):
        cfgs = eg.expand(yaml.safe_load((ROOT / "configs" / name).read_text()))
        assert eg.estimate(cfgs, eg.DEFAULT_TPS)["gpu_hours"] > 0
