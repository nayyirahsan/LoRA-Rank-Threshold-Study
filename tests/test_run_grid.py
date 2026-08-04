import importlib.util
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_grid", ROOT / "scripts" / "run_grid.py")
run_grid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_grid)


def _load(name: str):
    return run_grid.expand(yaml.safe_load((ROOT / "configs" / name).read_text()))


def test_resolve_lr_forms():
    assert run_grid.resolve_lr(3e-4, "lora", 4) == [3e-4]
    assert run_grid.resolve_lr([1e-4, 3e-4], "full", 0) == [1e-4, 3e-4]
    assert run_grid.resolve_lr({"full": 3e-5, "lora": {4: 1e-3}}, "lora", 4) == [1e-3]
    assert run_grid.resolve_lr({"full": 3e-5}, "base", 0) == [0.0]
    with pytest.raises(KeyError):
        run_grid.resolve_lr({"full": 3e-5}, "lora", 4)
    with pytest.raises(KeyError):
        run_grid.resolve_lr({"lora": {4: 1e-3}}, "lora", 64)


def test_grid_facts_counts_and_ordering():
    configs = _load("grid_facts.yaml")
    # per N: 1 base + 2 full seeds + 5 ranks x 2 seeds = 13; x 3 values of N
    assert len(configs) == 39
    for n in (1000, 4000, 16000):
        methods = [c.method for c in configs if c.n == n]
        assert methods.count("base") == 1 and methods.count("full") == 2 and methods.count("lora") == 10
        assert methods == sorted(methods, key=["base", "full", "lora", "qlora"].index)  # FT before LoRA


def test_every_shipped_config_expands():
    counts = {p.name: len(_load(p.name)) for p in (ROOT / "configs").glob("*.yaml")}
    assert counts["lr_cal.yaml"] == 18 and counts["grid_sql.yaml"] == 28 and counts["ablations.yaml"] == 9
    assert counts["smoke.yaml"] == 4


def _main(monkeypatch, tmp_path, fail_when, *extra):
    ran = []

    def fake_run(cfg, *args, **kwargs):
        ran.append(cfg.method)
        if fail_when(cfg):
            raise RuntimeError(f"boom: {cfg.method}")

    monkeypatch.setattr(run_grid, "run", fake_run)
    monkeypatch.setattr(run_grid.sys, "argv", ["run_grid.py", str(ROOT / "configs" / "smoke.yaml"),
                                               "--registry", str(tmp_path / "runs.jsonl"), *extra])
    return run_grid.main(), ran


def test_failed_config_does_not_stop_the_shard(monkeypatch, tmp_path):
    # smoke.yaml order: base, full, lora r=4, lora r=16. Full FT fails; LoRA runs must still happen.
    code, ran = _main(monkeypatch, tmp_path, lambda c: c.method == "full")
    assert ran == ["base", "full", "lora", "lora"]
    assert code == 1


def test_consecutive_failures_abort_the_shard(monkeypatch, tmp_path):
    code, ran = _main(monkeypatch, tmp_path, lambda c: c.method != "base", "--max-consecutive-failures", "2")
    assert ran == ["base", "full", "lora"]  # full and lora r=4 fail in a row -> abort before lora r=16
    assert code == 1


def test_clean_grid_exits_zero(monkeypatch, tmp_path):
    code, ran = _main(monkeypatch, tmp_path, lambda c: False)
    assert code == 0 and len(ran) == 4


def test_shards_partition_the_grid():
    configs = _load("grid_sql.yaml")
    shards = [configs[i::2] for i in range(2)]
    assert sorted(map(repr, shards[0] + shards[1])) == sorted(map(repr, configs))
