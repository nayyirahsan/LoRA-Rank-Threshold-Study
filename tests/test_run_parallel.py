import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def _run(tmp_path, *extra):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_parallel.py"), *extra,
         "--gpus", "2", "--log-dir", str(tmp_path / "logs"), "--poll-seconds", "0.2",
         "--registry", str(tmp_path / "runs.jsonl")],
        capture_output=True, text=True, cwd=ROOT, timeout=300,
    )


def test_all_shards_run_and_are_awaited(tmp_path):
    result = _run(tmp_path, "configs/smoke.yaml", "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    logs = sorted((tmp_path / "logs").glob("smoke_shard*.log"))
    assert [p.name for p in logs] == ["smoke_shard0.log", "smoke_shard1.log"]
    # smoke.yaml has 4 configs, split 2/2; each shard must have run to completion before we returned
    assert all("2 configs in shard" in p.read_text() for p in logs)


def test_failing_shard_makes_exit_nonzero(tmp_path):
    result = _run(tmp_path, "configs/does_not_exist.yaml", "--dry-run")
    assert result.returncode == 1
    assert "FAILED shards: [0, 1]" in result.stdout
