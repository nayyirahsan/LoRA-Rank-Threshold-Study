from lorathresh.registry import Registry, run_id


def test_run_id_is_order_independent():
    assert run_id({"a": 1, "b": 2}) == run_id({"b": 2, "a": 1})
    assert run_id({"a": 1}) != run_id({"a": 2})


def test_merge_file_adds_only_unknown_runs(tmp_path):
    # Two Kaggle sessions (or a local file and the Hub mirror) that each finished some runs.
    local, remote = Registry(tmp_path / "local.jsonl"), Registry(tmp_path / "remote.jsonl")
    shared = {"task": "sql", "rank": 4}
    local.record(shared, {"exec_acc": 0.5})
    remote.record(shared, {"exec_acc": 0.5})
    remote.record({"task": "sql", "rank": 8}, {"exec_acc": 0.6})
    assert local.merge_file(remote.path) == 1
    assert local.merge_file(remote.path) == 0
    assert sorted(r["config"]["rank"] for r in local.rows()) == [4, 8]


def test_duplicate_lines_count_once(tmp_path):
    # Two shards merging the same Hub rows at the same moment append them twice.
    reg = Registry(tmp_path / "runs.jsonl")
    reg.record({"seed": 0}, {"acc": 0.5})
    line = reg.path.read_text()
    reg.path.write_text(line + line)
    assert len(reg.rows()) == 1


def test_merge_into_missing_file(tmp_path):
    remote = Registry(tmp_path / "remote.jsonl")
    remote.record({"a": 1}, {})
    fresh = Registry(tmp_path / "new" / "runs.jsonl")
    assert fresh.merge_file(remote.path) == 1 and fresh.is_done({"a": 1})


def test_resume_skips_completed(tmp_path):
    reg = Registry(tmp_path / "runs.jsonl")
    cfg = {"task": "sql", "method": "lora", "rank": 8, "seed": 0}
    assert not reg.is_done(cfg)
    reg.record(cfg, {"exec_acc": 0.5})
    assert reg.is_done(cfg)
    assert not reg.is_done({**cfg, "seed": 1})
    assert Registry(tmp_path / "runs.jsonl").rows()[0]["metrics"] == {"exec_acc": 0.5}
