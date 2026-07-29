import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("kaggle_launch", ROOT / "scripts" / "kaggle_launch.py")
kaggle_launch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kaggle_launch)


def test_render_produces_valid_kernel_and_metadata(tmp_path):
    out = kaggle_launch.render("lr_cal", "someone", ["--push-repo", "me/ckpts"], tmp_path)
    code = (out / "kernel.py").read_text()
    compile(code, "kernel.py", "exec")  # placeholders replaced with valid Python
    assert "__GRID__" not in code and "__REPO__" not in code and "__EXTRA_ARGS__" not in code
    assert 'GRID = "lr_cal"' in code
    assert 'EXTRA_ARGS = ["--push-repo", "me/ckpts"]' in code
    assert kaggle_launch.REPO in code

    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["id"] == "someone/lora-rank-threshold-lr-cal"
    assert meta["title"] == "lora-rank-threshold-lr-cal"
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
    assert meta["code_file"] == "kernel.py" and meta["kernel_type"] == "script"


def test_render_without_extra_args(tmp_path):
    code = (kaggle_launch.render("grid_sql", "u", [], tmp_path) / "kernel.py").read_text()
    assert "EXTRA_ARGS = []" in code


def test_unknown_grid_rejected(tmp_path):
    with pytest.raises(SystemExit):
        kaggle_launch.render("nope", "u", [], tmp_path)


@pytest.fixture
def no_env_user(monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.setattr(kaggle_launch, "_cli", lambda: "/nonexistent/kaggle")  # never call a real CLI


def _write(home, rel, data):
    path = home / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_username_from_oauth_credentials(tmp_path, no_env_user):
    _write(tmp_path, ".kaggle/credentials.json", {"refresh_token": "x", "username": "oauth-user"})
    assert kaggle_launch.kaggle_username(home=tmp_path) == "oauth-user"


def test_username_from_legacy_kaggle_json(tmp_path, no_env_user):
    _write(tmp_path, ".kaggle/kaggle.json", {"username": "legacy-user", "key": "x"})
    assert kaggle_launch.kaggle_username(home=tmp_path) == "legacy-user"


def test_username_precedence(tmp_path, no_env_user, monkeypatch):
    _write(tmp_path, ".kaggle/credentials.json", {"username": "file-user"})
    monkeypatch.setenv("KAGGLE_USERNAME", "env-user")
    assert kaggle_launch.kaggle_username(home=tmp_path) == "env-user"
    assert kaggle_launch.kaggle_username("flag-user", home=tmp_path) == "flag-user"


def test_empty_username_falls_through_and_missing_everything_exits(tmp_path, no_env_user):
    _write(tmp_path, ".kaggle/credentials.json", {"username": ""})
    with pytest.raises(SystemExit, match="kaggle auth login"):
        kaggle_launch.kaggle_username(home=tmp_path)


def test_username_from_config_view_output():
    text = "Configuration values from /Users/x/.kaggle\n- username: cli-user\n- auth_method: oauth\n- path: None\n"
    assert kaggle_launch.username_from_config_view(text) == "cli-user"
    assert kaggle_launch.username_from_config_view("- username: None\n") is None
    assert kaggle_launch.username_from_config_view("Authentication required") is None


def test_every_config_has_a_distinct_kernel_slug():
    slugs = {kaggle_launch.kernel_slug(p.stem) for p in (ROOT / "configs").glob("*.yaml")}
    assert len(slugs) == len(list((ROOT / "configs").glob("*.yaml")))
