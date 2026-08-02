"""Submit a grid to Kaggle as a batch GPU kernel, check on it, and fetch its output.

    python scripts/kaggle_launch.py push lr_cal      # render + submit: 2xT4, internet on, private kernel
    python scripts/kaggle_launch.py status lr_cal
    python scripts/kaggle_launch.py watch lr_cal     # block until COMPLETE/ERROR; exit 0 only on COMPLETE
    python scripts/kaggle_launch.py pull lr_cal      # -> results/kaggle/lr_cal/
    python scripts/kaggle_launch.py push grid_facts -- --push-repo <hf-user>/lorathresh-ckpts

Needs Kaggle API credentials (~/.kaggle/kaggle.json). The kernel clones REPO at its default branch,
so push your commits before submitting. Arguments after `--` go to run_grid.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/nayyirahsan/lora-rank-threshold.git"
TEMPLATE = ROOT / "kaggle" / "kernel_template.py"


_CONFIG_VIEW_USER = re.compile(r"^-\s*username:\s*(\S+)\s*$", re.MULTILINE)


def _cli() -> str:
    return shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle"))


def username_from_config_view(text: str) -> str | None:
    """Parse `kaggle config view` output ("- username: <name>"). Unset values print as None."""
    match = _CONFIG_VIEW_USER.search(text)
    return match.group(1) if match and match.group(1).lower() != "none" else None


def kaggle_username(explicit: str | None = None, home: Path | None = None) -> str:
    """The kernel id needs a username, but Kaggle CLI 2.x's recommended auth (OAuth login, or a bare
    access token) doesn't hand one over directly. Sources, in order: --user, KAGGLE_USERNAME,
    OAuth's ~/.kaggle/credentials.json, a legacy kaggle.json, then `kaggle config view`.
    Only the username field is ever read from credential files."""
    if explicit:
        return explicit
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    home = home or Path.home()
    for path in (home / ".kaggle" / "credentials.json", home / ".kaggle" / "kaggle.json",
                 home / ".config" / "kaggle" / "kaggle.json"):
        if path.exists():
            try:
                name = json.loads(path.read_text()).get("username")
            except (OSError, json.JSONDecodeError):
                name = None
            if name:
                return name
    try:
        out = subprocess.run([_cli(), "config", "view"], capture_output=True, text=True, timeout=60).stdout
        name = username_from_config_view(out)
        if name:
            return name
    except (OSError, subprocess.TimeoutExpired):
        pass
    raise SystemExit("could not determine your Kaggle username: run `kaggle auth login`, or pass --user <name>")


def kernel_slug(grid: str) -> str:
    return "lora-rank-threshold-" + grid.replace("_", "-")


def render(grid: str, user: str, extra_args: list[str], out_dir: Path) -> Path:
    if not (ROOT / "configs" / f"{grid}.yaml").exists():
        raise SystemExit(f"unknown grid {grid!r}: no configs/{grid}.yaml")
    out_dir.mkdir(parents=True, exist_ok=True)
    code = (
        TEMPLATE.read_text()
        .replace("__REPO__", REPO)
        .replace("__GRID__", grid)
        .replace("__EXTRA_ARGS__", json.dumps(extra_args))  # a JSON list of strings is a valid Python literal
    )
    (out_dir / "kernel.py").write_text(code)
    metadata = {
        "id": f"{user}/{kernel_slug(grid)}",
        "title": kernel_slug(grid),  # Kaggle requires the title to slugify to the id
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return out_dir


def kaggle(*args: str) -> int:
    return subprocess.run([_cli(), *args], check=False).returncode


_STATUS = re.compile(r"KernelWorkerStatus\.([A-Z_]+)")
TERMINAL_STATES = {"COMPLETE", "ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED", "CANCELLED", "CANCELED"}


def classify_status(text: str) -> tuple[str, str | None]:
    """Classify `kaggle kernels status` output as ("terminal" | "running" | "transient", state).

    Only Kaggle's own KernelWorkerStatus value counts. The first watcher used a plain
    grep for "error", and it declared a healthy running kernel finished when *this machine* briefly
    lost DNS: the CLI's NameResolutionError text contains "Error". Anything without a status value
    (network failures, auth hiccups) is "transient", so the watch keeps going."""
    match = _STATUS.search(text)
    if not match:
        return "transient", None
    state = match.group(1)
    return ("terminal" if state in TERMINAL_STATES else "running"), state


def watch(kernel_id: str, poll_seconds: float, max_hours: float) -> int:
    """Poll until a terminal state. Returns 0 on COMPLETE, 1 on any other terminal state or timeout."""
    import time

    start, last = time.time(), None
    while time.time() - start < max_hours * 3600:
        out = subprocess.run([_cli(), "kernels", "status", kernel_id], capture_output=True, text=True)
        kind, state = classify_status(out.stdout + out.stderr)
        minutes = (time.time() - start) / 60
        if kind == "transient":
            print(f"{minutes:.0f} min: status check failed (will retry): {(out.stdout + out.stderr).strip()[:120]}", flush=True)
        elif state != last:
            print(f"{minutes:.0f} min: {state}", flush=True)
            last = state
        if kind == "terminal":
            return 0 if state == "COMPLETE" else 1
        time.sleep(poll_seconds)
    print(f"gave up after {max_hours} h; last state {last}", flush=True)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "status", "watch", "pull"])
    ap.add_argument("grid")
    ap.add_argument("--dest", default=None, help="pull destination (default results/kaggle/<grid>)")
    ap.add_argument("--poll-seconds", type=float, default=300.0, help="watch: seconds between status checks")
    ap.add_argument("--max-hours", type=float, default=13.0, help="watch: give up after this long (Kaggle caps runs at 12 h)")
    ap.add_argument("--user", default=None, help="Kaggle username, if it can't be found automatically")
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1 :] if "--" in argv else []
    args = ap.parse_args(argv[: argv.index("--")] if "--" in argv else argv)

    user = kaggle_username(args.user)
    kernel_id = f"{user}/{kernel_slug(args.grid)}"
    if args.action == "push":
        out = render(args.grid, user, extra, ROOT / "build" / "kaggle" / args.grid)
        print(f"submitting {kernel_id} from {out}")
        return kaggle("kernels", "push", "-p", str(out))
    if args.action == "status":
        return kaggle("kernels", "status", kernel_id)
    if args.action == "watch":
        return watch(kernel_id, args.poll_seconds, args.max_hours)
    dest = Path(args.dest or ROOT / "results" / "kaggle" / args.grid)
    dest.mkdir(parents=True, exist_ok=True)
    return kaggle("kernels", "output", kernel_id, "-p", str(dest))


if __name__ == "__main__":
    sys.exit(main())
