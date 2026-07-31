"""Kaggle batch kernel. Rendered and submitted by scripts/kaggle_launch.py; don't edit the rendered copy.

1. Clone the public repo at its default branch and install it.
2. Memory gate: a short full-FT run on SQL, the most memory-hungry config (longest sequences,
   152k-vocab logits). If it fails, retry with micro-batching and use that for the whole grid.
3. Run the grid on every GPU and wait for all shards (scripts/run_parallel.py).
4. Write the registry, shard logs, and aggregate tables/figures to /kaggle/working. Kaggle keeps
   that folder as the kernel's output; checkpoints go to /tmp so the output stays small.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "__REPO__"
GRID = "__GRID__"
EXTRA_ARGS = __EXTRA_ARGS__
WORK = Path("/kaggle/working")
SRC = Path("/tmp/repo")


def sh(cmd: str, check: bool = True) -> int:
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check).returncode


def preflight() -> None:
    """Fail in seconds with a diagnosis, not a clone traceback. Kaggle can accept a kernel that
    requested internet and GPU and still run it without them: both need a phone-verified account.
    The first submission of this kernel died on `git clone` with "Could not resolve host"."""
    import socket

    problems = []
    try:
        socket.create_connection(("github.com", 443), timeout=10).close()
    except OSError as e:
        problems.append(
            f"no internet ({e}). The job needs it for the repo, pip, the Qwen weights, and the SQL data. "
            "Phone-verify the Kaggle account (kaggle.com/settings), then re-push."
        )
    gpus = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True)
    n_gpus = len([line for line in gpus.stdout.splitlines() if line.startswith("GPU ")]) if gpus.returncode == 0 else 0
    if n_gpus == 0:
        problems.append("no GPU attached. Phone-verify the account and make sure the kernel's accelerator is GPU T4 x2.")
    elif n_gpus < 2:
        print(f"preflight: warning, only {n_gpus} GPU visible; the grid will run on one shard (about twice as slow)", flush=True)
    if problems:
        sys.exit("preflight failed:\n- " + "\n- ".join(problems))
    print(f"preflight: internet OK, {n_gpus} GPU(s)", flush=True)


preflight()
sh(f"git clone --depth 1 {REPO} {SRC}")
os.chdir(SRC)
sh("git log --oneline -1")
sh("pip install -q -e '.[gpu]'")
sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv")

gate = (
    "python -m lorathresh.train --task sql --n 2000 --method full --lr 3e-5 --epochs 1 --n-eval 100 "
    "--max-steps 30 --registry /tmp/gate.jsonl --output-dir /tmp/gate_ckpt --force"
)
micro: list[str] = []
if sh(f"CUDA_VISIBLE_DEVICES=0 {gate}", check=False) != 0:
    print("gate: full FT failed at full batch size; retrying with --micro-batch-size 8", flush=True)
    if sh(f"CUDA_VISIBLE_DEVICES=0 {gate} --micro-batch-size 8", check=False) != 0:
        sys.exit("gate: full FT fails even with micro-batching; see the log above")
    micro = ["--micro-batch-size", "8"]
shutil.rmtree("/tmp/gate_ckpt", ignore_errors=True)
(WORK / "results").mkdir(parents=True, exist_ok=True)
shutil.copy("/tmp/gate.jsonl", WORK / "results" / "gate.jsonl")
print(json.dumps({"grid": GRID, "micro_batch": micro, "extra_args": EXTRA_ARGS}), flush=True)

code = subprocess.run([
    sys.executable, "scripts/run_parallel.py", f"configs/{GRID}.yaml",
    "--output-dir", "/tmp/ckpt", "--log-dir", str(WORK / "logs"),
    "--registry", str(WORK / "results" / "runs.jsonl"), *micro, *EXTRA_ARGS,
]).returncode
sh(
    f"python scripts/aggregate.py --registry {WORK}/results/runs.jsonl --out {WORK}/results --figures {WORK}/figures",
    check=False,
)
sys.exit(code)
