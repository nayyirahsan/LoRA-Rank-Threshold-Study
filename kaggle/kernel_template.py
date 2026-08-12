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
# Kaggle's image ships torchao 0.10. peft 0.20 raises ImportError on *every* LoRA injection when an older torchao
# is importable. In the first full lr_cal run, both shards died on their first LoRA config, one of them
# after 94 min of full-FT runs. Nothing in this project uses torchao.
sh("pip uninstall -y -q torchao", check=False)
sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv")

gate = (
    "python -m lorathresh.train --task sql --n 2000 --method full --lr 3e-5 --epochs 1 --n-eval 100 "
    "--max-steps 30 --registry /tmp/gate.jsonl --output-dir /tmp/gate_ckpt --force"
)
micro: list[str] = []
if sh(f"CUDA_VISIBLE_DEVICES=0 {gate}", check=False) != 0:
    # Cap tokens, not rows, per micro-batch. lr_cal's fixed 8 rows (12.28GB peak on SQL) also split
    # 22-token facts batches that fit whole, slowing every facts run. 1024 tokens is below the ~1200
    # that 8 SQL rows needed; facts batches (32 x ~27 tokens) stay whole.
    print("gate: full FT failed at full batch size; retrying with --max-micro-tokens 1024", flush=True)
    if sh(f"CUDA_VISIBLE_DEVICES=0 {gate} --max-micro-tokens 1024", check=False) != 0:
        sys.exit("gate: full FT fails even with micro-batching; see the log above")
    micro = ["--max-micro-tokens", "1024"]
# Also gate the LoRA code path. The full-FT gate alone missed the torchao/peft incompatibility,
# which only fails when an adapter is injected.
lora_gate = gate.replace("--method full --lr 3e-5", "--method lora --rank 64 --lr 3e-4")
if sh(f"CUDA_VISIBLE_DEVICES=0 {lora_gate} {' '.join(micro)}", check=False) != 0:
    sys.exit("gate: LoRA r=64 failed; see the log above")
shutil.rmtree("/tmp/gate_ckpt", ignore_errors=True)
(WORK / "results").mkdir(parents=True, exist_ok=True)
shutil.copy("/tmp/gate.jsonl", WORK / "results" / "gate.jsonl")

# Resume: a registry committed to the repo from an earlier kernel run marks those configs done,
# so a re-submission doesn't repeat hours of finished GPU work.
seed = SRC / "results" / "kaggle" / GRID / "results" / "runs.jsonl"
if seed.exists():
    shutil.copy(seed, WORK / "results" / "runs.jsonl")
    print(f"resuming: {sum(1 for line in open(seed) if line.strip())} finished runs from {seed.relative_to(SRC)}", flush=True)
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
