"""Run one grid on every visible GPU (one shard per GPU) and wait for all shards to finish.

    python scripts/run_parallel.py configs/lr_cal.yaml --output-dir /tmp/ckpt [any run_grid.py flags]

Backgrounding shards with `nohup ... &` only works in an interactive notebook session. A Kaggle
batch run ("Save & Run All" or `kaggle kernels push`) ends when the main process exits, and that
kills background shards. This script blocks until every shard is done, so batch runs work, and it
exits non-zero if any shard failed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--gpus", type=int, default=None, help="shards to run; default: number of visible CUDA devices")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    args, grid_args = ap.parse_known_args()  # everything else goes to run_grid.py

    n = args.gpus
    if n is None:
        import torch

        n = max(1, torch.cuda.device_count())
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_grid = Path(__file__).with_name("run_grid.py")

    shards = []
    for i in range(n):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(i), "PYTHONUNBUFFERED": "1"}
        log_path = log_dir / f"{Path(args.grid).stem}_shard{i}.log"
        log = log_path.open("w")
        cmd = [sys.executable, str(run_grid), args.grid, "--shard", f"{i}/{n}", *grid_args]
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        shards.append({"i": i, "proc": proc, "log": log, "path": log_path})
        print(f"shard {i}/{n}: pid {proc.pid}, log {log_path}", flush=True)

    start, failed = time.time(), []
    running = list(shards)
    while running:
        time.sleep(args.poll_seconds)
        for s in list(running):
            code = s["proc"].poll()
            if code is None:
                continue
            s["log"].close()
            running.remove(s)
            print(f"shard {s['i']} exited {code} after {(time.time() - start) / 60:.1f} min", flush=True)
            if code != 0:
                failed.append(s["i"])
                print(s["path"].read_text()[-2000:], flush=True)  # tail of the failing shard's log
        if running:
            print(f"{(time.time() - start) / 60:.1f} min: shards still running {[s['i'] for s in running]}", flush=True)
    if failed:
        print(f"FAILED shards: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
