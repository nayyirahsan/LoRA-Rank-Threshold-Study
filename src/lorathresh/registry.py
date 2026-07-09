"""Append-only run registry so the grid can resume across Kaggle/Colab sessions.

A run is identified by a hash of its config, so re-launching a grid skips finished runs.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def run_id(config: dict) -> str:
    return hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


class Registry:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def rows(self) -> list[dict]:
        """Rows, first occurrence per run id.

        Duplicates can happen legitimately: two shards pulling the Hub mirror at the same moment
        both merge the same remote rows. Without dedup, analysis would count those seeds twice.
        """
        if not self.path.exists():
            return []
        seen: set[str] = set()
        out = []
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row["run_id"] not in seen:
                        seen.add(row["run_id"])
                        out.append(row)
        return out

    def is_done(self, config: dict) -> bool:
        rid = run_id(config)
        return any(r["run_id"] == rid for r in self.rows())

    def record(self, config: dict, metrics: dict) -> str:
        rid = run_id(config)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps({"run_id": rid, "time": time.time(), "config": config, "metrics": metrics}) + "\n")
        return rid

    def merge_file(self, other: str | Path) -> int:
        """Append rows from another registry file whose run ids aren't here yet. Returns the count added."""
        known = {r["run_id"] for r in self.rows()}
        added = []
        for row in Registry(other).rows():
            if row["run_id"] not in known:
                known.add(row["run_id"])
                added.append(row)
        if added:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.writelines(json.dumps(r) + "\n" for r in added)
        return len(added)

    # Kaggle/Colab sessions lose local files when they end, so the registry is mirrored to the
    # same private Hub repo as the checkpoints: pull-and-merge at startup, push after every recorded run.
    def _hub_filename(self) -> str:
        return f"registry/{self.path.name}"

    def pull_from_hub(self, repo_id: str) -> int:
        from huggingface_hub import hf_hub_download

        try:
            remote = hf_hub_download(repo_id, filename=self._hub_filename(), repo_type="model", force_download=True)
        except Exception as e:  # repo or file doesn't exist yet on the first session
            print(f"registry: nothing to pull from {repo_id} ({type(e).__name__})")
            return 0
        added = self.merge_file(remote)
        print(f"registry: merged {added} rows from {repo_id}")
        return added

    def push_to_hub(self, repo_id: str) -> None:
        """Best effort. The run is already recorded locally, and a failed mirror push (network, or two
        shards committing at once) must not crash a run that took an hour to train. The next push
        uploads the whole file, so nothing is lost."""
        from huggingface_hub import HfApi

        try:
            HfApi().upload_file(
                path_or_fileobj=str(self.path), path_in_repo=self._hub_filename(), repo_id=repo_id, repo_type="model"
            )
        except Exception as e:
            print(f"registry: push to {repo_id} failed ({type(e).__name__}: {e}); will retry on next record")
