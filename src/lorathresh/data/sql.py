"""Text-to-SQL task (b-mc2/sql-create-context): the skill/format side of H1.

Examples whose gold query can't be scored are dropped: gold fails to execute, or returns
nothing on every random DB. Examples are constant comparisons from numeric column names
(`WHERE 2008 = "153"`) and malformed gold (`score = 71 - 68 - 76 = 215`). In the matcher
audit these caused nearly all remaining false positives, so scoring them would add noise.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

from lorathresh.sqlexec import exact_match, execution_match

DATASET_ID = "b-mc2/sql-create-context"


@dataclass(frozen=True)
class SqlExample:
    context: str
    question: str
    answer: str


def format_prompt(context: str, question: str) -> str:
    return f"Schema: {context}\nQuestion: {question}\nSQL:"


def load_raw(path: str | None = None) -> list[SqlExample]:
    if path:
        with open(path) as f:
            rows = json.load(f)
    else:
        from datasets import load_dataset

        rows = load_dataset(DATASET_ID, split="train")
    return [SqlExample(r["context"], r["question"], r["answer"]) for r in rows]


def is_scorable(ex: SqlExample) -> bool:
    r = execution_match(ex.answer, ex.answer, ex.context)
    return r.gold_ok and r.informative


def make_splits(
    examples: list[SqlExample], n_train: int = 10_000, n_eval: int = 1_000, seed: int = 0
) -> tuple[list[SqlExample], list[SqlExample], dict]:
    """Deterministic, disjoint train/eval splits of scorable, deduplicated examples.

    Takes examples in a seeded shuffle order until both splits are full, so the whole
    dataset doesn't have to be filtered. Eval is filled first so it never depends on n_train.
    """
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    seen: set[tuple[str, str]] = set()
    eval_split: list[SqlExample] = []
    train: list[SqlExample] = []
    stats = {"duplicates": 0, "unscorable": 0, "considered": 0}
    for i in order:
        if len(eval_split) == n_eval and len(train) == n_train:
            break
        ex = examples[i]
        stats["considered"] += 1
        key = (ex.context, ex.question)
        if key in seen:
            stats["duplicates"] += 1
            continue
        seen.add(key)
        if not is_scorable(ex):
            stats["unscorable"] += 1
            continue
        (eval_split if len(eval_split) < n_eval else train).append(ex)
    stats["unscorable_rate"] = stats["unscorable"] / max(1, stats["considered"] - stats["duplicates"])
    return train, eval_split, stats


def score(prediction: str, ex: SqlExample) -> dict[str, bool]:
    pred = prediction.strip().split("\n", 1)[0].strip()
    return {
        "exec_match": execution_match(pred, ex.answer, ex.context).match,
        "exact_match": exact_match(pred, ex.answer),
    }
