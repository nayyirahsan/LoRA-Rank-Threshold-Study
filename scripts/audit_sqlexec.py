"""Audit the SQL execution matcher on real gold queries.

Applies realistic one-token errors to gold queries (a swapped literal, a flipped comparison,
a swapped aggregate) and measures how often the matcher wrongly accepts them. Reports the
rate over all examples and over the scorable ones kept by data/sql.py.

    python scripts/audit_sqlexec.py --data sql_create_context_v4.json --n 2000
"""
from __future__ import annotations

import argparse
import random
import re
import time
from collections import Counter

from lorathresh.data.sql import load_raw
from lorathresh.sqlexec import execution_match

_SWAPS = ((">", "<"), ("<", ">"), ("COUNT", "SUM"), ("MAX", "MIN"), ("MIN", "MAX"))


def perturb(sql: str) -> tuple[str | None, str | None]:
    m = re.search(r"\"[^\"]*\"|'[^']*'", sql)
    if m:
        return sql[: m.start()] + '"zzz_wrong"' + sql[m.end():], "literal"
    for a, b in _SWAPS:
        if a in sql:
            return sql.replace(a, b, 1), a
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="local JSON; omit to download from the HF Hub")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sample = random.Random(args.seed).sample(load_raw(args.data), args.n)
    start = time.time()
    gold_ok = scorable = 0
    tried: dict[str, Counter] = {"all": Counter(), "scorable": Counter()}
    accepted: dict[str, Counter] = {"all": Counter(), "scorable": Counter()}
    examples = []
    for ex in sample:
        r = execution_match(ex.answer, ex.answer, ex.context)
        gold_ok += r.gold_ok
        scorable += r.gold_ok and r.informative
        pred, kind = perturb(ex.answer)
        if pred is None or not r.gold_ok:
            continue
        wrongly_accepted = execution_match(pred, ex.answer, ex.context).match
        for subset in ("all", "scorable") if r.informative else ("all",):
            tried[subset][kind] += 1
            accepted[subset][kind] += wrongly_accepted
        if wrongly_accepted and r.informative:
            examples.append((ex.answer, pred))

    n = len(sample)
    print(f"examples: {n}   gold executes: {gold_ok / n:.3f}   scorable: {scorable / n:.3f}")
    for subset in ("all", "scorable"):
        t, a = sum(tried[subset].values()), sum(accepted[subset].values())
        by_kind = {k: f"{accepted[subset][k]}/{tried[subset][k]}" for k in tried[subset]}
        print(f"false positives ({subset}): {a}/{t} = {a / max(1, t):.3f}   {by_kind}")
    print(f"ms/example: {(time.time() - start) / n * 1000:.1f}")
    for gold, pred in examples[:10]:
        print(f"  GOLD: {gold}\n  PRED: {pred}")


if __name__ == "__main__":
    main()
