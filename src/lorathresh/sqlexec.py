"""Execution-based matching for text-to-SQL (b-mc2/sql-create-context).

The dataset gives only CREATE TABLE statements, with no rows. On empty tables almost any
query "matches", so we fill several random SQLite databases and compare result sets.
Rows are seeded with the literals from the gold query. Otherwise WHERE clauses would match
nothing and hide wrong predictions (a false-positive source this module exists to prevent).
"""
from __future__ import annotations

import random
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass

_CREATE_RE = re.compile(r"CREATE\s+TABLE\s+(\w+)\s*\((.*?)\)", re.I | re.S)
_STR_LIT_RE = re.compile(r"'([^']*)'|\"([^\"]*)\"")
_NUM_LIT_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
_NUMERIC_TYPES = ("INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL")
_MAX_VM_STEPS = 2_000_000


@dataclass(frozen=True)
class MatchResult:
    match: bool
    gold_ok: bool       # gold executed on every DB (if False, the example is unscorable)
    informative: bool   # gold returned a non-empty result on at least one DB


def parse_schema(context: str) -> dict[str, list[tuple[str, str]]]:
    """Table -> columns. The dataset sometimes declares one table several times with
    different columns, so declarations are merged (names compared case-insensitively, as SQLite does)."""
    schema: dict[str, list[tuple[str, str]]] = {}
    canonical: dict[str, str] = {}
    for table, body in _CREATE_RE.findall(context):
        table = canonical.setdefault(table.lower(), table)
        cols = schema.setdefault(table, [])
        seen = {c.lower() for c, _ in cols}
        for part in body.split(","):
            tokens = part.split()
            if tokens and tokens[0].strip('"`').lower() not in seen:
                name = tokens[0].strip('"`')
                seen.add(name.lower())
                cols.append((name, tokens[1].upper() if len(tokens) > 1 else "VARCHAR"))
    return schema


def extract_literals(sql: str) -> tuple[list[str], list[float]]:
    strings = [a or b for a, b in _STR_LIT_RE.findall(sql)]
    without_strings = _STR_LIT_RE.sub(" ", sql)
    numbers = [float(n) for n in _NUM_LIT_RE.findall(without_strings)]
    return strings, numbers


_COND_RE = re.compile(
    r"(?:\w+\.)?\"?(\w+)\"?\s*(>=|<=|=|>|<|\bLIKE\b)\s*('[^']*'|\"[^\"]*\"|-?\d+(?:\.\d+)?)", re.I
)
_N_WITNESS_ROWS = 3


def parse_conditions(sql: str, schema) -> dict[str, list[tuple[int, str, str]]]:
    """table -> [(column index, op, literal)] for `column op literal` comparisons in the query.

    Only columns present in the schema count, so constant comparisons like `2008 = "153"` are skipped.
    """
    out: dict[str, list[tuple[int, str, str]]] = {}
    for col, op, lit in _COND_RE.findall(sql):
        for table, cols in schema.items():
            idx = next((i for i, (c, _) in enumerate(cols) if c.lower() == col.lower()), None)
            if idx is not None:
                out.setdefault(table, []).append((idx, op.upper(), lit.strip("'\"")))
    return out


def _is_numeric_type(ctype: str) -> bool:
    return any(t in ctype for t in _NUMERIC_TYPES)


def _satisfying_value(op: str, lit: str, numeric_col: bool):
    if numeric_col and _is_number(lit):
        v = float(lit)
        v = int(v) if v == int(v) else v
        return {">": v + 1, "<": v - 1}.get(op, v)
    # TEXT affinity: SQLite compares these as strings, so "10" > "9" is false. Build
    # values that satisfy the comparison as text.
    if op == "LIKE":
        return lit.replace("%", "").replace("_", "")
    if op == ">":
        return lit + "1"
    if op == "<":
        return lit[:-1]
    return lit


def _random_value(ctype: str, str_pool, num_pool, rng: random.Random):
    if _is_numeric_type(ctype):
        return rng.choice(num_pool) if num_pool and rng.random() < 0.5 else rng.randint(0, 20)
    return rng.choice(str_pool) if str_pool and rng.random() < 0.5 else _rand_str(rng)


def _random_db(schema, strings, numbers, conditions, rng: random.Random, n_rows: int) -> sqlite3.Connection:
    """Fill each table with random rows, planting witness and near-miss rows first.

    Witness rows satisfy every gold `column op literal` condition on their table, so gold is
    non-empty even with several AND'ed conditions. Near-miss rows break exactly one condition,
    so a prediction with one wrong literal or comparison returns a different result.
    """
    conn = sqlite3.connect(":memory:")
    steps = 0

    def budget() -> int:  # abort runaway queries (e.g. accidental cross joins)
        nonlocal steps
        steps += 1000
        return 1 if steps > _MAX_VM_STEPS else 0

    # Numbers that appear as quoted strings in this dataset ("2009") are mixed into both pools.
    str_pool = strings + [_fmt_num(n) for n in numbers]
    num_pool = numbers + [float(s) for s in strings if _is_number(s)]
    for table, cols in schema.items():
        col_defs = ", ".join(f'"{c}" {t}' for c, t in cols)
        conn.execute(f'CREATE TABLE "{table}" ({col_defs})')
        conds = conditions.get(table, [])
        planted: list[list] = []
        if conds:
            witness = lambda: [_random_value(t, str_pool, num_pool, rng) for _, t in cols]
            for _ in range(_N_WITNESS_ROWS):
                row = witness()
                for idx, op, lit in conds:
                    row[idx] = _satisfying_value(op, lit, _is_numeric_type(cols[idx][1]))
                planted.append(row)
            for broken in range(len(conds)):
                row = witness()
                for j, (idx, op, lit) in enumerate(conds):
                    if j != broken:
                        row[idx] = _satisfying_value(op, lit, _is_numeric_type(cols[idx][1]))
                planted.append(row)
        rows = planted + [
            [_random_value(t, str_pool, num_pool, rng) for _, t in cols] for _ in range(max(0, n_rows - len(planted)))
        ]
        for row in rows:
            conn.execute(f'INSERT INTO "{table}" VALUES ({",".join("?" * len(row))})', row)
    conn.set_progress_handler(budget, 1000)
    return conn


def _fmt_num(n: float) -> str:
    return str(int(n)) if n == int(n) else str(n)


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _rand_str(rng: random.Random) -> str:
    return "".join(rng.choice("abcdefghij") for _ in range(4))


def _execute(conn: sqlite3.Connection, sql: str):
    try:
        return conn.execute(sql).fetchall()
    except (sqlite3.Error, sqlite3.Warning, ValueError):
        return None


def _same(gold_rows, pred_rows, ordered: bool) -> bool:
    if pred_rows is None:
        return False
    if ordered:
        return gold_rows == pred_rows
    return Counter(map(repr, gold_rows)) == Counter(map(repr, pred_rows))


def execution_match(pred: str, gold: str, context: str, n_dbs: int = 3, n_rows: int = 20, seed: int = 0) -> MatchResult:
    schema = parse_schema(context)
    strings, numbers = extract_literals(gold)
    conditions = parse_conditions(gold, schema)
    ordered = "order by" in gold.lower()
    match, gold_ok, informative = True, True, False
    for i in range(n_dbs):
        conn = _random_db(schema, strings, numbers, conditions, random.Random(seed * 1000 + i), n_rows)
        gold_rows = _execute(conn, gold)
        if gold_rows is None:
            gold_ok = False
            match = False
            conn.close()
            break
        informative |= bool(gold_rows) and any(v is not None for row in gold_rows for v in row)
        pred_rows = _execute(conn, pred)
        conn.close()
        if not _same(gold_rows, pred_rows, ordered):
            match = False
    return MatchResult(match=match, gold_ok=gold_ok, informative=informative)


def normalize_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";")
    try:
        import sqlglot

        sql = sqlglot.transpile(sql, read="sqlite", write="sqlite")[0]
    except Exception:
        pass
    return " ".join(sql.lower().split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize_sql(pred) == normalize_sql(gold)
