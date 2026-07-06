from lorathresh.data.sql import SqlExample, is_scorable, make_splits, score

CTX = "CREATE TABLE head (name VARCHAR, age INTEGER)"


def _examples(n: int) -> list[SqlExample]:
    return [SqlExample(CTX, f"Who is older than {i}?", f"SELECT name FROM head WHERE age > {i}") for i in range(n)]


def test_degenerate_gold_is_unscorable():
    # Numeric column names: SQLite reads 2008 as a constant, so gold is empty on every DB.
    ctx = 'CREATE TABLE t ("2007" VARCHAR, "2008" VARCHAR)'
    assert not is_scorable(SqlExample(ctx, "q", 'SELECT 2007 FROM t WHERE 2008 = "153"'))
    assert is_scorable(_examples(1)[0])


def test_splits_disjoint_deterministic_and_sized():
    exs = _examples(60)
    train, ev, stats = make_splits(exs, n_train=30, n_eval=20, seed=0)
    assert len(train) == 30 and len(ev) == 20
    assert not {e.question for e in train} & {e.question for e in ev}
    assert (train, ev) == make_splits(exs, n_train=30, n_eval=20, seed=0)[:2]


def test_eval_split_independent_of_train_size():
    exs = _examples(60)
    assert make_splits(exs, n_train=10, n_eval=20)[1] == make_splits(exs, n_train=35, n_eval=20)[1]


def test_duplicates_and_unscorable_dropped():
    bad = SqlExample('CREATE TABLE t ("2008" VARCHAR)', "bad", 'SELECT 2008 FROM t WHERE 2008 = "x"')
    exs = _examples(10) + _examples(10) + [bad]
    train, ev, stats = make_splits(exs, n_train=100, n_eval=5)
    assert len(train) + len(ev) == 10
    assert stats["duplicates"] == 10 and stats["unscorable"] == 1


def test_score_uses_first_line():
    ex = _examples(1)[0]
    assert score(" SELECT name FROM head WHERE 0 < age\nSchema: ...", ex) == {"exec_match": True, "exact_match": False}
