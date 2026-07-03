from lorathresh.sqlexec import exact_match, execution_match, extract_literals, parse_schema

CTX = "CREATE TABLE head (name VARCHAR, age INTEGER, born_state VARCHAR)"


def test_parse_schema():
    assert parse_schema(CTX) == {"head": [("name", "VARCHAR"), ("age", "INTEGER"), ("born_state", "VARCHAR")]}


def test_duplicate_table_declarations_are_merged():
    ctx = "CREATE TABLE Faculty (Fname VARCHAR); CREATE TABLE faculty (Lname VARCHAR, fname VARCHAR)"
    assert parse_schema(ctx) == {"Faculty": [("Fname", "VARCHAR"), ("Lname", "VARCHAR")]}
    assert execution_match("SELECT Lname FROM Faculty", "SELECT lname FROM faculty", ctx).match


def test_extract_literals():
    strings, numbers = extract_literals("SELECT name FROM head WHERE born_state = 'California' AND age > 56")
    assert strings == ["California"] and numbers == [56.0]


def test_equivalent_queries_match():
    gold = "SELECT name FROM head WHERE age > 56"
    pred = "select NAME from head where 56 < age"
    assert execution_match(pred, gold, CTX).match


def test_different_literal_does_not_match():
    # The false positive this module guards against: on empty tables these would "match".
    gold = "SELECT COUNT(*) FROM head WHERE born_state = 'California'"
    pred = "SELECT COUNT(*) FROM head WHERE born_state = 'Texas'"
    r = execution_match(pred, gold, CTX)
    assert r.gold_ok and r.informative and not r.match


def test_one_wrong_literal_among_many_conditions_is_caught():
    # Found by the real-data audit: with several AND'ed conditions no random row satisfied gold,
    # so gold and a wrong prediction both returned empty and "matched".
    ctx = "CREATE TABLE t (engine VARCHAR, tyre VARCHAR, chassis VARCHAR, driver VARCHAR)"
    gold = 'SELECT engine FROM t WHERE tyre = "g" AND chassis = "003 002" AND driver = "jackie stewart"'
    pred = 'SELECT engine FROM t WHERE tyre = "zzz" AND chassis = "003 002" AND driver = "jackie stewart"'
    r = execution_match(pred, gold, ctx)
    assert r.informative and not r.match
    assert execution_match(gold, gold, ctx).match


def test_flipped_comparison_on_text_column_is_caught():
    ctx = "CREATE TABLE t (rider VARCHAR, laps VARCHAR, grid VARCHAR)"
    gold = "SELECT rider FROM t WHERE laps = 14 AND grid > 23"
    pred = "SELECT rider FROM t WHERE laps = 14 AND grid < 23"
    r = execution_match(pred, gold, ctx)
    assert r.informative and not r.match


def test_wrong_column_does_not_match():
    assert not execution_match("SELECT age FROM head", "SELECT name FROM head", CTX).match


def test_invalid_prediction_is_mismatch():
    assert not execution_match("SELEC name FRM head", "SELECT name FROM head", CTX).match


def test_order_sensitivity_only_with_order_by():
    gold = "SELECT name FROM head ORDER BY age"
    assert not execution_match("SELECT name FROM head ORDER BY age DESC", gold, CTX).match


def test_exact_match_normalization():
    assert exact_match("SELECT  name FROM head;", "select name from head")
    assert not exact_match("SELECT age FROM head", "SELECT name FROM head")
