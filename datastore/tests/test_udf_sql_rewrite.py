"""Compiling a Python rule into the query, when its author asks for that.

Three things a function can become, and the difference matters to the person
who wrote it: pandas, a call on the server, or part of the statement. The last
one is opt-in, because reading a body and finding it translatable proves
nothing about types, NULLs, exceptions or truthiness - so most of this file is
about what is refused, and the rest compares a translated rule against the very
Python it replaced.
"""

import pandas as pd
import pytest

from datastore import DataStore
from datastore.udf import build_rewrite
from datastore.udf_sql import RewrittenCall, numeric_arg_types, sql_rewrite_for

FRAME = pd.DataFrame(
    {
        "revenue": [-10.0, 0.0, 74.99, 75.0, 149.99, 150.0, 1000.0],
        "channel": ["a", "b", "a", "c", "b", "a", "c"],
    }
)


# --- inside the stable set ------------------------------------------------


def graduated(value):
    if value >= 150:
        return value * 0.92
    if value >= 75:
        return value * 0.95
    return value * 0.97


def with_docstring(value):
    """A rule that documents itself."""
    return value * 2


def two_arguments(value, rate):
    return value * (1 - rate)


def ternary(value):
    return value * 2 if value > 0 else 0.0


def negated(value):
    return -value


def compared(value):
    return value > 100


def divided_by_a_literal(value):
    return value / 3


# --- outside it, each for a reason ----------------------------------------


def uses_and(value):
    return value > 100 and value < 200


def calls_a_builtin(value):
    return min(value, 100.0)


def rounds(value):
    return round(value, 2)


def converts(value):
    return float(value) * 2


def divides_by_an_argument(value, divisor):
    return value / divisor


def floor_divides(value):
    return value // 3


def returns_a_string(value):
    return "high" if value > 100 else "low"


def loops(value):
    total = 0.0
    for _ in range(3):
        total += value
    return total


def has_else(value):
    if value > 0:
        return value
    else:
        return 0.0


def sql_of(fn):
    rewrite = sql_rewrite_for(fn)
    assert rewrite is not None, f"{fn.__name__} should translate"
    return rewrite.sql([f'"{name}"' for name in rewrite.parameters])


def test_the_stable_set_translates():
    assert sql_of(graduated) == (
        'multiIf(("value" >= 150), ("value" * 0.92), '
        '("value" >= 75), ("value" * 0.95), ("value" * 0.97))'
    )
    assert sql_of(with_docstring) == '("value" * 2)'
    assert sql_of(two_arguments) == '("value" * (1 - "rate"))'
    assert sql_of(ternary) == 'if(("value" > 0), ("value" * 2), 0.0)'
    assert sql_of(negated) == '(-"value")'
    assert sql_of(compared) == '("value" > 100)'
    assert sql_of(divided_by_a_literal) == '("value" / 3)'


@pytest.mark.parametrize(
    "fn",
    [uses_and, calls_a_builtin, rounds, converts, divides_by_an_argument,
     floor_divides, returns_a_string, loops, has_else, len, min],
    ids=lambda fn: getattr(fn, "__name__", str(fn)),
)
def test_everything_else_is_refused(fn):
    """Each of these disagrees with SQL somewhere, or cannot be shown to agree."""
    assert sql_rewrite_for(fn) is None


def test_a_lambda_is_refused():
    assert sql_rewrite_for(lambda value: value * 2) is None


def test_only_numeric_declarations_qualify():
    assert numeric_arg_types(["Float64", "Int64"]) is True
    assert numeric_arg_types(["Decimal(10, 2)"]) is True
    assert numeric_arg_types(["String"]) is False
    assert numeric_arg_types(["Nullable(Float64)"]) is False
    assert numeric_arg_types([]) is False
    assert numeric_arg_types(None) is False


# --- what each of the three treatments compiles to -------------------------


def test_an_ordinary_function_is_left_to_pandas():
    """Being translatable is not the same as having been asked to translate."""
    store = DataStore(FRAME)

    sql = store.assign(out=store["revenue"].apply(graduated)).to_sql(
        execution_format=True
    )

    assert "multiIf" not in sql
    assert "out" not in sql


def test_a_registered_function_is_called_rather_than_translated():
    import chdb

    @chdb.func(arg_types=["Float64"], return_type="Float64")
    def called_rule(value):
        return value * 0.5

    store = DataStore(FRAME)
    sql = store.assign(half=store["revenue"].apply(called_rule)).to_sql(
        execution_format=True
    )

    assert "called_rule(" in sql
    assert "0.5" not in sql


def test_asking_for_sql_puts_the_rule_in_the_statement():
    import chdb

    @chdb.func(arg_types=["Float64"], return_type="Float64", rewrite="sql")
    def inlined_rule(value):
        if value >= 150:
            return value * 0.92
        return value * 0.97

    store = DataStore(FRAME)
    sql = store.assign(out=store["revenue"].apply(inlined_rule)).to_sql(
        execution_format=True
    )

    assert "multiIf" in sql
    assert "inlined_rule(" not in sql
    # The declared type is honoured, as it would have been for a call.
    assert 'CAST("revenue" AS Float64)' in sql


def test_asking_for_the_impossible_says_why():
    import chdb

    with pytest.raises(ValueError, match="outside the stable set"):

        @chdb.func(arg_types=["Float64"], return_type="Float64", rewrite="sql")
        def not_translatable(value):
            total = 0.0
            for _ in range(3):
                total += value
            return total

    with pytest.raises(ValueError, match="numeric"):

        @chdb.func(arg_types=["String"], return_type="String", rewrite="sql")
        def not_numeric(value):
            return value

    with pytest.raises(ValueError, match="rewrite mode"):

        @chdb.func(arg_types=["Float64"], return_type="Float64", rewrite="magic")
        def wrong_mode(value):
            return value


def test_build_rewrite_refuses_the_same_way():
    with pytest.raises(ValueError, match="numeric"):
        build_rewrite(graduated, ["String"], "sql")
    with pytest.raises(ValueError, match="outside the stable set"):
        build_rewrite(loops, ["Float64"], "sql")


# --- the answer has to be the same ----------------------------------------


@pytest.mark.parametrize(
    "fn", [graduated, with_docstring, ternary, negated, divided_by_a_literal],
    ids=lambda fn: fn.__name__,
)
def test_a_translated_rule_answers_what_the_python_answers(fn):
    import chdb

    translated = chdb.func(
        arg_types=["Float64"], return_type="Float64", rewrite="sql"
    )(fn)
    store = DataStore(FRAME)

    rewritten = store.assign(out=store["revenue"].apply(translated)).to_pandas()["out"]
    in_python = FRAME["revenue"].map(fn)

    pd.testing.assert_series_equal(
        rewritten.reset_index(drop=True).astype("float64"),
        in_python.reset_index(drop=True).astype("float64"),
        check_names=False,
    )


def test_a_translated_rule_survives_aliasing_and_rebuilding():
    from copy import copy

    import chdb

    @chdb.func(arg_types=["Float64"], return_type="Float64", rewrite="sql")
    def aliased_rule(value):
        return value * 3

    store = DataStore(FRAME)
    call = store["revenue"].apply(aliased_rule)._expr

    assert isinstance(call, RewrittenCall)
    assert isinstance(copy(call), RewrittenCall)
    assert isinstance(call.as_("out"), RewrittenCall)
    assert isinstance(call.rebuild_with_args(list(call.args)), RewrittenCall)


def test_the_report_says_which_of_the_two_happened():
    """A reader has to be able to tell a call from a compiled-in rule."""
    import chdb

    from datastore.pushdown import set_plan_observer

    translated = chdb.func(
        arg_types=["Float64"], return_type="Float64", rewrite="sql"
    )(compared)

    reports = []
    set_plan_observer(lambda placements: reports.append(
        [placement.as_dict() for placement in placements]
    ))
    try:
        store = DataStore(FRAME)
        store.assign(out=store["revenue"].apply(translated)).to_pandas()
    finally:
        set_plan_observer(None)

    sql_segments = [
        segment for report in reports for segment in report if segment["kind"] == "sql"
    ]
    assert sql_segments
    assert {"name": "compared", "via": "sql-rewrite"} in sql_segments[-1]["udfs"]
