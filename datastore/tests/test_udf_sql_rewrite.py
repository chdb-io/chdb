"""A Python rule that can be said in SQL is said in SQL.

The translation only earns its place if the answer is unchanged, so most of
this file compares the rewritten expression against the very Python it replaced,
over values chosen to sit on the boundaries the rule tests.
"""

import pandas as pd
import pytest

from datastore import DataStore
from datastore.udf_sql import RewrittenCall, sql_rewrite_for

FRAME = pd.DataFrame(
    {
        "revenue": [-10.0, 0.0, 74.99, 75.0, 149.99, 150.0, 1000.0],
        "channel": ["a", "b", "a", "c", "b", "a", "c"],
    }
)


# --- the shapes a scalar business rule comes in ---------------------------


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


def clamped(value):
    return min(max(value, 0.0), 100.0)


def ternary(value):
    return value * 2 if value > 0 else 0.0


def rounded(value):
    return round(value / 3, 2)


def negated(value):
    return -value


def compared(value):
    return value > 100


# --- the shapes that must be left alone -----------------------------------


def loops(value):
    total = 0.0
    for _ in range(3):
        total += value
    return total


def calls_a_method(value):
    return value.bit_length()


def imports_something(value):
    import math

    return math.log(value)


def uses_a_global(value):
    return value * FACTOR  # noqa: F821 - deliberately unbound


def floor_divides(value):
    return value // 3


def assigns_first(value):
    doubled = value * 2
    return doubled


def has_else(value):
    if value > 0:
        return value
    else:
        return 0.0


def sql_of(fn):
    rewrite = sql_rewrite_for(fn)
    assert rewrite is not None, f"{fn.__name__} should translate"
    return rewrite.sql([f'"{name}"' for name in rewrite.parameters])


def test_a_graduated_rule_becomes_multi_if():
    assert sql_of(graduated) == (
        'multiIf(("revenue" >= 150), ("revenue" * 0.92), '
        '("revenue" >= 75), ("revenue" * 0.95), ("revenue" * 0.97))'
    ).replace('"revenue"', '"value"')


def test_the_supported_shapes_translate():
    assert sql_of(with_docstring) == '("value" * 2)'
    assert sql_of(two_arguments) == '("value" * (1 - "rate"))'
    assert sql_of(clamped) == 'least(greatest("value", 0.0), 100.0)'
    assert sql_of(ternary) == 'if(("value" > 0), ("value" * 2), 0.0)'
    assert sql_of(rounded) == 'round(("value" / 3), 2)'
    assert sql_of(negated) == '(-"value")'
    assert sql_of(compared) == '("value" > 100)'


@pytest.mark.parametrize(
    "fn",
    [loops, calls_a_method, imports_something, uses_a_global, floor_divides,
     assigns_first, has_else, len, min],
    ids=lambda fn: getattr(fn, "__name__", str(fn)),
)
def test_anything_outside_the_list_is_left_alone(fn):
    """A wrong answer computed quickly is worse than a right one in a subprocess."""
    assert sql_rewrite_for(fn) is None


def test_a_lambda_is_left_alone():
    assert sql_rewrite_for(lambda value: value * 2) is None


# --- the answer has to be the same ----------------------------------------


@pytest.mark.parametrize(
    "fn", [graduated, with_docstring, clamped, ternary, rounded, negated],
    ids=lambda fn: fn.__name__,
)
def test_the_rewrite_answers_what_the_python_answers(fn):
    store = DataStore(FRAME)

    rewritten = store.assign(out=store["revenue"].apply(fn)).to_pandas()["out"]
    in_python = FRAME["revenue"].map(fn)

    pd.testing.assert_series_equal(
        rewritten.reset_index(drop=True).astype("float64"),
        in_python.reset_index(drop=True).astype("float64"),
        check_names=False,
    )


def test_apply_compiles_the_rule_into_the_query():
    store = DataStore(FRAME)

    sql = store.assign(net=store["revenue"].apply(graduated)).to_sql(
        execution_format=True
    )

    assert "multiIf" in sql
    # No function call survives: nothing has to be registered or deployed.
    assert "graduated" not in sql


def test_a_function_that_cannot_be_translated_still_goes_to_pandas():
    store = DataStore(FRAME)

    sql = store.assign(out=store["revenue"].apply(loops)).to_sql(
        execution_format=True
    )

    assert "out" not in sql


def test_the_rewrite_survives_aliasing_and_rebuilding():
    from copy import copy

    store = DataStore(FRAME)
    call = store["revenue"].apply(graduated)._expr

    assert isinstance(call, RewrittenCall)
    assert isinstance(copy(call), RewrittenCall)
    assert isinstance(call.as_("net"), RewrittenCall)
    assert isinstance(call.rebuild_with_args(list(call.args)), RewrittenCall)


def test_a_registered_udf_is_rewritten_rather_than_called():
    """Being deployable does not make a call the better plan."""
    import chdb

    from datastore.udf import binding_named

    @chdb.func(arg_types=["Float64"], return_type="Float64")
    def registered_rule(value):
        return value * 0.5

    binding = binding_named("registered_rule")
    assert binding is not None and binding.rewrite is not None

    store = DataStore(FRAME)
    sql = store.assign(half=store["revenue"].apply(registered_rule)).to_sql(
        execution_format=True
    )

    assert "registered_rule(" not in sql
    assert '("revenue" * 0.5)' in sql or 'CAST("revenue" AS Float64) * 0.5' in sql


def test_a_declared_type_is_honoured_by_the_rewrite():
    """The Python would have been handed the declared type; the SQL is too."""
    import chdb

    @chdb.func(arg_types=["Float64"], return_type="Float64")
    def typed_rule(value):
        return value * 2

    store = DataStore(pd.DataFrame({"revenue": [1, 2, 3]}))
    sql = store.assign(out=store["revenue"].apply(typed_rule)).to_sql(
        execution_format=True
    )

    assert 'CAST("revenue" AS Float64)' in sql
