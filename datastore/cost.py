"""What a Python UDF costs on each engine, and which one that makes cheaper.

Most operators are not a real choice. A filter or an aggregation reduces the
data, so running it on the server that owns the rows is better by every measure.
A scalar Python UDF is different: it returns one row for every row it reads, so
placing it does not save any data movement - it decides which per-row cost the
query pays.

Both sides are per-row, which makes the comparison unusually clean:

    remote   rows x remote_udf_per_row
    local    rows x local_udf_per_row + rows x bytes_per_row / bandwidth

The row count cancels. What decides is the width of a row against the speed of
the link, and both are knowable before the query runs - no cardinality estimate
required.

The constants are measured, not guessed (ClickHouse 26.7, chDB 3.7, 3.5M rows):

    remote_udf_per_row  0.40 us   a ClickHouse executable UDF hands rows to an
                                  external process one at a time; with the
                                  function body deleted the cost barely moves,
                                  so this is transport, not Python
    local_udf_per_row   0.047 us  chDB calls the same function in-process
                                  (pandas .map, for comparison, is 0.087 us)

They are defaults, not laws: a slower server, a heavier function or a different
ClickHouse version moves them, and :func:`set_udf_cost_model` exists so a
deployment that has measured its own can say so.
"""

from dataclasses import dataclass, replace
from typing import Mapping, Optional, Tuple

__all__ = [
    "UdfCostModel",
    "bytes_per_row",
    "choose_udf_target",
    "current_udf_cost_model",
    "set_udf_cost_model",
]

# Bytes on the wire for one value of a ClickHouse type. Variable-length types
# have no honest fixed answer; the nominal figure below is deliberately small,
# because underestimating a row's width biases the choice towards the local
# engine, which is where the query would have run without pushdown at all.
_TYPE_BYTES = {
    "int8": 1, "uint8": 1, "bool": 1, "boolean": 1,
    "int16": 2, "uint16": 2, "date": 2,
    "int32": 4, "uint32": 4, "float32": 4, "datetime": 4, "date32": 4,
    "int64": 8, "uint64": 8, "float64": 8, "datetime64": 8, "decimal": 8,
    "int128": 16, "uint128": 16, "uuid": 16, "decimal128": 16,
    "int256": 32, "uint256": 32, "decimal256": 32,
}
_NOMINAL_VARIABLE_BYTES = 12  # String, Array, Map, ... : short values assumed


@dataclass(frozen=True)
class UdfCostModel:
    """Per-row costs, in microseconds, of running a scalar UDF on each engine."""

    remote_udf_per_row_us: float = 0.40
    local_udf_per_row_us: float = 0.047
    # What a link is assumed to carry when nothing has measured it. Left as None
    # on purpose: an unmeasured link is a reason to keep the work where it
    # already runs, not to guess a number that decides the query.
    default_bandwidth_bytes_per_s: Optional[float] = None


_MODEL = UdfCostModel()


def current_udf_cost_model() -> UdfCostModel:
    return _MODEL


def set_udf_cost_model(**overrides) -> UdfCostModel:
    """Replace the measured defaults with numbers from this deployment."""
    global _MODEL
    _MODEL = replace(_MODEL, **overrides)
    return _MODEL


def column_bytes(type_name: str) -> int:
    """Bytes one value of ``type_name`` takes on the wire, as far as we can tell."""
    text = str(type_name or "").strip().lower()
    if text.startswith("nullable(") and text.endswith(")"):
        # A null map costs a byte a row on top of the value.
        return 1 + column_bytes(text[len("nullable("):-1])
    if text.startswith("lowcardinality(") and text.endswith(")"):
        # Dictionary-encoded on the wire; the index is what repeats.
        return 4
    head = text.split("(", 1)[0]
    return _TYPE_BYTES.get(head, _NOMINAL_VARIABLE_BYTES)


def bytes_per_row(schema: Mapping, columns=None) -> int:
    """How wide one row is, over the columns that would cross the wire."""
    if not schema:
        return _NOMINAL_VARIABLE_BYTES
    names = list(columns) if columns else list(schema)
    total = sum(column_bytes(schema.get(name)) for name in names if name in schema)
    return total or _NOMINAL_VARIABLE_BYTES


def choose_udf_target(
    row_bytes: int,
    bandwidth_bytes_per_s: Optional[float],
    model: Optional[UdfCostModel] = None,
) -> Tuple[bool, str]:
    """Whether the server is the cheaper place to call the UDF, and why.

    Returns ``(prefer_remote, sentence)``. The sentence carries the arithmetic,
    because a placement a reader cannot check is a placement they have to trust.
    """
    model = model or _MODEL
    bandwidth = bandwidth_bytes_per_s or model.default_bandwidth_bytes_per_s
    saved = model.remote_udf_per_row_us - model.local_udf_per_row_us
    if saved <= 0:
        # A server that calls the function for less than this engine does wins
        # before the wire is even considered.
        return True, (
            f"the server calls the UDF at {model.remote_udf_per_row_us:.3f}us a "
            f"row against {model.local_udf_per_row_us:.3f}us here, so the call "
            f"went to the rows whatever the link costs"
        )
    if bandwidth is None or bandwidth <= 0:
        return False, (
            f"a UDF costs about {model.remote_udf_per_row_us:.2f}us a row on the "
            f"server against {model.local_udf_per_row_us:.3f}us here, and nothing "
            f"has measured this link yet, so the rows stayed where the cheaper "
            f"call is"
        )
    transfer_us = row_bytes / bandwidth * 1e6
    if transfer_us > saved:
        return True, (
            f"moving a {row_bytes}-byte row costs {transfer_us:.2f}us at "
            f"{bandwidth / 1e6:.0f} MB/s, more than the {saved:.2f}us the server "
            f"charges to call the UDF, so the call went to the rows"
        )
    return False, (
        f"moving a {row_bytes}-byte row costs {transfer_us:.2f}us at "
        f"{bandwidth / 1e6:.0f} MB/s, less than the {saved:.2f}us the server "
        f"charges to call the UDF, so the rows came to the call"
    )
