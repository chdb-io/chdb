"""
Translate pandas 3.x ``pd.col(...)`` expressions into chdb-ds expressions.

``pandas.api.typing.Expression`` is a black-box ``Callable[[DataFrame], Any]``
with no inspectable AST. We evaluate it against a ``_FakeFrame`` whose
``__getitem__`` returns a chdb-ds ``Field``; pandas then builds the rest of
the tree on top of chdb-ds nodes (``ArithmeticExpression`` /
``BinaryCondition`` / ``Function`` / etc.) using Field's existing operator
and accessor overloads.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


def _parse_pandas_version() -> tuple:
    """Parse pandas version into a comparable tuple."""
    try:
        import pandas as _pd

        parts = _pd.__version__.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor)
    except Exception:
        return (0, 0)


PANDAS_3_PLUS: bool = _parse_pandas_version() >= (3, 0)


def _resolve_pandas_expression_type():
    # ``None`` on pandas < 3; every helper degrades to a no-op.
    if not PANDAS_3_PLUS:
        return None
    try:
        from pandas.api.typing import Expression as _PdExpr  # type: ignore[attr-defined]
        return _PdExpr
    except Exception:
        return None


_PANDAS_EXPRESSION_TYPE = _resolve_pandas_expression_type()


def is_pandas_col_expression(obj: Any) -> bool:
    """True iff *obj* is a pandas 3.x ``pd.col(...)`` expression.

    Safe on pandas < 3 (returns False).
    """
    if _PANDAS_EXPRESSION_TYPE is None:
        return False
    return isinstance(obj, _PANDAS_EXPRESSION_TYPE)


class _FakeColumns:
    """Stand-in for ``DataFrame.columns`` with just enough surface for
    ``pd.col``'s membership check and ``tolist()`` error path.

    Permissive by default: every name reports as present, so translation
    doesn't fail on schema-vs-name skew. Pass an iterable to enable
    strict membership checks.
    """

    __slots__ = ("_names", "_strict")

    def __init__(self, names: Optional[Iterable[Any]]) -> None:
        if names is None:
            self._names = None
            self._strict = False
        else:
            self._names = list(names)
            self._strict = True

    def __contains__(self, name: Any) -> bool:
        if not self._strict:
            return True
        return name in self._names

    def tolist(self) -> list:
        return [] if self._names is None else list(self._names)

    def __iter__(self):
        return iter([] if self._names is None else self._names)

    def __len__(self) -> int:
        return 0 if self._names is None else len(self._names)


class _FakeFrame:
    """Symbolic ``DataFrame`` whose ``__getitem__`` yields chdb-ds Fields."""

    __slots__ = ("_columns",)

    def __init__(self, columns: Optional[Iterable[Any]]) -> None:
        self._columns = _FakeColumns(columns)

    @property
    def columns(self) -> _FakeColumns:
        return self._columns

    def __getitem__(self, name: Any):
        from .expressions import Field

        if not isinstance(name, str):
            name = str(name)
        return Field(name)


class PandasColTranslationError(TypeError):
    """Raised when a ``pd.col`` expression cannot be translated to SQL
    *and* there is no way to fall back (currently unused; we always
    fall back via :class:`PandasFallbackExpr` instead)."""


class PandasFallbackExpr:
    """Wrapper for a ``pd.col`` expression that chdb-ds cannot translate
    to SQL (e.g. ``.astype("category")``, ``np.log(...)``, ``.apply(...)``).

    At execute time the expression evaluator unwraps this node and runs
    the original pandas Expression against the in-memory DataFrame
    (``original._eval_expression(df)``), so semantics match pandas exactly.

    SQL builder treats this node as non-pushable: any LazyOp whose
    expression tree contains a ``PandasFallbackExpr`` falls back to the
    pandas segment via the existing ``can_push_to_sql`` machinery.

    This class deliberately does *not* inherit ``Expression`` — that would
    let it sneak into SQL paths via the catch-all ``isinstance(expr,
    Expression)`` branch in ExpressionEvaluator. Instead we let it match
    on a dedicated isinstance check.
    """

    __slots__ = ("original",)

    def __init__(self, original: Any) -> None:
        self.original = original

    def __repr__(self) -> str:
        return f"PandasFallbackExpr({self.original!r})"

    def to_sql(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError(
            "PandasFallbackExpr cannot be pushed to SQL; should be "
            "evaluated by ExpressionEvaluator on a real DataFrame."
        )


def translate_pandas_expression(
    expr: Any,
    columns: Optional[Iterable[Any]] = None,
) -> Any:
    """Translate a ``pd.col`` ``Expression`` into a chdb-ds expression tree.

    Non-pandas-Expression inputs pass through unchanged. If *columns* is
    given, pandas' own "column must exist" check is enforced during
    symbolic eval; otherwise validation is deferred to SQL execution.

    If symbolic eval fails (e.g. ``.astype("category")``, ``np.log(...)``,
    or any method we don't model on ``Field``), the *whole* original
    pandas Expression is wrapped in a :class:`PandasFallbackExpr` and
    returned. The downstream pipeline will then route the containing op
    to the pandas execution segment, where pandas itself evaluates the
    expression against the real DataFrame.
    """
    if not is_pandas_col_expression(expr):
        return expr

    fake = _FakeFrame(columns)
    try:
        return expr._eval_expression(fake)  # noqa: SLF001
    except (AttributeError, TypeError):
        # Cannot push down; wrap so the pandas segment runs it instead.
        return PandasFallbackExpr(expr)


def maybe_translate(expr: Any, columns: Optional[Iterable[Any]] = None) -> Any:
    """Translate-if-applicable, else identity. Same as
    :func:`translate_pandas_expression`."""
    return translate_pandas_expression(expr, columns=columns)


__all__ = [
    "PANDAS_3_PLUS",
    "PandasColTranslationError",
    "PandasFallbackExpr",
    "is_pandas_col_expression",
    "translate_pandas_expression",
    "maybe_translate",
]
