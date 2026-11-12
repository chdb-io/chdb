"""
Enumerations for DataStore
"""

from enum import Enum

from .exceptions import JoinTypeError

__all__ = ['JoinType']


class JoinType(Enum):
    """JOIN types supported by DataStore"""

    inner = "INNER"
    left = "LEFT"
    right = "RIGHT"
    outer = "FULL OUTER"
    left_outer = "LEFT OUTER"
    right_outer = "RIGHT OUTER"
    full_outer = "FULL OUTER"
    cross = "CROSS"
    left_anti = "LEFT ANTI"
    right_anti = "RIGHT ANTI"

    def __str__(self):
        return self.value

    @classmethod
    def parse(cls, how) -> "JoinType":
        """
        The join type ``how`` names: a JoinType, or one of the spellings
        ``join()`` and ``merge()`` accept, in any case, with spaces or
        underscores: inner, left, left outer, right, right outer, outer,
        full, full outer, cross, left anti, right anti.
        """
        if isinstance(how, cls):
            return how
        key = " ".join(str(how).lower().replace("_", " ").split())
        join_type = _SPELLINGS.get(key)
        if join_type is not None:
            return join_type
        if key.endswith("semi"):
            raise JoinTypeError(
                f"Unsupported join type: {how!r}. pandas has no semi join; "
                "filter with isin() on the keys instead"
            )
        raise JoinTypeError(
            f"Invalid join type: {how!r}. Expected one of: {', '.join(_SPELLINGS)}"
        )

    @property
    def pandas_how(self) -> str:
        """The ``how`` of ``pd.merge`` for this join type."""
        return _PANDAS_HOW[self]

    @property
    def runs_in_sql(self) -> bool:
        """
        Whether a SQL JOIN gives pandas' result for this join type. A cross
        join's row order and an anti join's row labels come from pandas.
        """
        return self not in (JoinType.cross, JoinType.left_anti, JoinType.right_anti)


_SPELLINGS = {
    "inner": JoinType.inner,
    "left": JoinType.left,
    "left outer": JoinType.left,
    "right": JoinType.right,
    "right outer": JoinType.right,
    "outer": JoinType.outer,
    "full": JoinType.outer,
    "full outer": JoinType.outer,
    "cross": JoinType.cross,
    "left anti": JoinType.left_anti,
    "right anti": JoinType.right_anti,
}

_PANDAS_HOW = {
    JoinType.inner: "inner",
    JoinType.left: "left",
    JoinType.left_outer: "left",
    JoinType.right: "right",
    JoinType.right_outer: "right",
    JoinType.outer: "outer",
    JoinType.cross: "cross",
    JoinType.left_anti: "left_anti",
    JoinType.right_anti: "right_anti",
}
