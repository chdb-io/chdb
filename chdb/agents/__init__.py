"""chdb.agents — canonical agent-tool surface for chDB.

`ChDBTool` is the single Python implementation that agent frameworks shim, and
the reference implementation of the language-neutral contract in CONTRACT.md
(exercised by conformance/cases.jsonl, shared with the TypeScript binding).
"""

from .errors import (
    ChDBError,
    ChDBReadOnlyError,
    ChDBSyntaxError,
    ChDBUnknownObjectError,
)
from .safety import InvalidIdentifier, quote_ident
from .tool import ChDBTool, QueryResult

__all__ = [
    "ChDBTool",
    "QueryResult",
    "ChDBError",
    "ChDBReadOnlyError",
    "ChDBSyntaxError",
    "ChDBUnknownObjectError",
    "InvalidIdentifier",
    "quote_ident",
]
