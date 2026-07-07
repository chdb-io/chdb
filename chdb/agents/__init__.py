"""chdb.agents — canonical agent-tool surface for chDB.

`ChDBTool` is the single Python implementation that agent frameworks shim, and
the reference implementation of the language-neutral contract in CONTRACT.md
(exercised by conformance/cases.jsonl, shared with the TypeScript binding).

.. note::
   **Beta / experimental (introduced in chdb 4.2.0).** ``chdb.agents`` is a new
   API that downstream integrations (mcp-clickhouse, langchain, llama-index)
   build on. Its surface may change in a minor release while it stabilizes — pin
   a version if you depend on it. The behavior it does expose is governed by
   CONTRACT.md; probe it with ``capabilities()`` / ``CONTRACT_VERSION`` rather
   than guessing from the package version.
"""

__beta__ = True

from .descriptors import CONTRACT_VERSION, capabilities, load_descriptors, tool_specs
from .errors import (
    ChDBError,
    ChDBReadOnlyError,
    ChDBSyntaxError,
    ChDBUnknownObjectError,
)
from .safety import InvalidIdentifier, quote_ident, quote_string
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
    "quote_string",
    "CONTRACT_VERSION",
    "capabilities",
    "load_descriptors",
    "tool_specs",
    "__beta__",
]
