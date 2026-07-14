"""smolagents tools over chdb.agents.ChDBTool.

Usage::

    from chdb.agents.smolagents import chdb_smol_tools
    from smolagents import CodeAgent, InferenceClientModel

    agent = CodeAgent(
        tools=chdb_smol_tools(attachments={"events": "data/events.parquet"}),
        model=InferenceClientModel(),
    )
    agent.run("What are the top 5 event types by count?")

Requires the ``smolagents`` package (not a chdb dependency); importing this
module without it raises a descriptive ImportError.

Each tool returns the JSON envelope produced by ``ChDBTool.call()`` —
``{"ok": true, "result": …}`` or ``{"ok": false, "error": {code, type,
message}}`` — so the model always reads engine errors and can self-correct.
Tool names, descriptions, and input metadata come from the descriptors this
package bundles (the single source of truth across chDB bindings).

These tools hold a live engine, so they are for local use, not for
``push_to_hub`` (smolagents serializes tool source code and literal init
parameters when sharing; a process-bound database session cannot survive
that round trip).
"""

import json

try:
    from smolagents import Tool
except ImportError as exc:  # pragma: no cover - exercised only without smolagents
    raise ImportError(
        "The 'smolagents' package is required for chdb.agents.smolagents. "
        "Install it with: pip install smolagents"
    ) from exc

from chdb.agents.descriptors import load_descriptors
from chdb.agents.tool import ChDBTool

_DESCRIPTORS = {t["name"]: t for t in load_descriptors()["tools"]}


def _inputs_for(tool_name):
    """Build the smolagents ``inputs`` dict from the bundled descriptors."""
    inputs = {}
    for param in _DESCRIPTORS[tool_name].get("params", []):
        entry = {"type": param["type"], "description": param["description"]}
        if not param.get("required", False):
            entry["nullable"] = True
        inputs[param["name"]] = entry
    return inputs


class _ChDBBaseTool(Tool):
    """Shared engine plumbing for the chDB smolagents tools.

    Each tool instance lazily creates its own chDB session from the
    constructor arguments on first use, or reuses an existing
    ``chdb.agents.ChDBTool`` passed as ``engine``. Tools that must see each
    other's state (attach_file followed by run_select_query) have to share
    one engine — build the suite with ``chdb_smol_tools()`` or pass the same
    ``engine`` to each tool.
    """

    output_type = "string"

    def __init__(
        self,
        path=":memory:",
        *,
        read_only=True,
        max_rows=1000,
        max_bytes=1_000_000,
        max_execution_time=None,
        file_allowlist=None,
        attachments=None,
        engine=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._engine = engine
        self._engine_config = dict(
            path=path,
            read_only=read_only,
            max_rows=max_rows,
            max_bytes=max_bytes,
            max_execution_time=max_execution_time,
            file_allowlist=file_allowlist,
            attachments=attachments,
        )
        self._owns_engine = engine is None

    @property
    def engine(self):
        """The underlying ``chdb.agents.ChDBTool``, created on first use."""
        if self._engine is None:
            config = self._engine_config
            self._engine = ChDBTool(
                config["path"],
                read_only=config["read_only"],
                max_rows=config["max_rows"],
                max_bytes=config["max_bytes"],
                max_execution_time=config["max_execution_time"],
                file_allowlist=config["file_allowlist"],
                attachments=config["attachments"],
            )
        return self._engine

    def close(self):
        """Close the engine this tool created; an injected one stays with its owner."""
        if self._owns_engine and self._engine is not None:
            self._engine.close()
            self._engine = None

    def _dispatch(self, arguments):
        return json.dumps(self.engine.call(self.name, arguments))


class ChDBRunSelectQueryTool(_ChDBBaseTool):
    name = "run_select_query"
    description = _DESCRIPTORS["run_select_query"]["description"]
    inputs = _inputs_for("run_select_query")

    def forward(self, sql: str, params: dict | None = None) -> str:
        return self._dispatch({"sql": sql, "params": params})


class ChDBListDatabasesTool(_ChDBBaseTool):
    name = "list_databases"
    description = _DESCRIPTORS["list_databases"]["description"]
    inputs = _inputs_for("list_databases")

    def forward(self) -> str:
        return self._dispatch({})


class ChDBListTablesTool(_ChDBBaseTool):
    name = "list_tables"
    description = _DESCRIPTORS["list_tables"]["description"]
    inputs = _inputs_for("list_tables")

    def forward(self, database: str | None = None) -> str:
        return self._dispatch({"database": database})


class ChDBDescribeTableTool(_ChDBBaseTool):
    name = "describe_table"
    description = _DESCRIPTORS["describe_table"]["description"]
    inputs = _inputs_for("describe_table")

    def forward(self, target: str, database: str | None = None) -> str:
        return self._dispatch({"target": target, "database": database})


class ChDBGetSampleDataTool(_ChDBBaseTool):
    name = "get_sample_data"
    description = _DESCRIPTORS["get_sample_data"]["description"]
    inputs = _inputs_for("get_sample_data")

    def forward(
        self, target: str, database: str | None = None, limit: int | None = None
    ) -> str:
        return self._dispatch({"target": target, "database": database, "limit": limit})


class ChDBListFunctionsTool(_ChDBBaseTool):
    name = "list_functions"
    description = _DESCRIPTORS["list_functions"]["description"]
    inputs = _inputs_for("list_functions")

    def forward(self, like: str | None = None, limit: int | None = None) -> str:
        return self._dispatch({"like": like, "limit": limit})


class ChDBAttachFileTool(_ChDBBaseTool):
    name = "attach_file"
    description = _DESCRIPTORS["attach_file"]["description"]
    inputs = _inputs_for("attach_file")

    def forward(self, name: str, path: str, format: str | None = None) -> str:
        return self._dispatch({"name": name, "path": path, "format": format})


def chdb_smol_tools(
    path=":memory:",
    *,
    read_only=True,
    max_rows=1000,
    max_bytes=1_000_000,
    max_execution_time=None,
    file_allowlist=None,
    attachments=None,
    engine=None,
):
    """Build the chDB tool suite for smolagents over one shared engine.

    All returned tools run against the same chDB session, so a table
    registered by attach_file (or declared via ``attachments``) is visible
    to run_select_query and the introspection tools. attach_file is included
    only for writable suites (``read_only=False``); on a read-only session,
    declare files via ``attachments`` instead.

    Lifecycle: when the factory creates the engine itself, the FIRST
    returned tool owns it — ``tools[0].close()`` releases the session for
    the whole suite. Pass an existing ``chdb.agents.ChDBTool`` as ``engine``
    to control its lifecycle yourself instead (the other arguments are then
    ignored and no tool will close it).
    """
    factory_owns_engine = engine is None
    if engine is None:
        engine = ChDBTool(
            path,
            read_only=read_only,
            max_rows=max_rows,
            max_bytes=max_bytes,
            max_execution_time=max_execution_time,
            file_allowlist=file_allowlist,
            attachments=attachments,
        )

    tool_classes = [
        ChDBRunSelectQueryTool,
        ChDBListDatabasesTool,
        ChDBListTablesTool,
        ChDBDescribeTableTool,
        ChDBGetSampleDataTool,
        ChDBListFunctionsTool,
    ]
    if not getattr(engine, "read_only", True):
        tool_classes.append(ChDBAttachFileTool)

    tools = [tool_class(engine=engine) for tool_class in tool_classes]
    if factory_owns_engine:
        # Hand ownership of the factory-created session to the first tool,
        # so the suite has exactly one deterministic closer and a plain
        # `chdb_smol_tools()` call cannot leak the engine.
        tools[0]._owns_engine = True
    return tools
