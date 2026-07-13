"""Model-visible tool descriptors and the contract version/capability surface.

`descriptors.json` (shipped next to this module, vendored byte-identical in
every chdb-io binding) is the single source of truth for the agent-tool names,
descriptions, and argument schemas the model sees. This module turns it into
framework-consumable specs, so adapters generate their schemas instead of
hand-copying text that then drifts between languages:

- ``tool_specs(dialect=...)`` — JSON-schema tool definitions in the shape each
  runtime family expects (``anthropic`` | ``openai`` | ``mcp``).
- ``capabilities()`` — ``{contract_version, tools, features}``: what this
  binding implements, for downstream feature-probing (a consumer checks
  ``features["dataframe_query"]`` instead of guessing from the package version).
"""

import copy
import json
import os

from .errors import ChDBError

__all__ = ["CONTRACT_VERSION", "load_descriptors", "tool_specs", "capabilities"]

# The agent-tool contract version (semver). Bumped whenever descriptors.json,
# conformance/cases.jsonl, or normative CONTRACT.md text changes. Tests assert
# it equals the contract_version field of both data files.
CONTRACT_VERSION = "0.2.0"

_DESCRIPTORS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "descriptors.json")
_descriptors_cache = None


def load_descriptors():
    """Return the parsed descriptors.json (the file is read once and cached;
    each call returns a deep copy, so a caller mutating the result cannot
    corrupt what tool_specs()/capabilities() generate for everyone else)."""
    global _descriptors_cache
    if _descriptors_cache is None:
        try:
            with open(_DESCRIPTORS_PATH, "r", encoding="utf-8") as fh:
                _descriptors_cache = json.load(fh)
        except Exception as e:
            # A missing/broken descriptors.json takes down every tool_specs()/
            # capabilities() consumer — surface it as a diagnosable typed error
            # instead of a raw IOError/JSONDecodeError.
            raise ChDBError(
                "cannot load agent tool descriptors from {!r}: {}".format(_DESCRIPTORS_PATH, e)
            )
    return copy.deepcopy(_descriptors_cache)


# The param types descriptors.json may declare. An unknown type must fail
# loudly: silently rendering it as a permissive schema would degrade the
# model-visible argument contract without any signal.
_PARAM_TYPES = ("string", "integer", "object")


def _json_schema(params):
    properties = {}
    required = []
    for p in params:
        if p["type"] not in _PARAM_TYPES:
            raise ChDBError(
                "unknown descriptor param type {!r} for {!r} (expected one of {})".format(
                    p["type"], p["name"], ", ".join(_PARAM_TYPES)
                )
            )
        prop = {"type": p["type"]}
        if p.get("description"):
            prop["description"] = p["description"]
        properties[p["name"]] = prop
        if p.get("required"):
            required.append(p["name"])
    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def tool_specs(dialect="anthropic"):
    """Tool definitions generated from descriptors.json.

    ``anthropic``: ``{name, description, input_schema}`` (also the historical
    ``ChDBTool.tool_specs()`` shape); ``openai``: ``{type: "function",
    function: {...}}``; ``mcp``: ``{name, description, inputSchema}``.
    """
    tools = load_descriptors()["tools"]
    if dialect == "anthropic":
        return [
            {"name": t["name"], "description": t["description"], "input_schema": _json_schema(t["params"])}
            for t in tools
        ]
    if dialect == "openai":
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": _json_schema(t["params"]),
                },
            }
            for t in tools
        ]
    if dialect == "mcp":
        return [
            {"name": t["name"], "description": t["description"], "inputSchema": _json_schema(t["params"])}
            for t in tools
        ]
    raise ChDBError(
        "unknown tool_specs dialect: {!r} (expected 'anthropic', 'openai', or 'mcp')".format(dialect),
        type="INVALID_ARGUMENT",
    )


def capabilities():
    """What this binding implements, keyed for downstream feature-probing.

    ``features`` marks the capability-gated parts of the contract: conformance
    cases carrying ``"requires": "<feature>"`` run only where that feature is
    true (``dataframe_query`` is Python-only — the agent runtime and the engine
    share a process; ``async`` here means ``aquery``/``acall``, which run the
    sync engine call in a worker thread).
    """
    return {
        "contract_version": CONTRACT_VERSION,
        "tools": [t["name"] for t in load_descriptors()["tools"]],
        "features": {
            "dataframe_query": True,
            "attachments": True,
            "file_allowlist": True,
            "max_execution_time": True,
            "async": True,
            "streaming": False,
        },
    }
