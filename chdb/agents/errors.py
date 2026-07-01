"""Typed errors for chdb.agents, parsed from chDB's stable exception shape.

chDB raises messages of the form::

    Code: 164. DB::Exception: <message>. (READONLY)

`parse_error` turns that into a `ChDBError` carrying the numeric `code`, the
symbolic `type`, and a cleaned `message`, so callers (and, across languages,
every other chdb-io agent tool) map the *same* engine failure to the *same*
typed surface. The regex and the code->class mapping are the single source of
truth referenced by the cross-language CONTRACT.md.
"""

import re

__all__ = [
    "ChDBError",
    "ChDBReadOnlyError",
    "ChDBSyntaxError",
    "ChDBUnknownObjectError",
    "parse_error",
]

# `Code: N. DB::Exception: <msg>. (TYPE)` — TYPE is the LAST (UPPER_SNAKE) token.
# `msg` is greedy so a parenthesized UPPER_SNAKE inside the message body (e.g.
# "... (SOME_ENUM) ...") stays in the message and the real trailing type wins.
_ERR_RE = re.compile(
    r"Code:\s*(?P<code>\d+)\.\s*DB::Exception:\s*(?P<msg>.*)\((?P<type>[A-Z0-9_]+)\)",
    re.S,
)


class ChDBError(Exception):
    """Base error. `code`/`type`/`message` are always populated."""

    def __init__(self, message, code=0, type="UNKNOWN"):
        super().__init__(message)
        self.code = code
        self.type = type
        self.message = message

    def to_dict(self):
        return {"code": self.code, "type": self.type, "message": self.message}


class ChDBReadOnlyError(ChDBError):
    """A write/DDL was rejected because the tool session is read-only (code 164)."""


class ChDBSyntaxError(ChDBError):
    """Parse / type / argument error in the submitted SQL."""


class ChDBUnknownObjectError(ChDBError):
    """Unknown table / database / function / column / setting."""


# ClickHouse error code -> ChDBError subclass. Kept small and explicit; the
# CONTRACT lists exactly these so other languages classify identically.
_CODE_TO_CLASS = {
    164: ChDBReadOnlyError,          # READONLY
    62: ChDBSyntaxError,             # SYNTAX_ERROR
    46: ChDBUnknownObjectError,      # UNKNOWN_FUNCTION
    47: ChDBUnknownObjectError,      # UNKNOWN_IDENTIFIER
    60: ChDBUnknownObjectError,      # UNKNOWN_TABLE
    81: ChDBUnknownObjectError,      # UNKNOWN_DATABASE
    115: ChDBUnknownObjectError,     # UNKNOWN_SETTING
}

_TYPE_TO_CLASS = {
    "READONLY": ChDBReadOnlyError,
}


def parse_error(exc_or_message):
    """Return a typed `ChDBError` for a raw engine exception or message string.

    Non-conforming input yields a generic `ChDBError` wrapping the text, so the
    caller never has to special-case "the message didn't parse".
    """
    message = exc_or_message.args[0] if isinstance(exc_or_message, BaseException) and exc_or_message.args else str(exc_or_message)
    m = _ERR_RE.search(message)
    if not m:
        return ChDBError(message.strip())
    code = int(m.group("code"))
    type_ = m.group("type")
    # greedy msg keeps the trailing ". " that precedes the (TYPE); trim it
    msg = m.group("msg").strip().rstrip(".").strip()
    cls = _CODE_TO_CLASS.get(code) or _TYPE_TO_CLASS.get(type_, ChDBError)
    return cls(msg, code=code, type=type_)
