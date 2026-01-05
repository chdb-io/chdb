"""
Function Definitions - Core function implementations using the registry.

This module defines all functions using the FunctionRegistry decorator.
Each function is defined ONCE here and automatically made available via:
- Expression methods (ds['col'].upper())
- F namespace (F.upper(expr))
- Accessor methods (ds['col'].str.upper())
- ColumnExpr methods (through delegation)

To add a new function:
1. Add @register_function decorator with metadata
2. Implement the builder function that returns a Function/AggregateFunction/WindowFunction
3. The function is automatically available everywhere

Convention:
- Primary name: snake_case (to_datetime, group_array)
- ClickHouse name: camelCase or original (toDateTime, groupArray)
- Aliases: include both styles for convenience
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .function_registry import (
    FunctionType,
    FunctionCategory,
    register_function,
)
from .exceptions import UnsupportedOperationError

__all__ = ['ensure_functions_registered']


# Flag to track initialization
_functions_registered = False


def ensure_functions_registered():
    """
    Ensure all functions are registered. Safe to call multiple times.

    This is called automatically when importing from datastore,
    but can be called explicitly if needed.
    """
    global _functions_registered
    if _functions_registered:
        return
    _functions_registered = True
    # Functions are registered via decorators when this module is imported
    # Just importing this module registers all functions


# =============================================================================
# STRING FUNCTIONS
# =============================================================================


@register_function(
    name='upper',
    clickhouse_name='upper',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['uppercase', 'ucase'],
    doc='Convert string to uppercase. Maps to upper(x).',
)
def _build_upper(expr, alias=None):
    from .functions import Function

    return Function('upper', expr, alias=alias)


@register_function(
    name='lower',
    clickhouse_name='lower',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['lowercase', 'lcase'],
    doc='Convert string to lowercase. Maps to lower(x).',
)
def _build_lower(expr, alias=None):
    from .functions import Function

    return Function('lower', expr, alias=alias)


@register_function(
    name='length',
    clickhouse_name='length',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['len'],
    doc='String length in bytes. Maps to length(x).',
)
def _build_length(expr, alias=None):
    from .functions import Function

    return Function('length', expr, alias=alias)


@register_function(
    name='char_length',
    clickhouse_name='char_length',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['character_length', 'charLength'],
    doc='String length in characters. Maps to char_length(x).',
)
def _build_char_length(expr, alias=None):
    from .functions import Function

    return Function('char_length', expr, alias=alias)


@register_function(
    name='substring',
    clickhouse_name='substring',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['substr', 'mid'],
    doc='Extract substring. Maps to substring(s, offset, length).',
    min_args=1,
    max_args=2,
)
def _build_substring(expr, offset: int, length: int = None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if length is not None:
        return Function('substring', expr, Literal(offset), Literal(length), alias=alias)
    return Function('substring', expr, Literal(offset), alias=alias)


@register_function(
    name='concat',
    clickhouse_name='concat',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['concatenate'],
    doc='Concatenate strings. Maps to concat(...).',
    min_args=1,
    max_args=-1,
)
def _build_concat(*args, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('concat', *[Expression.wrap(a) for a in args], alias=alias)


# Sentinel to detect missing value argument in replace()
_REPLACE_SENTINEL = object()


@register_function(
    name='replace',
    clickhouse_name='multiIf',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['replaceAll'],
    doc='Replace values in Series. Supports: replace(to_replace, value), replace({k: v}), replace([k], [v]).',
)
def _build_replace(expr, to_replace, value=_REPLACE_SENTINEL, regex: bool = False, alias=None):
    """
    Replace values in a Series (pandas-compatible value replacement).

    Args:
        expr: The expression to operate on
        to_replace: Value(s) to replace. Can be:
            - scalar: single value to replace
            - dict: {old_value: new_value, ...}
            - list: list of values to replace (requires value to be list)
        value: Replacement value(s). Required if to_replace is scalar or list.
               Can be None to replace with null.
        regex: If True and to_replace is string, use regex substring replacement
        alias: Optional alias for the result

    Examples:
        replace(1, 100)           -> replaces 1 with 100
        replace({1: 100, 2: 200}) -> replaces 1 with 100 and 2 with 200
        replace([1, 2], [100, 200]) -> same as above
        replace(1, None)          -> replaces 1 with null
    """
    from .functions import Function
    from .expressions import Literal, Expression

    # Handle regex string replacement (for str.replace compatibility)
    if regex and isinstance(to_replace, str) and value is not _REPLACE_SENTINEL and isinstance(value, str):
        return Function('replaceRegexpAll', expr, Literal(to_replace), Literal(value), alias=alias)

    # Handle dict: replace({k1: v1, k2: v2, ...})
    if isinstance(to_replace, dict):
        if not to_replace:
            # Empty dict, return expr unchanged
            return expr
        # Build multiIf args: cond1, val1, cond2, val2, ..., default
        args = []
        for k, v in to_replace.items():
            # Condition: expr = k
            cond = Expression.wrap(expr) == Expression.wrap(k)
            args.append(cond)
            args.append(Expression.wrap(v))
        # Default: original value
        args.append(expr)
        return Function('multiIf', *args, alias=alias)

    # Handle list: replace([k1, k2], [v1, v2]) or replace([k1, k2], single_value)
    if isinstance(to_replace, (list, tuple)):
        if value is _REPLACE_SENTINEL:
            raise ValueError("value is required when to_replace is a list")
        if not to_replace:
            # Empty list, return expr unchanged
            return expr
        # If value is a single scalar, broadcast it to all to_replace values
        if not isinstance(value, (list, tuple)):
            # Single value replacement: replace([v1, v2], new_value)
            # Build multiIf: if(expr in [v1, v2], new_value, expr)
            args = []
            for k in to_replace:
                cond = Expression.wrap(expr) == Expression.wrap(k)
                args.append(cond)
                args.append(Expression.wrap(value))
            args.append(expr)
            return Function('multiIf', *args, alias=alias)
        if len(to_replace) != len(value):
            raise ValueError("to_replace and value must have the same length")
        # Build multiIf args
        args = []
        for k, v in zip(to_replace, value):
            cond = Expression.wrap(expr) == Expression.wrap(k)
            args.append(cond)
            args.append(Expression.wrap(v))
        args.append(expr)
        return Function('multiIf', *args, alias=alias)

    # Handle single value: replace(to_replace, value)
    if value is _REPLACE_SENTINEL:
        raise ValueError("value is required for single value replacement")
    # Use if(expr = to_replace, value, expr)
    cond = Expression.wrap(expr) == Expression.wrap(to_replace)
    return Function('if', cond, Expression.wrap(value), expr, alias=alias)


@register_function(
    name='str_replace',
    clickhouse_name='replace',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['replaceAll', 'replace'],  # 'replace' alias for str accessor
    doc='Replace substring occurrences. Maps to replace(s, from, to). Use regex=True for regex.',
)
def _build_str_replace(expr, pattern: str, replacement: str, regex: bool = False, alias=None):
    """
    Replace substrings in a string column (pandas str.replace compatible).

    Args:
        expr: The expression to operate on
        pattern: Substring pattern to find
        replacement: Replacement string
        regex: If True, use regex replacement
        alias: Optional alias for the result

    Examples:
        str.replace('a', 'b')     -> replaces all 'a' with 'b'
        str.replace('a+', 'b', regex=True) -> regex replacement
    """
    from .functions import Function
    from .expressions import Literal

    if regex:
        return Function('replaceRegexpAll', expr, Literal(pattern), Literal(replacement), alias=alias)
    else:
        return Function('replace', expr, Literal(pattern), Literal(replacement), alias=alias)


@register_function(
    name='trim',
    clickhouse_name='trim',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['strip'],
    doc='Trim whitespace or specific characters from both sides. Maps to trim(x) or trimBoth(x, chars).',
)
def _build_trim(expr, to_strip=None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if to_strip is not None:
        # Use trimBoth to trim specific characters
        return Function('trimBoth', expr, Literal(to_strip), alias=alias)
    return Function('trim', expr, alias=alias)


@register_function(
    name='ltrim',
    clickhouse_name='trimLeft',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['trimLeft', 'lstrip'],
    doc='Trim whitespace or specific characters from left. Maps to trimLeft(x) or trimLeft(x, chars).',
)
def _build_ltrim(expr, to_strip=None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if to_strip is not None:
        return Function('trimLeft', expr, Literal(to_strip), alias=alias)
    return Function('trimLeft', expr, alias=alias)


@register_function(
    name='rtrim',
    clickhouse_name='trimRight',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['trimRight', 'rstrip'],
    doc='Trim whitespace or specific characters from right. Maps to trimRight(x) or trimRight(x, chars).',
)
def _build_rtrim(expr, to_strip=None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if to_strip is not None:
        return Function('trimRight', expr, Literal(to_strip), alias=alias)
    return Function('trimRight', expr, alias=alias)


@register_function(
    name='reverse',
    clickhouse_name='reverse',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Reverse string. Maps to reverse(x).',
)
def _build_reverse(expr, alias=None):
    from .functions import Function

    return Function('reverse', expr, alias=alias)


@register_function(
    name='starts_with',
    clickhouse_name='startsWith',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['startsWith', 'startswith'],
    doc='Check if string starts with prefix. Maps to startsWith(s, prefix).',
)
def _build_starts_with(expr, prefix: str, na=None, alias=None):
    """
    Build startswith expression.

    Args:
        expr: The expression to check
        prefix: Prefix to check for
        na: Fill value for missing values (ignored, for pandas compatibility)
        alias: Optional alias for the result
    """
    from .functions import Function
    from .expressions import Literal

    return Function('startsWith', expr, Literal(prefix), alias=alias)


@register_function(
    name='ends_with',
    clickhouse_name='endsWith',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['endsWith', 'endswith'],
    doc='Check if string ends with suffix. Maps to endsWith(s, suffix).',
)
def _build_ends_with(expr, suffix: str, na=None, alias=None):
    """
    Build endswith expression.

    Args:
        expr: The expression to check
        suffix: Suffix to check for
        na: Fill value for missing values (ignored, for pandas compatibility)
        alias: Optional alias for the result
    """
    from .functions import Function
    from .expressions import Literal

    return Function('endsWith', expr, Literal(suffix), alias=alias)


@register_function(
    name='contains',
    clickhouse_name='position',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['has_substring'],
    doc='Check if string contains substring. Returns position (>0 if found). Maps to position(s, needle). Supports pandas-compatible parameters: case, flags, na, regex.',
)
def _build_contains(expr, needle: str, case: bool = True, flags: int = 0, na=None, regex: bool = True, alias=None):
    """
    Build contains expression.

    Args:
        expr: The expression to check
        needle: Pattern to search for
        case: If True (default), case sensitive matching
        flags: Regex flags (currently ignored for chDB, used by pandas)
        na: Fill value for missing values (currently passed to pandas execution)
        regex: If True (default), treat needle as regex pattern
        alias: Optional alias for the result

    Note:
        The `na` and `flags` parameters are primarily for pandas compatibility.
        chDB currently has issues with NaN handling, so operations with na parameter
        should be executed via pandas.

    Returns:
        A boolean expression for use in filtering.
        - For regex=True: uses match() function with .* prefix for contains semantics
        - For regex=False: uses position() > 0 (1-based position, 0 if not found)
    """
    from .functions import Function
    from .expressions import Literal

    # Store pandas-specific kwargs for execution
    pandas_kwargs = {
        'pat': needle,
        'case': case,
        'flags': flags,
        'na': na,
        'regex': regex,
    }

    if regex:
        # For regex patterns, use chDB's match() function which supports regex
        # match(haystack, pattern) matches from the start of the string
        # For "contains" semantics (match anywhere in string), prepend .*
        # For case-insensitive, wrap the pattern with (?i) prefix
        if not case:
            pattern = f'(?i).*{needle}'
        else:
            pattern = f'.*{needle}'
        match_func = Function(
            'match',
            expr,
            Literal(pattern),
            pandas_name='contains',
            pandas_kwargs=pandas_kwargs,
        )
        # match() returns UInt8 (0 or 1), use > 0 for boolean
        result = match_func > 0
    else:
        # For literal string matching, use position() which is faster
        # For case-insensitive matching, use positionCaseInsensitive
        if not case:
            position_func = Function(
                'positionCaseInsensitive',
                expr,
                Literal(needle),
                pandas_name='contains',
                pandas_kwargs=pandas_kwargs,
            )
        else:
            position_func = Function(
                'position', expr, Literal(needle), pandas_name='contains', pandas_kwargs=pandas_kwargs
            )
        # Return boolean expression (position > 0) instead of position value
        result = position_func > 0

    if alias:
        result._alias = alias
    return result


@register_function(
    name='left',
    clickhouse_name='left',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Get leftmost N characters. Maps to left(s, n).',
)
def _build_left(expr, n: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('left', expr, Literal(n), alias=alias)


@register_function(
    name='right',
    clickhouse_name='right',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Get rightmost N characters. Maps to right(s, n).',
)
def _build_right(expr, n: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('right', expr, Literal(n), alias=alias)


@register_function(
    name='pad',
    clickhouse_name='leftPad',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['leftPad', 'lpad'],
    doc='Pad string to length with fill char. Maps to leftPad(s, length, fill).',
)
def _build_pad(expr, length: int, fill: str = ' ', fillchar: str = None, alias=None):
    from .functions import Function
    from .expressions import Literal

    # Support both 'fill' and 'fillchar' (pandas uses fillchar)
    actual_fill = fillchar if fillchar is not None else fill
    return Function('leftPad', expr, Literal(length), Literal(actual_fill), alias=alias)


@register_function(
    name='rpad',
    clickhouse_name='rightPad',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['rightPad'],
    doc='Pad string on right to length with fill char. Maps to rightPad(s, length, fill).',
)
def _build_rpad(expr, length: int, fill: str = ' ', fillchar: str = None, alias=None):
    from .functions import Function
    from .expressions import Literal

    # Support both 'fill' and 'fillchar' (pandas uses fillchar)
    actual_fill = fillchar if fillchar is not None else fill
    return Function('rightPad', expr, Literal(length), Literal(actual_fill), alias=alias)


@register_function(
    name='repeat',
    clickhouse_name='repeat',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Repeat string N times. Maps to repeat(s, n).',
)
def _build_repeat(expr, n: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('repeat', expr, Literal(n), alias=alias)


@register_function(
    name='split',
    clickhouse_name='splitByString',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['splitByString'],
    doc='Split string by separator. Maps to splitByString(sep, s). If sep is None, splits by whitespace. When expand=True, returns DataFrame via pandas fallback.',
)
def _build_split(expr, sep: str = None, n: int = -1, expand: bool = False, regex: bool = None, alias=None):
    from .functions import Function
    from .expressions import Literal
    from .column_expr import ColumnExpr

    # Handle expand=True case with pandas fallback
    if expand:
        # Need to use pandas fallback for expand=True
        # Create a ColumnExpr with executor that uses pandas str.split
        if isinstance(expr, ColumnExpr):
            col_expr = expr
        else:
            # If expr is not ColumnExpr, we need the datastore from somewhere
            # This case is handled in the accessor wrapper
            col_expr = None

        if col_expr is not None:

            def executor():
                series = col_expr._execute()
                return series.str.split(pat=sep, n=n if n != -1 else None, expand=True, regex=regex)

            return ColumnExpr(executor=executor, datastore=col_expr._datastore)
        else:
            # Fallback: wrap in a function that will fail gracefully
            raise UnsupportedOperationError(
                operation="str.split(expand=True)",
                reason="string split with expand=True requires ColumnExpr context to create multiple columns",
                suggestion="Use series.str.split(expand=True) on a pandas Series directly",
            )

    # Wrap with ifNull to handle Nullable columns - ClickHouse doesn't support Nullable(Array)
    # This converts NULL to empty string, which produces empty array []
    safe_expr = Function('ifNull', expr, Literal(''))

    if sep is None:
        # Default: split by whitespace (pandas behavior)
        return Function('splitByWhitespace', safe_expr, alias=alias)
    else:
        return Function('splitByString', Literal(sep), safe_expr, alias=alias)


# ---------- Additional Pandas .str methods ----------


@register_function(
    name='capitalize',
    clickhouse_name='initcap',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['initcap'],
    doc='Capitalize first letter. Maps to initcap(s).',
)
def _build_capitalize(expr, alias=None):
    from .functions import Function

    return Function('initcap', expr, alias=alias)


@register_function(
    name='title',
    clickhouse_name='initcap',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Titlecase string. Maps to initcap(s).',
)
def _build_title(expr, alias=None):
    from .functions import Function

    return Function('initcap', expr, alias=alias)


@register_function(
    name='swapcase',
    clickhouse_name='swapcase',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Swap case. ClickHouse does not have native swapcase.',
    accessor_only=True,
)
def _build_swapcase(expr, alias=None):
    # ClickHouse doesn't have swapcase, use expression
    from .functions import Function

    return Function('swapcase', expr, alias=alias)


@register_function(
    name='center',
    clickhouse_name='center',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Center string in width. Maps to leftPad + rightPad combination.',
)
def _build_center(expr, width: int, fillchar: str = ' ', alias=None):
    from .functions import Function
    from .expressions import Literal

    # Use leftPad and rightPad to center
    return Function(
        'leftPad',
        Function('rightPad', expr, Literal(width), Literal(fillchar)),
        Literal(width),
        Literal(fillchar),
        alias=alias,
    )


@register_function(
    name='ljust',
    clickhouse_name='rightPad',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Left justify string. Maps to rightPad(s, width, fill).',
)
def _build_ljust(expr, width: int, fillchar: str = ' ', alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('rightPad', expr, Literal(width), Literal(fillchar), alias=alias)


@register_function(
    name='rjust',
    clickhouse_name='leftPad',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Right justify string. Maps to leftPad(s, width, fill).',
)
def _build_rjust(expr, width: int, fillchar: str = ' ', alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('leftPad', expr, Literal(width), Literal(fillchar), alias=alias)


@register_function(
    name='zfill',
    clickhouse_name='leftPad',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Pad with zeros on the left. Maps to leftPad(s, width, \'0\').',
)
def _build_zfill(expr, width: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('leftPad', expr, Literal(width), Literal('0'), alias=alias)


@register_function(
    name='find',
    clickhouse_name='position',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['index'],
    doc='Find substring position. Maps to position(s, sub) - returns 0-based index.',
)
def _build_find(expr, sub: str, start: int = 0, alias=None):
    from .functions import Function
    from .expressions import Literal

    if start > 0:
        return Function('position', expr, Literal(sub), Literal(start + 1), alias=alias)
    return Function('position', expr, Literal(sub), alias=alias)


@register_function(
    name='rfind',
    clickhouse_name='positionCaseInsensitiveUTF8',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['rindex'],
    doc='Find last occurrence of substring.',
)
def _build_rfind(expr, sub: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    # ClickHouse doesn't have rfind, use last occurrence logic
    return Function('position', Function('reverse', expr), Literal(sub[::-1]), alias=alias)


@register_function(
    name='count_substring',
    clickhouse_name='countSubstrings',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['countSubstrings'],
    doc='Count occurrences of substring. Maps to countSubstrings(s, sub).',
)
def _build_count_substring(expr, sub: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('countSubstrings', expr, Literal(sub), alias=alias)


@register_function(
    name='isalpha',
    clickhouse_name='match',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Check if all characters are alphabetic.',
)
def _build_isalpha(expr, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('match', expr, Literal('^[a-zA-Z]+$'), alias=alias)


@register_function(
    name='isdigit',
    clickhouse_name='match',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['isnumeric', 'isdecimal'],
    doc='Check if all characters are digits.',
)
def _build_isdigit(expr, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('match', expr, Literal('^[0-9]+$'), alias=alias)


@register_function(
    name='isalnum',
    clickhouse_name='match',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Check if all characters are alphanumeric.',
)
def _build_isalnum(expr, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('match', expr, Literal('^[a-zA-Z0-9]+$'), alias=alias)


@register_function(
    name='isspace',
    clickhouse_name='match',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Check if all characters are whitespace.',
)
def _build_isspace(expr, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('match', expr, Literal('^\\s+$'), alias=alias)


@register_function(
    name='isupper',
    clickhouse_name='equals',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Check if all characters are uppercase.',
)
def _build_isupper(expr, alias=None):
    from .functions import Function

    return Function('equals', expr, Function('upper', expr), alias=alias)


@register_function(
    name='islower',
    clickhouse_name='equals',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Check if all characters are lowercase.',
)
def _build_islower(expr, alias=None):
    from .functions import Function

    return Function('equals', expr, Function('lower', expr), alias=alias)


@register_function(
    name='match',
    clickhouse_name='match',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['regex_match'],
    doc='Match regex pattern. Maps to match(s, pattern).',
)
def _build_match(expr, pattern: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('match', expr, Literal(pattern), alias=alias)


@register_function(
    name='extract',
    clickhouse_name='extract',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['regex_extract'],
    doc='Extract regex pattern. Maps to extract(s, pattern).',
)
def _build_extract(expr, pattern: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('extract', expr, Literal(pattern), alias=alias)


@register_function(
    name='slice',
    clickhouse_name='substring',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Slice string. Maps to substring(s, start, length).',
)
def _build_slice(expr, start: int = None, stop: int = None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if start is None:
        start = 1
    else:
        start = start + 1 if start >= 0 else start
    if stop is None:
        return Function('substring', expr, Literal(start), alias=alias)
    length = stop - (start - 1) if start > 0 else stop
    return Function('substring', expr, Literal(start), Literal(length), alias=alias)


@register_function(
    name='encode',
    clickhouse_name='encodeURLComponent',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Encode string. Default to URL encoding.',
)
def _build_encode(expr, encoding: str = 'utf-8', alias=None):
    from .functions import Function

    return Function('encodeURLComponent', expr, alias=alias)


@register_function(
    name='decode',
    clickhouse_name='decodeURLComponent',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Decode string. Default from URL encoding.',
)
def _build_decode(expr, encoding: str = 'utf-8', alias=None):
    from .functions import Function

    return Function('decodeURLComponent', expr, alias=alias)


@register_function(
    name='wrap',
    clickhouse_name='wrapText',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Wrap text to specified width.',
    accessor_only=True,
)
def _build_wrap(expr, width: int, alias=None):
    # ClickHouse doesn't have text wrap, return as-is
    return expr


@register_function(
    name='join_str',
    clickhouse_name='arrayStringConcat',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['str_join'],
    doc='Join array elements with separator.',
)
def _build_join_str(expr, sep: str = '', alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('arrayStringConcat', expr, Literal(sep), alias=alias)


@register_function(
    name='normalize',
    clickhouse_name='normalizeUTF8NFD',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Normalize unicode string.',
)
def _build_normalize(expr, form: str = 'NFD', alias=None):
    from .functions import Function

    form_map = {
        'NFD': 'normalizeUTF8NFD',
        'NFC': 'normalizeUTF8NFC',
        'NFKD': 'normalizeUTF8NFKD',
        'NFKC': 'normalizeUTF8NFKC',
    }
    func_name = form_map.get(form.upper(), 'normalizeUTF8NFD')
    return Function(func_name, expr, alias=alias)


@register_function(
    name='removeprefix',
    clickhouse_name='trimLeft',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Remove prefix from string.',
)
def _build_removeprefix(expr, prefix: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    # if startsWith(s, prefix) then substring(s, length(prefix)+1) else s
    return Function(
        'if',
        Function('startsWith', expr, Literal(prefix)),
        Function('substring', expr, Literal(len(prefix) + 1)),
        expr,
        alias=alias,
    )


@register_function(
    name='removesuffix',
    clickhouse_name='trimRight',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Remove suffix from string.',
)
def _build_removesuffix(expr, suffix: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    # if endsWith(s, suffix) then substring(s, 1, length(s)-length(suffix)) else s
    return Function(
        'if',
        Function('endsWith', expr, Literal(suffix)),
        Function('substring', expr, Literal(1), Function('minus', Function('length', expr), Literal(len(suffix)))),
        expr,
        alias=alias,
    )


# =============================================================================
# DATETIME FUNCTIONS
# =============================================================================


@register_function(
    name='to_date',
    clickhouse_name='toDate',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toDate', 'as_date'],
    doc='Convert to Date type. Maps to toDate(x).',
)
def _build_to_date(expr, alias=None):
    from .functions import Function

    return Function('toDate', expr, alias=alias)


@register_function(
    name='to_datetime',
    clickhouse_name='toDateTime',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toDateTime', 'as_datetime', 'to_timestamp'],
    doc='Convert to DateTime type. Maps to toDateTime(x) or toDateTime(x, timezone).',
)
def _build_to_datetime(expr, timezone: str = None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if timezone:
        return Function('toDateTime', expr, Literal(timezone), alias=alias)
    return Function('toDateTime', expr, alias=alias)


@register_function(
    name='year',
    clickhouse_name='toYear',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toYear'],
    doc='Extract year from date/datetime. Maps to toYear(x).',
    accessor_only=True,  # Only via .dt accessor, not as Expression method
)
def _build_year(expr, alias=None):
    from .functions import Function

    return Function('toYear', expr, alias=alias)


@register_function(
    name='month',
    clickhouse_name='toMonth',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toMonth'],
    doc='Extract month from date/datetime (1-12). Maps to toMonth(x).',
    accessor_only=True,
)
def _build_month(expr, alias=None):
    from .functions import Function

    return Function('toMonth', expr, alias=alias)


@register_function(
    name='day',
    clickhouse_name='toDayOfMonth',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toDayOfMonth', 'day_of_month', 'dayOfMonth'],
    doc='Extract day of month from date/datetime (1-31). Maps to toDayOfMonth(x).',
    accessor_only=True,
)
def _build_day(expr, alias=None):
    from .functions import Function

    return Function('toDayOfMonth', expr, alias=alias)


@register_function(
    name='hour',
    clickhouse_name='toHour',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toHour'],
    doc='Extract hour from datetime (0-23). Maps to toHour(x).',
    accessor_only=True,
)
def _build_hour(expr, alias=None):
    from .functions import Function

    return Function('toHour', expr, alias=alias)


@register_function(
    name='minute',
    clickhouse_name='toMinute',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toMinute'],
    doc='Extract minute from datetime (0-59). Maps to toMinute(x).',
    accessor_only=True,
)
def _build_minute(expr, alias=None):
    from .functions import Function

    return Function('toMinute', expr, alias=alias)


@register_function(
    name='second',
    clickhouse_name='toSecond',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toSecond'],
    doc='Extract second from datetime (0-59). Maps to toSecond(x).',
    accessor_only=True,
)
def _build_second(expr, alias=None):
    from .functions import Function

    return Function('toSecond', expr, alias=alias)


@register_function(
    name='day_of_week',
    clickhouse_name='toDayOfWeek',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toDayOfWeek', 'dayOfWeek', 'weekday'],
    doc='Day of week (1=Monday, 7=Sunday). Maps to toDayOfWeek(x).',
    accessor_only=True,
)
def _build_day_of_week(expr, alias=None):
    from .functions import Function

    return Function('toDayOfWeek', expr, alias=alias)


@register_function(
    name='day_of_year',
    clickhouse_name='toDayOfYear',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toDayOfYear', 'dayOfYear'],
    doc='Day of year (1-365/366). Maps to toDayOfYear(x).',
    accessor_only=True,
)
def _build_day_of_year(expr, alias=None):
    from .functions import Function

    return Function('toDayOfYear', expr, alias=alias)


@register_function(
    name='quarter',
    clickhouse_name='toQuarter',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toQuarter'],
    doc='Quarter (1-4). Maps to toQuarter(x).',
    accessor_only=True,
)
def _build_quarter(expr, alias=None):
    from .functions import Function

    return Function('toQuarter', expr, alias=alias)


@register_function(
    name='week',
    clickhouse_name='toWeek',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toWeek', 'week_of_year', 'weekOfYear'],
    doc='Week number (1-52/53). Maps to toWeek(x).',
    accessor_only=True,
)
def _build_week(expr, mode: int = 0, alias=None):
    from .functions import Function
    from .expressions import Literal

    if mode:
        return Function('toWeek', expr, Literal(mode), alias=alias)
    return Function('toWeek', expr, alias=alias)


@register_function(
    name='date_diff',
    clickhouse_name='dateDiff',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['dateDiff', 'date_difference'],
    doc='Date difference. Maps to dateDiff(unit, start, end).',
)
def _build_date_diff(unit: str, start, end, alias=None):
    from .functions import Function
    from .expressions import Expression, Literal

    return Function('dateDiff', Literal(unit), Expression.wrap(start), Expression.wrap(end), alias=alias)


@register_function(
    name='date_add',
    clickhouse_name='date_add',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['dateAdd'],
    doc='Add interval to date. Maps to date_add(unit, interval, date).',
)
def _build_date_add(unit: str, interval: int, date, alias=None):
    from .functions import Function
    from .expressions import Expression, Literal

    return Function('date_add', Literal(unit), Literal(interval), Expression.wrap(date), alias=alias)


@register_function(
    name='date_trunc',
    clickhouse_name='date_trunc',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['dateTrunc', 'truncate'],
    doc='Truncate date to unit. Maps to date_trunc(unit, date).',
)
def _build_date_trunc(unit: str, date, alias=None):
    from .functions import Function
    from .expressions import Expression, Literal

    return Function('date_trunc', Literal(unit), Expression.wrap(date), alias=alias)


@register_function(
    name='now',
    clickhouse_name='now',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    doc='Current datetime. Maps to now().',
)
def _build_now(alias=None):
    from .functions import Function

    return Function('now', alias=alias)


@register_function(
    name='today',
    clickhouse_name='today',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    doc='Current date. Maps to today().',
)
def _build_today(alias=None):
    from .functions import Function

    return Function('today', alias=alias)


@register_function(
    name='yesterday',
    clickhouse_name='yesterday',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    doc="Yesterday's date. Maps to yesterday().",
)
def _build_yesterday(alias=None):
    from .functions import Function

    return Function('yesterday', alias=alias)


@register_function(
    name='to_start_of_day',
    clickhouse_name='toStartOfDay',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toStartOfDay'],
    doc='Round down to start of day. Maps to toStartOfDay(x).',
    accessor_only=True,
)
def _build_to_start_of_day(expr, alias=None):
    from .functions import Function

    return Function('toStartOfDay', expr, alias=alias)


@register_function(
    name='to_start_of_week',
    clickhouse_name='toStartOfWeek',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toStartOfWeek'],
    doc='Round down to start of week. Maps to toStartOfWeek(x).',
    accessor_only=True,
)
def _build_to_start_of_week(expr, mode: int = 0, alias=None):
    from .functions import Function
    from .expressions import Literal

    if mode:
        return Function('toStartOfWeek', expr, Literal(mode), alias=alias)
    return Function('toStartOfWeek', expr, alias=alias)


@register_function(
    name='to_start_of_month',
    clickhouse_name='toStartOfMonth',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toStartOfMonth'],
    doc='Round down to start of month. Maps to toStartOfMonth(x).',
    accessor_only=True,
)
def _build_to_start_of_month(expr, alias=None):
    from .functions import Function

    return Function('toStartOfMonth', expr, alias=alias)


@register_function(
    name='to_start_of_quarter',
    clickhouse_name='toStartOfQuarter',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toStartOfQuarter'],
    doc='Round down to start of quarter. Maps to toStartOfQuarter(x).',
    accessor_only=True,
)
def _build_to_start_of_quarter(expr, alias=None):
    from .functions import Function

    return Function('toStartOfQuarter', expr, alias=alias)


@register_function(
    name='to_start_of_year',
    clickhouse_name='toStartOfYear',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toStartOfYear'],
    doc='Round down to start of year. Maps to toStartOfYear(x).',
    accessor_only=True,
)
def _build_to_start_of_year(expr, alias=None):
    from .functions import Function

    return Function('toStartOfYear', expr, alias=alias)


# ---------- Additional Pandas .dt methods ----------


@register_function(
    name='to_start_of_hour',
    clickhouse_name='toStartOfHour',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toStartOfHour'],
    doc='Round down to start of hour.',
    accessor_only=True,
)
def _build_to_start_of_hour(expr, alias=None):
    from .functions import Function

    return Function('toStartOfHour', expr, alias=alias)


@register_function(
    name='to_start_of_minute',
    clickhouse_name='toStartOfMinute',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toStartOfMinute'],
    doc='Round down to start of minute.',
    accessor_only=True,
)
def _build_to_start_of_minute(expr, alias=None):
    from .functions import Function

    return Function('toStartOfMinute', expr, alias=alias)


@register_function(
    name='to_start_of_second',
    clickhouse_name='toStartOfSecond',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toStartOfSecond'],
    doc='Round down to start of second.',
    accessor_only=True,
)
def _build_to_start_of_second(expr, alias=None):
    from .functions import Function

    return Function('toStartOfSecond', expr, alias=alias)


@register_function(
    name='millisecond',
    clickhouse_name='toMillisecond',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toMillisecond'],
    doc='Extract millisecond (0-999).',
    accessor_only=True,
)
def _build_millisecond(expr, alias=None):
    from .functions import Function

    return Function('toMillisecond', expr, alias=alias)


@register_function(
    name='microsecond',
    clickhouse_name='toMicrosecond',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toMicrosecond'],
    doc='Extract microsecond.',
    accessor_only=True,
)
def _build_microsecond(expr, alias=None):
    from .functions import Function

    return Function('toMicrosecond', expr, alias=alias)


@register_function(
    name='nanosecond',
    clickhouse_name='toNanosecond',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toNanosecond'],
    doc='Extract nanosecond.',
    accessor_only=True,
)
def _build_nanosecond(expr, alias=None):
    from .functions import Function

    # ClickHouse DateTime64 supports nanoseconds
    return Function('toNanosecond', expr, alias=alias)


@register_function(
    name='date',
    clickhouse_name='toDate',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    doc='Extract date part from datetime.',
    accessor_only=True,
)
def _build_date_part(expr, alias=None):
    from .functions import Function

    return Function('toDate', expr, alias=alias)


@register_function(
    name='time',
    clickhouse_name='toTime',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toTime'],
    doc='Extract time part from datetime.',
    accessor_only=True,
)
def _build_time(expr, alias=None):
    from .functions import Function

    return Function('toTime', expr, alias=alias)


@register_function(
    name='is_month_start',
    clickhouse_name='equals',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    doc='Check if date is first day of month.',
    accessor_only=True,
)
def _build_is_month_start(expr, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('equals', Function('toDayOfMonth', expr), Literal(1), alias=alias)


@register_function(
    name='is_month_end',
    clickhouse_name='equals',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    doc='Check if date is last day of month.',
    accessor_only=True,
)
def _build_is_month_end(expr, alias=None):
    from .functions import Function

    return Function(
        'equals',
        Function('toDayOfMonth', expr),
        Function('toDayOfMonth', Function('toLastDayOfMonth', expr)),
        alias=alias,
    )


@register_function(
    name='is_quarter_start',
    clickhouse_name='equals',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    doc='Check if date is first day of quarter.',
    accessor_only=True,
)
def _build_is_quarter_start(expr, alias=None):
    from .functions import Function

    return Function('equals', expr, Function('toStartOfQuarter', expr), alias=alias)


@register_function(
    name='is_quarter_end',
    clickhouse_name='equals',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    doc='Check if date is last day of quarter.',
    accessor_only=True,
)
def _build_is_quarter_end(expr, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function(
        'equals',
        Function('toDate', expr),
        Function('subtractDays', Function('addMonths', Function('toStartOfQuarter', expr), Literal(3)), Literal(1)),
        alias=alias,
    )


@register_function(
    name='is_year_start',
    clickhouse_name='equals',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    doc='Check if date is first day of year.',
    accessor_only=True,
)
def _build_is_year_start(expr, alias=None):
    from .functions import Function

    return Function('equals', expr, Function('toStartOfYear', expr), alias=alias)


@register_function(
    name='is_year_end',
    clickhouse_name='equals',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    doc='Check if date is last day of year.',
    accessor_only=True,
)
def _build_is_year_end(expr, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function(
        'equals', Function('toDayOfYear', expr), Literal(365), alias=alias  # Simplified, doesn't handle leap years
    )


@register_function(
    name='is_leap_year',
    clickhouse_name='isLeapYear',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    doc='Check if year is a leap year.',
    accessor_only=True,
)
def _build_is_leap_year(expr, alias=None):
    from .functions import Function

    # ClickHouse doesn't have isLeapYear, calculate
    year_expr = Function('toYear', expr)
    from .expressions import Literal

    # Leap year: divisible by 4 AND (not divisible by 100 OR divisible by 400)
    return Function(
        'or',
        Function('equals', Function('modulo', year_expr, Literal(400)), Literal(0)),
        Function(
            'and',
            Function('equals', Function('modulo', year_expr, Literal(4)), Literal(0)),
            Function('notEquals', Function('modulo', year_expr, Literal(100)), Literal(0)),
        ),
        alias=alias,
    )


@register_function(
    name='days_in_month',
    clickhouse_name='toDayOfMonth',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['daysinmonth'],
    doc='Get number of days in month.',
    accessor_only=True,
)
def _build_days_in_month(expr, alias=None):
    from .functions import Function

    return Function('toDayOfMonth', Function('toLastDayOfMonth', expr), alias=alias)


@register_function(
    name='day_name',
    clickhouse_name='dateName',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    doc='Get name of day of week.',
)
def _build_day_name(expr, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('dateName', Literal('weekday'), expr, alias=alias)


@register_function(
    name='month_name',
    clickhouse_name='dateName',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    doc='Get name of month.',
)
def _build_month_name(expr, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('dateName', Literal('month'), expr, alias=alias)


@register_function(
    name='strftime',
    clickhouse_name='formatDateTime',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['formatDateTime', 'format_datetime'],
    doc='Format datetime as string. Maps to formatDateTime(dt, format).',
)
def _build_strftime(expr, date_format: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('formatDateTime', expr, Literal(date_format), alias=alias)


@register_function(
    name='tz_convert',
    clickhouse_name='toTimezone',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toTimezone'],
    doc='Convert to timezone. Maps to toTimezone(dt, tz).',
)
def _build_tz_convert(expr, tz: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('toTimezone', expr, Literal(tz), alias=alias)


@register_function(
    name='tz_localize',
    clickhouse_name='toTimezone',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    doc='Localize to timezone.',
)
def _build_tz_localize(expr, tz: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('toTimezone', expr, Literal(tz), alias=alias)


@register_function(
    name='floor_dt',
    clickhouse_name='toStartOfInterval',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    doc='Floor datetime to interval.',
)
def _build_floor_dt(expr, freq: str = 'D', alias=None):
    from .functions import Function

    freq_map = {
        'D': 'toStartOfDay',
        'H': 'toStartOfHour',
        'M': 'toStartOfMonth',
        'W': 'toStartOfWeek',
        'Q': 'toStartOfQuarter',
        'Y': 'toStartOfYear',
    }
    func_name = freq_map.get(freq.upper(), 'toStartOfDay')
    return Function(func_name, expr, alias=alias)


@register_function(
    name='ceil_dt',
    clickhouse_name='dateCeil',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    doc='Ceil datetime to interval.',
)
def _build_ceil_dt(expr, freq: str = 'D', alias=None):
    from .functions import Function
    from .expressions import Literal

    # ClickHouse doesn't have direct ceil, use floor + add interval
    return Function('addDays', Function('toStartOfDay', expr), Literal(1), alias=alias)


@register_function(
    name='total_seconds',
    clickhouse_name='toUnixTimestamp',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toUnixTimestamp', 'timestamp'],
    doc='Convert to Unix timestamp (seconds). Maps to toUnixTimestamp(dt).',
)
def _build_total_seconds(expr, alias=None):
    from .functions import Function

    return Function('toUnixTimestamp', expr, alias=alias)


@register_function(
    name='add_years',
    clickhouse_name='addYears',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['addYears'],
    doc='Add years to date/datetime.',
)
def _build_add_years(expr, years: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('addYears', expr, Literal(years), alias=alias)


@register_function(
    name='add_months',
    clickhouse_name='addMonths',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['addMonths'],
    doc='Add months to date/datetime.',
)
def _build_add_months(expr, months: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('addMonths', expr, Literal(months), alias=alias)


@register_function(
    name='add_weeks',
    clickhouse_name='addWeeks',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['addWeeks'],
    doc='Add weeks to date/datetime.',
)
def _build_add_weeks(expr, weeks: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('addWeeks', expr, Literal(weeks), alias=alias)


@register_function(
    name='add_days',
    clickhouse_name='addDays',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['addDays'],
    doc='Add days to date/datetime.',
)
def _build_add_days(expr, days: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('addDays', expr, Literal(days), alias=alias)


@register_function(
    name='add_hours',
    clickhouse_name='addHours',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['addHours'],
    doc='Add hours to datetime.',
)
def _build_add_hours(expr, hours: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('addHours', expr, Literal(hours), alias=alias)


@register_function(
    name='add_minutes',
    clickhouse_name='addMinutes',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['addMinutes'],
    doc='Add minutes to datetime.',
)
def _build_add_minutes(expr, minutes: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('addMinutes', expr, Literal(minutes), alias=alias)


@register_function(
    name='add_seconds',
    clickhouse_name='addSeconds',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['addSeconds'],
    doc='Add seconds to datetime.',
)
def _build_add_seconds(expr, seconds: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('addSeconds', expr, Literal(seconds), alias=alias)


@register_function(
    name='subtract_years',
    clickhouse_name='subtractYears',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['subtractYears'],
    doc='Subtract years from date/datetime.',
)
def _build_subtract_years(expr, years: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('subtractYears', expr, Literal(years), alias=alias)


@register_function(
    name='subtract_months',
    clickhouse_name='subtractMonths',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['subtractMonths'],
    doc='Subtract months from date/datetime.',
)
def _build_subtract_months(expr, months: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('subtractMonths', expr, Literal(months), alias=alias)


@register_function(
    name='subtract_days',
    clickhouse_name='subtractDays',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['subtractDays'],
    doc='Subtract days from date/datetime.',
)
def _build_subtract_days(expr, days: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('subtractDays', expr, Literal(days), alias=alias)


# =============================================================================
# MATH FUNCTIONS
# =============================================================================


@register_function(
    name='abs',
    clickhouse_name='abs',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Absolute value. Maps to abs(x).',
)
def _build_abs(expr, alias=None):
    from .functions import Function

    return Function('abs', expr, alias=alias)


@register_function(
    name='round',
    clickhouse_name='round',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Round to N decimal places. Maps to round(x, N).',
)
def _build_round(expr, precision: int = 0, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('round', expr, Literal(precision), alias=alias)


@register_function(
    name='floor',
    clickhouse_name='floor',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Round down. Maps to floor(x).',
)
def _build_floor(expr, alias=None):
    from .functions import Function

    return Function('floor', expr, alias=alias)


@register_function(
    name='ceil',
    clickhouse_name='ceiling',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['ceiling'],
    doc='Round up. Maps to ceiling(x).',
)
def _build_ceil(expr, alias=None):
    from .functions import Function

    return Function('ceiling', expr, alias=alias)


@register_function(
    name='sqrt',
    clickhouse_name='sqrt',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Square root. Maps to sqrt(x).',
)
def _build_sqrt(expr, alias=None):
    from .functions import Function

    return Function('sqrt', expr, alias=alias)


@register_function(
    name='pow',
    clickhouse_name='pow',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['power'],
    doc='Power. Maps to pow(base, exponent).',
)
def _build_pow(base, exponent, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('pow', Expression.wrap(base), Expression.wrap(exponent), alias=alias)


@register_function(
    name='exp',
    clickhouse_name='exp',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Exponential (e^x). Maps to exp(x).',
)
def _build_exp(expr, alias=None):
    from .functions import Function

    return Function('exp', expr, alias=alias)


@register_function(
    name='log',
    clickhouse_name='log',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['ln'],
    doc='Natural logarithm. Maps to log(x).',
)
def _build_log(expr, base: float = None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if base is not None:
        return Function('log', Literal(base), expr, alias=alias)
    return Function('log', expr, alias=alias)


@register_function(
    name='log10',
    clickhouse_name='log10',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Base-10 logarithm. Maps to log10(x).',
)
def _build_log10(expr, alias=None):
    from .functions import Function

    return Function('log10', expr, alias=alias)


@register_function(
    name='log2',
    clickhouse_name='log2',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Base-2 logarithm. Maps to log2(x).',
)
def _build_log2(expr, alias=None):
    from .functions import Function

    return Function('log2', expr, alias=alias)


@register_function(
    name='sin',
    clickhouse_name='sin',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Sine. Maps to sin(x).',
)
def _build_sin(expr, alias=None):
    from .functions import Function

    return Function('sin', expr, alias=alias)


@register_function(
    name='cos',
    clickhouse_name='cos',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Cosine. Maps to cos(x).',
)
def _build_cos(expr, alias=None):
    from .functions import Function

    return Function('cos', expr, alias=alias)


@register_function(
    name='tan',
    clickhouse_name='tan',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Tangent. Maps to tan(x).',
)
def _build_tan(expr, alias=None):
    from .functions import Function

    return Function('tan', expr, alias=alias)


@register_function(
    name='sign',
    clickhouse_name='sign',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Sign (-1, 0, 1). Maps to sign(x).',
)
def _build_sign(expr, alias=None):
    from .functions import Function

    return Function('sign', expr, alias=alias)


@register_function(
    name='mod',
    clickhouse_name='modulo',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['modulo'],
    doc='Modulo. Maps to modulo(a, b).',
)
def _build_mod(a, b, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('modulo', Expression.wrap(a), Expression.wrap(b), alias=alias)


# =============================================================================
# CONDITIONAL FUNCTIONS
# =============================================================================


@register_function(
    name='if_',
    clickhouse_name='if',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['if_then_else', 'iff'],
    doc='Conditional expression. Maps to if(condition, then, else).',
)
def _build_if(condition, then_value, else_value, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function(
        'if', Expression.wrap(condition), Expression.wrap(then_value), Expression.wrap(else_value), alias=alias
    )


@register_function(
    name='coalesce',
    clickhouse_name='coalesce',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    doc='Return first non-NULL value. Maps to coalesce(...).',
    min_args=1,
    max_args=-1,
)
def _build_coalesce(*args, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('coalesce', *[Expression.wrap(a) for a in args], alias=alias)


@register_function(
    name='if_null',
    clickhouse_name='ifNull',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['ifNull', 'nvl'],
    doc='Return default if expr is NULL. Maps to ifNull(x, default).',
)
def _build_if_null(expr, default, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('ifNull', expr, Expression.wrap(default), alias=alias)


@register_function(
    name='null_if',
    clickhouse_name='nullIf',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['nullIf'],
    doc='Return NULL if expr equals value. Maps to nullIf(x, value).',
)
def _build_null_if(expr, value, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('nullIf', expr, Expression.wrap(value), alias=alias)


@register_function(
    name='multi_if',
    clickhouse_name='multiIf',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['multiIf', 'case_when'],
    doc='Multiple conditions (CASE WHEN equivalent). Maps to multiIf(cond1, then1, ..., else).',
    min_args=3,
    max_args=-1,
)
def _build_multi_if(*args, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('multiIf', *[Expression.wrap(a) for a in args], alias=alias)


# =============================================================================
# TYPE CONVERSION FUNCTIONS
# =============================================================================


@register_function(
    name='to_string',
    clickhouse_name='toString',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toString', 'str'],
    doc='Convert to String. Maps to toString(x).',
)
def _build_to_string(expr, alias=None):
    from .functions import Function

    return Function('toString', expr, alias=alias)


@register_function(
    name='to_int8',
    clickhouse_name='toInt8',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toInt8'],
    doc='Convert to Int8. Maps to toInt8(x).',
)
def _build_to_int8(expr, alias=None):
    from .functions import Function

    return Function('toInt8', expr, alias=alias)


@register_function(
    name='to_int16',
    clickhouse_name='toInt16',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toInt16'],
    doc='Convert to Int16. Maps to toInt16(x).',
)
def _build_to_int16(expr, alias=None):
    from .functions import Function

    return Function('toInt16', expr, alias=alias)


@register_function(
    name='to_int32',
    clickhouse_name='toInt32',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toInt32'],
    doc='Convert to Int32. Maps to toInt32(x).',
)
def _build_to_int32(expr, alias=None):
    from .functions import Function

    return Function('toInt32', expr, alias=alias)


@register_function(
    name='to_int64',
    clickhouse_name='toInt64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toInt64'],
    doc='Convert to Int64. Maps to toInt64(x).',
)
def _build_to_int64(expr, alias=None):
    from .functions import Function

    return Function('toInt64', expr, alias=alias)


@register_function(
    name='to_float32',
    clickhouse_name='toFloat32',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toFloat32'],
    doc='Convert to Float32. Maps to toFloat32(x).',
)
def _build_to_float32(expr, alias=None):
    from .functions import Function

    return Function('toFloat32', expr, alias=alias)


@register_function(
    name='to_float64',
    clickhouse_name='toFloat64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toFloat64'],
    doc='Convert to Float64. Maps to toFloat64(x).',
)
def _build_to_float64(expr, alias=None):
    from .functions import Function

    return Function('toFloat64', expr, alias=alias)


# =============================================================================
# AGGREGATE FUNCTIONS
# =============================================================================


@register_function(
    name='sum',
    clickhouse_name='sum',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['total'],
    doc='Sum aggregate. Maps to sum(x).',
)
def _build_sum(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('sum', expr, alias=alias)


@register_function(
    name='avg',
    clickhouse_name='avg',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['mean', 'average'],
    doc='Average aggregate. Maps to avg(x).',
)
def _build_avg(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('avg', expr, alias=alias)


@register_function(
    name='count',
    clickhouse_name='count',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['cnt'],
    doc='Count aggregate. Maps to count(x) or count(*).',
)
def _build_count(expr='*', alias=None):
    from .functions import AggregateFunction
    from .expressions import Expression, Literal

    if expr == '*':
        return AggregateFunction('count', Literal('*'), alias=alias)
    return AggregateFunction('count', Expression.wrap(expr), alias=alias)


@register_function(
    name='max',
    clickhouse_name='max',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['maximum'],
    doc='Maximum aggregate. Maps to max(x).',
)
def _build_max(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('max', expr, alias=alias)


@register_function(
    name='min',
    clickhouse_name='min',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['minimum'],
    doc='Minimum aggregate. Maps to min(x).',
)
def _build_min(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('min', expr, alias=alias)


@register_function(
    name='count_distinct',
    clickhouse_name='uniq',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['uniq', 'nunique', 'distinct_count'],
    doc='Count distinct values. Maps to uniq(x).',
)
def _build_count_distinct(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('uniq', expr, alias=alias)


@register_function(
    name='stddev',
    clickhouse_name='stddevPop',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['stddevPop', 'stddev_pop'],
    doc='Standard deviation (population). Maps to stddevPop(x).',
)
def _build_stddev(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('stddevPop', expr, alias=alias)


@register_function(
    name='stddev_samp',
    clickhouse_name='stddevSamp',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['stddevSamp', 'std'],
    doc='Standard deviation (sample). Maps to stddevSamp(x).',
)
def _build_stddev_samp(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('stddevSamp', expr, alias=alias)


@register_function(
    name='variance',
    clickhouse_name='varPop',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['varPop', 'var', 'var_pop'],
    doc='Variance (population). Maps to varPop(x).',
)
def _build_variance(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('varPop', expr, alias=alias)


@register_function(
    name='var_samp',
    clickhouse_name='varSamp',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['varSamp'],
    doc='Variance (sample). Maps to varSamp(x).',
)
def _build_var_samp(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('varSamp', expr, alias=alias)


@register_function(
    name='median',
    clickhouse_name='median',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    doc='Median (50th percentile). Maps to median(x).',
)
def _build_median(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('median', expr, alias=alias)


@register_function(
    name='group_array',
    clickhouse_name='groupArray',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['groupArray', 'collect_list'],
    doc='Collect values into array. Maps to groupArray(x).',
)
def _build_group_array(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('groupArray', expr, alias=alias)


@register_function(
    name='group_uniq_array',
    clickhouse_name='groupUniqArray',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['groupUniqArray', 'collect_set'],
    doc='Collect unique values into array. Maps to groupUniqArray(x).',
)
def _build_group_uniq_array(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('groupUniqArray', expr, alias=alias)


@register_function(
    name='any_value',
    clickhouse_name='any',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['any', 'first'],
    doc='Any value from the group. Maps to any(x).',
)
def _build_any_value(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('any', expr, alias=alias)


@register_function(
    name='any_last',
    clickhouse_name='anyLast',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['anyLast', 'last'],
    doc='Last encountered value. Maps to anyLast(x).',
)
def _build_any_last(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('anyLast', expr, alias=alias)


# ---------- Additional Pandas aggregation/statistics methods ----------


@register_function(
    name='prod',
    clickhouse_name='product',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['product'],
    doc='Product of values. Maps to product(x).',
)
def _build_prod(expr, alias=None):
    from .functions import AggregateFunction

    # ClickHouse uses product() or we can use exp(sum(log(x)))
    return AggregateFunction('product', expr, alias=alias)


@register_function(
    name='all_true',
    clickhouse_name='min',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['all'],
    doc='Check if all values are true. Maps to min(x) for boolean.',
)
def _build_all_true(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('min', expr, alias=alias)


@register_function(
    name='skew',
    clickhouse_name='skewPop',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['skewPop', 'skewness'],
    doc='Skewness of distribution. Maps to skewPop(x).',
)
def _build_skew(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('skewPop', expr, alias=alias)


@register_function(
    name='kurt',
    clickhouse_name='kurtPop',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['kurtPop', 'kurtosis'],
    doc='Kurtosis of distribution. Maps to kurtPop(x).',
)
def _build_kurt(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('kurtPop', expr, alias=alias)


@register_function(
    name='corr',
    clickhouse_name='corr',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['correlation'],
    doc='Correlation between two columns. Maps to corr(x, y).',
)
def _build_corr(expr, other, alias=None):
    from .functions import AggregateFunction
    from .expressions import Expression

    return AggregateFunction('corr', expr, Expression.wrap(other), alias=alias)


@register_function(
    name='cov',
    clickhouse_name='covarPop',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['covarPop', 'covariance'],
    doc='Covariance between two columns. Maps to covarPop(x, y).',
)
def _build_cov(expr, other, alias=None):
    from .functions import AggregateFunction
    from .expressions import Expression

    return AggregateFunction('covarPop', expr, Expression.wrap(other), alias=alias)


@register_function(
    name='mode',
    clickhouse_name='topK',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    doc='Most frequent value. Maps to topK(1)(x).',
)
def _build_mode(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('topK(1)', expr, alias=alias)


@register_function(
    name='sem',
    clickhouse_name='stddevPop',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    doc='Standard error of mean. Maps to stddevPop(x)/sqrt(count(x)).',
)
def _build_sem(expr, alias=None):
    from .functions import AggregateFunction, Function

    # SEM = stddev / sqrt(n)
    return Function(
        'divide', AggregateFunction('stddevPop', expr), Function('sqrt', AggregateFunction('count', expr)), alias=alias
    )


@register_function(
    name='cumsum',
    clickhouse_name='runningAccumulate',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    aliases=['runningSum'],
    doc='Cumulative sum. Use with window function.',
    supports_over=True,
)
def _build_cumsum(expr, alias=None):
    from .functions import WindowFunction

    return WindowFunction('sum', expr, alias=alias)


@register_function(
    name='cummax',
    clickhouse_name='runningAccumulate',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    doc='Cumulative max. Use with window function.',
    supports_over=True,
)
def _build_cummax(expr, alias=None):
    from .functions import WindowFunction

    return WindowFunction('max', expr, alias=alias)


@register_function(
    name='cummin',
    clickhouse_name='runningAccumulate',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    doc='Cumulative min. Use with window function.',
    supports_over=True,
)
def _build_cummin(expr, alias=None):
    from .functions import WindowFunction

    return WindowFunction('min', expr, alias=alias)


@register_function(
    name='diff',
    clickhouse_name='lagInFrame',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    doc='Difference with previous value. Maps to x - lagInFrame(x, 1).',
    supports_over=True,
)
def _build_diff(expr, periods: int = 1, alias=None):
    from .functions import Function, WindowFunction
    from .expressions import Literal

    lag = WindowFunction('lagInFrame', expr, Literal(periods))
    return Function('minus', expr, lag, alias=alias)


@register_function(
    name='pct_change',
    clickhouse_name='lagInFrame',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    doc='Percentage change from previous value.',
    supports_over=True,
)
def _build_pct_change(expr, periods: int = 1, alias=None):
    from .functions import Function, WindowFunction
    from .expressions import Literal

    lag = WindowFunction('lagInFrame', expr, Literal(periods))
    # (x - lag) / lag
    return Function('divide', Function('minus', expr, lag), lag, alias=alias)


@register_function(
    name='shift',
    clickhouse_name='lagInFrame',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    doc='Shift values by periods. Positive=lag, negative=lead.',
    supports_over=True,
)
def _build_shift(expr, periods: int = 1, alias=None):
    from .functions import WindowFunction
    from .expressions import Literal

    if periods >= 0:
        return WindowFunction('lagInFrame', expr, Literal(periods), alias=alias)
    else:
        return WindowFunction('leadInFrame', expr, Literal(-periods), alias=alias)


@register_function(
    name='clip',
    clickhouse_name='greatest',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Clip values to range. Maps to greatest(lower, least(upper, x)).',
)
def _build_clip(expr, lower=None, upper=None, alias=None):
    from .functions import Function
    from .expressions import Literal

    result = expr
    if lower is not None:
        result = Function('greatest', Literal(lower), result)
    if upper is not None:
        result = Function('least', Literal(upper), result)
    if alias:
        # Add alias handling
        result.alias = alias
    return result


@register_function(
    name='fillna',
    clickhouse_name='ifNull',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['fill_null'],
    doc='Fill NULL values. Maps to ifNull(x, value).',
)
def _build_fillna(expr, value, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('ifNull', expr, Expression.wrap(value), alias=alias)


@register_function(
    name='isna',
    clickhouse_name='isNull',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['isnull'],
    doc='Check if NULL. Maps to toBool(isNull(x)) for pandas bool compatibility.',
)
def _build_isna(expr, alias=None):
    from .functions import Function

    # Wrap with toBool() to return bool dtype instead of uint8
    # This ensures pandas compatibility (pandas isna() returns bool)
    return Function('toBool', Function('isNull', expr), alias=alias)


@register_function(
    name='notna',
    clickhouse_name='isNotNull',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['notnull'],
    doc='Check if not NULL. Maps to toBool(isNotNull(x)) for pandas bool compatibility.',
)
def _build_notna(expr, alias=None):
    from .functions import Function

    # Wrap with toBool() to return bool dtype instead of uint8
    # This ensures pandas compatibility (pandas notna() returns bool)
    return Function('toBool', Function('isNotNull', expr), alias=alias)


@register_function(
    name='where_expr',
    clickhouse_name='if',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    doc='Where condition is true, return self, else other. Maps to if(cond, x, other).',
)
def _build_where_expr(expr, cond, other=None, alias=None):
    from .functions import Function
    from .expressions import Expression, Literal

    if other is None:
        other = Literal(None)
    return Function('if', Expression.wrap(cond), expr, Expression.wrap(other), alias=alias)


@register_function(
    name='mask',
    clickhouse_name='if',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    doc='Where condition is true, return other, else self. Maps to if(cond, other, x).',
)
def _build_mask(expr, cond, other=None, alias=None):
    from .functions import Function
    from .expressions import Expression, Literal

    if other is None:
        other = Literal(None)
    return Function('if', Expression.wrap(cond), Expression.wrap(other), expr, alias=alias)


@register_function(
    name='argmin',
    clickhouse_name='argMin',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['argMin', 'idxmin'],
    doc='Value of first arg where second arg is minimum. Maps to argMin(arg, val).',
)
def _build_argmin(expr, val, alias=None):
    from .functions import AggregateFunction
    from .expressions import Expression

    return AggregateFunction('argMin', expr, Expression.wrap(val), alias=alias)


@register_function(
    name='argmax',
    clickhouse_name='argMax',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['argMax', 'idxmax'],
    doc='Value of first arg where second arg is maximum. Maps to argMax(arg, val).',
)
def _build_argmax(expr, val, alias=None):
    from .functions import AggregateFunction
    from .expressions import Expression

    return AggregateFunction('argMax', expr, Expression.wrap(val), alias=alias)


# =============================================================================
# WINDOW FUNCTIONS
# =============================================================================


@register_function(
    name='row_number',
    clickhouse_name='row_number',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    aliases=['rowNumber'],
    doc='Assign unique row number within partition. Requires OVER clause.',
    supports_over=True,
)
def _build_row_number(alias=None):
    from .functions import WindowFunction

    return WindowFunction('row_number', alias=alias)


@register_function(
    name='rank',
    clickhouse_name='rank',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    doc='Rank within partition with gaps. Requires OVER clause.',
    supports_over=True,
)
def _build_rank(alias=None):
    from .functions import WindowFunction

    return WindowFunction('rank', alias=alias)


@register_function(
    name='dense_rank',
    clickhouse_name='dense_rank',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    aliases=['denseRank'],
    doc='Rank within partition without gaps. Requires OVER clause.',
    supports_over=True,
)
def _build_dense_rank(alias=None):
    from .functions import WindowFunction

    return WindowFunction('dense_rank', alias=alias)


@register_function(
    name='ntile',
    clickhouse_name='ntile',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    doc='Divide partition into N buckets. Requires OVER clause.',
    supports_over=True,
)
def _build_ntile(n: int, alias=None):
    from .functions import WindowFunction
    from .expressions import Literal

    return WindowFunction('ntile', Literal(n), alias=alias)


@register_function(
    name='lead',
    clickhouse_name='leadInFrame',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    aliases=['leadInFrame'],
    doc='Access following row value. Requires OVER clause.',
    supports_over=True,
)
def _build_lead(expr, offset: int = 1, default=None, alias=None):
    from .functions import WindowFunction
    from .expressions import Expression, Literal

    args = [Expression.wrap(expr), Literal(offset)]
    if default is not None:
        args.append(Expression.wrap(default))
    return WindowFunction('leadInFrame', *args, alias=alias)


@register_function(
    name='lag',
    clickhouse_name='lagInFrame',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    aliases=['lagInFrame'],
    doc='Access preceding row value. Requires OVER clause.',
    supports_over=True,
)
def _build_lag(expr, offset: int = 1, default=None, alias=None):
    from .functions import WindowFunction
    from .expressions import Expression, Literal

    args = [Expression.wrap(expr), Literal(offset)]
    if default is not None:
        args.append(Expression.wrap(default))
    return WindowFunction('lagInFrame', *args, alias=alias)


@register_function(
    name='first_value',
    clickhouse_name='first_value',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    aliases=['firstValue', 'first_value_respect_nulls'],
    doc='First value in window frame. Requires OVER clause.',
    supports_over=True,
)
def _build_first_value(expr, alias=None):
    from .functions import WindowFunction
    from .expressions import Expression

    return WindowFunction('first_value', Expression.wrap(expr), alias=alias)


@register_function(
    name='last_value',
    clickhouse_name='last_value',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    aliases=['lastValue', 'last_value_respect_nulls'],
    doc='Last value in window frame. Requires OVER clause.',
    supports_over=True,
)
def _build_last_value(expr, alias=None):
    from .functions import WindowFunction
    from .expressions import Expression

    return WindowFunction('last_value', Expression.wrap(expr), alias=alias)


# =============================================================================
# HASH FUNCTIONS
# =============================================================================


@register_function(
    name='md5',
    clickhouse_name='MD5',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['MD5'],
    doc='MD5 hash. Maps to MD5(x).',
)
def _build_md5(expr, alias=None):
    from .functions import Function

    return Function('MD5', expr, alias=alias)


@register_function(
    name='sha256',
    clickhouse_name='SHA256',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['SHA256'],
    doc='SHA256 hash. Maps to SHA256(x).',
)
def _build_sha256(expr, alias=None):
    from .functions import Function

    return Function('SHA256', expr, alias=alias)


@register_function(
    name='city_hash64',
    clickhouse_name='cityHash64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['cityHash64'],
    doc='CityHash64 (fast non-crypto hash). Maps to cityHash64(x).',
)
def _build_city_hash64(expr, alias=None):
    from .functions import Function

    return Function('cityHash64', expr, alias=alias)


@register_function(
    name='sip_hash64',
    clickhouse_name='sipHash64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['sipHash64'],
    doc='SipHash64. Maps to sipHash64(x).',
)
def _build_sip_hash64(expr, alias=None):
    from .functions import Function

    return Function('sipHash64', expr, alias=alias)


# =============================================================================
# ARRAY FUNCTIONS
# =============================================================================


@register_function(
    name='array',
    clickhouse_name='array',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    doc='Create array. Maps to array(...).',
    min_args=0,
    max_args=-1,
)
def _build_array(*args, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('array', *[Expression.wrap(a) for a in args], alias=alias)


@register_function(
    name='array_join',
    clickhouse_name='arrayJoin',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayJoin', 'explode'],
    doc='Expand array to rows. Maps to arrayJoin(arr).',
)
def _build_array_join(expr, alias=None):
    from .functions import Function

    return Function('arrayJoin', expr, alias=alias)


@register_function(
    name='has',
    clickhouse_name='has',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['array_contains'],
    doc='Check if array contains element. Maps to has(arr, elem).',
)
def _build_has(arr, elem, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('has', Expression.wrap(arr), Expression.wrap(elem), alias=alias)


@register_function(
    name='array_length',
    clickhouse_name='length',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayLength', 'size'],
    doc='Array length. Maps to length(arr).',
)
def _build_array_length(expr, alias=None):
    from .functions import Function

    return Function('length', expr, alias=alias)


# =============================================================================
# JSON FUNCTIONS
# =============================================================================


def _build_json_path_args(path: str):
    """
    Convert a dot-separated JSON path into separate arguments for ClickHouse JSON functions.

    ClickHouse JSONExtract* functions use separate arguments for nested paths:
    - 'name' -> Literal('name')
    - 'user.name' -> Literal('user'), Literal('name')

    Args:
        path: Dot-separated JSON path

    Returns:
        List of Literal expressions
    """
    from .expressions import Literal

    parts = path.split('.')
    return [Literal(part) for part in parts]


@register_function(
    name='json_extract_string',
    clickhouse_name='JSONExtractString',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractString'],
    doc='Extract string from JSON. Supports nested paths like "user.name".',
)
def _build_json_extract_string(json, path: str, alias=None):
    from .functions import Function

    path_args = _build_json_path_args(path)
    return Function('JSONExtractString', json, *path_args, alias=alias)


@register_function(
    name='json_extract_int',
    clickhouse_name='JSONExtractInt',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractInt'],
    doc='Extract integer from JSON. Supports nested paths like "user.id".',
)
def _build_json_extract_int(json, path: str, alias=None):
    from .functions import Function

    path_args = _build_json_path_args(path)
    return Function('JSONExtractInt', json, *path_args, alias=alias)


@register_function(
    name='json_extract_float',
    clickhouse_name='JSONExtractFloat',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractFloat'],
    doc='Extract float from JSON. Supports nested paths like "data.price".',
)
def _build_json_extract_float(json, path: str, alias=None):
    from .functions import Function

    path_args = _build_json_path_args(path)
    return Function('JSONExtractFloat', json, *path_args, alias=alias)


@register_function(
    name='json_extract_bool',
    clickhouse_name='JSONExtractBool',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractBool'],
    doc='Extract boolean from JSON. Supports nested paths like "user.active".',
)
def _build_json_extract_bool(json, path: str, alias=None):
    from .functions import Function

    path_args = _build_json_path_args(path)
    return Function('JSONExtractBool', json, *path_args, alias=alias)


# =============================================================================
# MORE ARRAY FUNCTIONS
# =============================================================================


@register_function(
    name='array_sum',
    clickhouse_name='arraySum',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arraySum'],
    doc='Sum of array elements. Maps to arraySum(arr).',
)
def _build_array_sum(expr, alias=None):
    from .functions import Function

    return Function('arraySum', expr, alias=alias)


@register_function(
    name='array_avg',
    clickhouse_name='arrayAvg',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayAvg'],
    doc='Average of array elements. Maps to arrayAvg(arr).',
)
def _build_array_avg(expr, alias=None):
    from .functions import Function

    return Function('arrayAvg', expr, alias=alias)


@register_function(
    name='array_min',
    clickhouse_name='arrayMin',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayMin'],
    doc='Minimum of array elements. Maps to arrayMin(arr).',
)
def _build_array_min(expr, alias=None):
    from .functions import Function

    return Function('arrayMin', expr, alias=alias)


@register_function(
    name='array_max',
    clickhouse_name='arrayMax',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayMax'],
    doc='Maximum of array elements. Maps to arrayMax(arr).',
)
def _build_array_max(expr, alias=None):
    from .functions import Function

    return Function('arrayMax', expr, alias=alias)


@register_function(
    name='array_count',
    clickhouse_name='arrayCount',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayCount'],
    doc='Count elements matching condition. Maps to arrayCount(arr) or arrayCount(lambda, arr).',
)
def _build_array_count(expr, alias=None):
    from .functions import Function

    return Function('arrayCount', expr, alias=alias)


@register_function(
    name='array_distinct',
    clickhouse_name='arrayDistinct',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayDistinct', 'unique'],
    doc='Get unique elements. Maps to arrayDistinct(arr).',
)
def _build_array_distinct(expr, alias=None):
    from .functions import Function

    return Function('arrayDistinct', expr, alias=alias)


@register_function(
    name='array_sort',
    clickhouse_name='arraySort',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arraySort'],
    doc='Sort array. Maps to arraySort(arr).',
)
def _build_array_sort(expr, alias=None):
    from .functions import Function

    return Function('arraySort', expr, alias=alias)


@register_function(
    name='array_reverse_sort',
    clickhouse_name='arrayReverseSort',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayReverseSort'],
    doc='Sort array in descending order. Maps to arrayReverseSort(arr).',
)
def _build_array_reverse_sort(expr, alias=None):
    from .functions import Function

    return Function('arrayReverseSort', expr, alias=alias)


@register_function(
    name='array_reverse',
    clickhouse_name='arrayReverse',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayReverse'],
    doc='Reverse array. Maps to arrayReverse(arr).',
)
def _build_array_reverse(expr, alias=None):
    from .functions import Function

    return Function('arrayReverse', expr, alias=alias)


@register_function(
    name='array_first',
    clickhouse_name='arrayElement',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['first_element'],
    doc='Get first element. Maps to arrayElement(arr, 1).',
)
def _build_array_first(expr, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('arrayElement', expr, Literal(1), alias=alias)


@register_function(
    name='array_last',
    clickhouse_name='arrayElement',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['last_element'],
    doc='Get last element. Maps to arrayElement(arr, -1).',
)
def _build_array_last(expr, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('arrayElement', expr, Literal(-1), alias=alias)


@register_function(
    name='array_element',
    clickhouse_name='arrayElement',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayElement'],  # Removed 'get' to avoid conflict with pandas Series.get()
    doc='Get element at index. Maps to arrayElement(arr, index).',
)
def _build_array_element(expr, index: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('arrayElement', expr, Literal(index), alias=alias)


@register_function(
    name='array_slice',
    clickhouse_name='arraySlice',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arraySlice'],
    doc='Slice array. Maps to arraySlice(arr, offset, length).',
)
def _build_array_slice(expr, offset: int, length: int = None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if length is not None:
        return Function('arraySlice', expr, Literal(offset), Literal(length), alias=alias)
    return Function('arraySlice', expr, Literal(offset), alias=alias)


@register_function(
    name='array_concat',
    clickhouse_name='arrayConcat',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayConcat'],
    doc='Concatenate arrays. Maps to arrayConcat(arr1, arr2, ...).',
    min_args=1,
    max_args=-1,
)
def _build_array_concat(*args, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('arrayConcat', *[Expression.wrap(a) for a in args], alias=alias)


@register_function(
    name='array_flatten',
    clickhouse_name='arrayFlatten',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayFlatten', 'flatten'],
    doc='Flatten nested array. Maps to arrayFlatten(arr).',
)
def _build_array_flatten(expr, alias=None):
    from .functions import Function

    return Function('arrayFlatten', expr, alias=alias)


@register_function(
    name='array_compact',
    clickhouse_name='arrayCompact',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayCompact'],
    doc='Remove consecutive duplicates. Maps to arrayCompact(arr).',
)
def _build_array_compact(expr, alias=None):
    from .functions import Function

    return Function('arrayCompact', expr, alias=alias)


@register_function(
    name='array_uniq',
    clickhouse_name='arrayUniq',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayUniq'],
    doc='Count unique elements. Maps to arrayUniq(arr).',
)
def _build_array_uniq(expr, alias=None):
    from .functions import Function

    return Function('arrayUniq', expr, alias=alias)


@register_function(
    name='array_enumerate',
    clickhouse_name='arrayEnumerate',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayEnumerate'],
    doc='Return array of indices. Maps to arrayEnumerate(arr).',
)
def _build_array_enumerate(expr, alias=None):
    from .functions import Function

    return Function('arrayEnumerate', expr, alias=alias)


@register_function(
    name='array_pop_back',
    clickhouse_name='arrayPopBack',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayPopBack'],
    doc='Remove last element. Maps to arrayPopBack(arr).',
)
def _build_array_pop_back(expr, alias=None):
    from .functions import Function

    return Function('arrayPopBack', expr, alias=alias)


@register_function(
    name='array_pop_front',
    clickhouse_name='arrayPopFront',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayPopFront'],
    doc='Remove first element. Maps to arrayPopFront(arr).',
)
def _build_array_pop_front(expr, alias=None):
    from .functions import Function

    return Function('arrayPopFront', expr, alias=alias)


@register_function(
    name='array_push_back',
    clickhouse_name='arrayPushBack',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayPushBack', 'append'],
    doc='Add element to end. Maps to arrayPushBack(arr, elem).',
)
def _build_array_push_back(expr, elem, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('arrayPushBack', expr, Expression.wrap(elem), alias=alias)


@register_function(
    name='array_push_front',
    clickhouse_name='arrayPushFront',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayPushFront', 'prepend'],
    doc='Add element to front. Maps to arrayPushFront(arr, elem).',
)
def _build_array_push_front(expr, elem, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('arrayPushFront', expr, Expression.wrap(elem), alias=alias)


@register_function(
    name='array_filter',
    clickhouse_name='arrayFilter',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayFilter'],
    doc='Filter array by lambda. Maps to arrayFilter(lambda, arr).',
)
def _build_array_filter(expr, lambda_expr, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('arrayFilter', Expression.wrap(lambda_expr), expr, alias=alias)


@register_function(
    name='array_map',
    clickhouse_name='arrayMap',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayMap'],
    doc='Apply lambda to each element. Maps to arrayMap(lambda, arr).',
)
def _build_array_map(expr, lambda_expr, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('arrayMap', Expression.wrap(lambda_expr), expr, alias=alias)


@register_function(
    name='array_reduce',
    clickhouse_name='arrayReduce',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayReduce'],
    doc='Reduce array with aggregate function. Maps to arrayReduce(agg_func, arr).',
)
def _build_array_reduce(expr, agg_func: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('arrayReduce', Literal(agg_func), expr, alias=alias)


@register_function(
    name='array_exists',
    clickhouse_name='arrayExists',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayExists', 'any_match'],
    doc='Check if any element matches. Maps to arrayExists(lambda, arr).',
)
def _build_array_exists(expr, lambda_expr=None, alias=None):
    from .functions import Function
    from .expressions import Expression

    if lambda_expr is not None:
        return Function('arrayExists', Expression.wrap(lambda_expr), expr, alias=alias)
    return Function('arrayExists', expr, alias=alias)


@register_function(
    name='array_all',
    clickhouse_name='arrayAll',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayAll', 'all_match'],
    doc='Check if all elements match. Maps to arrayAll(lambda, arr).',
)
def _build_array_all(expr, lambda_expr=None, alias=None):
    from .functions import Function
    from .expressions import Expression

    if lambda_expr is not None:
        return Function('arrayAll', Expression.wrap(lambda_expr), expr, alias=alias)
    return Function('arrayAll', expr, alias=alias)


@register_function(
    name='array_cum_sum',
    clickhouse_name='arrayCumSum',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayCumSum'],
    doc='Cumulative sum of array. Maps to arrayCumSum(arr).',
)
def _build_array_cum_sum(expr, alias=None):
    from .functions import Function

    return Function('arrayCumSum', expr, alias=alias)


@register_function(
    name='array_difference',
    clickhouse_name='arrayDifference',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayDifference'],
    doc='Difference between consecutive elements. Maps to arrayDifference(arr).',
)
def _build_array_difference(expr, alias=None):
    from .functions import Function

    return Function('arrayDifference', expr, alias=alias)


@register_function(
    name='array_product',
    clickhouse_name='arrayProduct',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayProduct'],
    doc='Product of array elements. Maps to arrayProduct(arr).',
)
def _build_array_product(expr, alias=None):
    from .functions import Function

    return Function('arrayProduct', expr, alias=alias)


@register_function(
    name='array_string_concat',
    clickhouse_name='arrayStringConcat',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayStringConcat', 'join'],
    doc='Join array elements to string. Maps to arrayStringConcat(arr, sep).',
)
def _build_array_string_concat(expr, sep: str = '', alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('arrayStringConcat', expr, Literal(sep), alias=alias)


@register_function(
    name='index_of',
    clickhouse_name='indexOf',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['indexOf', 'array_index'],
    doc='Find index of element. Maps to indexOf(arr, elem).',
)
def _build_index_of(expr, elem, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('indexOf', expr, Expression.wrap(elem), alias=alias)


@register_function(
    name='count_equal',
    clickhouse_name='countEqual',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['countEqual'],
    doc='Count occurrences of element. Maps to countEqual(arr, elem).',
)
def _build_count_equal(expr, elem, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('countEqual', expr, Expression.wrap(elem), alias=alias)


# =============================================================================
# MORE JSON FUNCTIONS
# =============================================================================


@register_function(
    name='json_extract_raw',
    clickhouse_name='JSONExtractRaw',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractRaw', 'get_raw'],
    doc='Extract raw JSON. Maps to JSONExtractRaw(json, path).',
)
def _build_json_extract_raw(json, path: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('JSONExtractRaw', json, Literal(path), alias=alias)


@register_function(
    name='json_extract_array_raw',
    clickhouse_name='JSONExtractArrayRaw',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractArrayRaw', 'get_array'],
    doc='Extract array as raw JSON strings. Maps to JSONExtractArrayRaw(json, path).',
)
def _build_json_extract_array_raw(json, path: str = '', alias=None):
    from .functions import Function
    from .expressions import Literal

    if path:
        return Function('JSONExtractArrayRaw', json, Literal(path), alias=alias)
    return Function('JSONExtractArrayRaw', json, alias=alias)


@register_function(
    name='json_extract_keys',
    clickhouse_name='JSONExtractKeys',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractKeys', 'keys'],
    doc='Extract keys from JSON object. Maps to JSONExtractKeys(json).',
)
def _build_json_extract_keys(json, alias=None):
    from .functions import Function

    return Function('JSONExtractKeys', json, alias=alias)


@register_function(
    name='json_extract_values',
    clickhouse_name='JSONExtractValues',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractValues', 'values'],
    doc='Extract values from JSON object. Maps to JSONExtractValues(json).',
)
def _build_json_extract_values(json, alias=None):
    from .functions import Function

    return Function('JSONExtractValues', json, alias=alias)


@register_function(
    name='json_type',
    clickhouse_name='JSONType',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONType', 'type'],
    doc='Get JSON value type. Maps to JSONType(json, path).',
)
def _build_json_type(json, path: str = '', alias=None):
    from .functions import Function
    from .expressions import Literal

    if path:
        return Function('JSONType', json, Literal(path), alias=alias)
    return Function('JSONType', json, alias=alias)


@register_function(
    name='json_length',
    clickhouse_name='JSONLength',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONLength'],
    doc='Get length of JSON array or object. Maps to JSONLength(json, path).',
)
def _build_json_length(json, path: str = '', alias=None):
    from .functions import Function
    from .expressions import Literal

    if path:
        return Function('JSONLength', json, Literal(path), alias=alias)
    return Function('JSONLength', json, alias=alias)


@register_function(
    name='json_has',
    clickhouse_name='JSONHas',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONHas', 'has_key'],
    doc='Check if JSON has key. Maps to JSONHas(json, key).',
)
def _build_json_has(json, key: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('JSONHas', json, Literal(key), alias=alias)


@register_function(
    name='is_valid_json',
    clickhouse_name='isValidJSON',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['isValidJSON', 'is_valid'],
    doc='Check if string is valid JSON. Maps to isValidJSON(str).',
)
def _build_is_valid_json(expr, alias=None):
    from .functions import Function

    return Function('isValidJSON', expr, alias=alias)


@register_function(
    name='to_json_string',
    clickhouse_name='toJSONString',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['toJSONString'],
    doc='Convert to JSON string. Maps to toJSONString(x).',
)
def _build_to_json_string(expr, alias=None):
    from .functions import Function

    return Function('toJSONString', expr, alias=alias)


@register_function(
    name='json_all_paths',
    clickhouse_name='JSONAllPaths',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONAllPaths', 'all_paths'],
    doc='Get all paths in JSON document. Maps to JSONAllPaths(json).',
)
def _build_json_all_paths(json, alias=None):
    from .functions import Function

    return Function('JSONAllPaths', json, alias=alias)


@register_function(
    name='json_all_paths_with_types',
    clickhouse_name='JSONAllPathsWithTypes',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONAllPathsWithTypes', 'all_paths_with_types'],
    doc='Get all paths with their types in JSON document. Maps to JSONAllPathsWithTypes(json).',
)
def _build_json_all_paths_with_types(json, alias=None):
    from .functions import Function

    return Function('JSONAllPathsWithTypes', json, alias=alias)


# ---------- More JSON Functions from ClickHouse docs ----------


@register_function(
    name='json_array_length',
    clickhouse_name='JSONArrayLength',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONArrayLength'],
    doc='Get length of JSON array. Maps to JSONArrayLength(json, path).',
)
def _build_json_array_length(json, path: str = None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if path:
        return Function('JSONArrayLength', json, Literal(path), alias=alias)
    return Function('JSONArrayLength', json, alias=alias)


@register_function(
    name='json_dynamic_paths',
    clickhouse_name='JSONDynamicPaths',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONDynamicPaths'],
    doc='Get dynamic paths from JSON column. Maps to JSONDynamicPaths(json).',
)
def _build_json_dynamic_paths(json, alias=None):
    from .functions import Function

    return Function('JSONDynamicPaths', json, alias=alias)


@register_function(
    name='json_dynamic_paths_with_types',
    clickhouse_name='JSONDynamicPathsWithTypes',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONDynamicPathsWithTypes'],
    doc='Get dynamic paths with types from JSON column. Maps to JSONDynamicPathsWithTypes(json).',
)
def _build_json_dynamic_paths_with_types(json, alias=None):
    from .functions import Function

    return Function('JSONDynamicPathsWithTypes', json, alias=alias)


@register_function(
    name='json_extract',
    clickhouse_name='JSONExtract',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtract'],
    doc='Extract value from JSON with type. Maps to JSONExtract(json, path, type).',
)
def _build_json_extract(json, path: str, type_name: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('JSONExtract', json, Literal(path), Literal(type_name), alias=alias)


@register_function(
    name='json_extract_uint',
    clickhouse_name='JSONExtractUInt',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractUInt'],
    doc='Extract unsigned integer from JSON. Maps to JSONExtractUInt(json, path).',
)
def _build_json_extract_uint(json, path: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('JSONExtractUInt', json, Literal(path), alias=alias)


@register_function(
    name='json_extract_keys_and_values',
    clickhouse_name='JSONExtractKeysAndValues',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractKeysAndValues'],
    doc='Extract keys and values from JSON. Maps to JSONExtractKeysAndValues(json, path, type).',
)
def _build_json_extract_keys_and_values(json, path: str, type_name: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('JSONExtractKeysAndValues', json, Literal(path), Literal(type_name), alias=alias)


@register_function(
    name='json_extract_keys_and_values_raw',
    clickhouse_name='JSONExtractKeysAndValuesRaw',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractKeysAndValuesRaw'],
    doc='Extract keys and values as raw strings from JSON. Maps to JSONExtractKeysAndValuesRaw(json, path).',
)
def _build_json_extract_keys_and_values_raw(json, path: str = None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if path:
        return Function('JSONExtractKeysAndValuesRaw', json, Literal(path), alias=alias)
    return Function('JSONExtractKeysAndValuesRaw', json, alias=alias)


@register_function(
    name='json_merge_patch',
    clickhouse_name='JSONMergePatch',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONMergePatch'],
    doc='Merge multiple JSON objects. Maps to JSONMergePatch(json1, json2, ...).',
)
def _build_json_merge_patch(*jsons, alias=None):
    from .functions import Function

    return Function('JSONMergePatch', *jsons, alias=alias)


@register_function(
    name='json_shared_data_paths',
    clickhouse_name='JSONSharedDataPaths',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONSharedDataPaths'],
    doc='Get shared data paths from JSON column. Maps to JSONSharedDataPaths(json).',
)
def _build_json_shared_data_paths(json, alias=None):
    from .functions import Function

    return Function('JSONSharedDataPaths', json, alias=alias)


@register_function(
    name='json_shared_data_paths_with_types',
    clickhouse_name='JSONSharedDataPathsWithTypes',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONSharedDataPathsWithTypes'],
    doc='Get shared data paths with types from JSON column. Maps to JSONSharedDataPathsWithTypes(json).',
)
def _build_json_shared_data_paths_with_types(json, alias=None):
    from .functions import Function

    return Function('JSONSharedDataPathsWithTypes', json, alias=alias)


@register_function(
    name='json_exists',
    clickhouse_name='JSON_EXISTS',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSON_EXISTS'],
    doc='Check if path exists in JSON (SQL/JSON standard). Maps to JSON_EXISTS(json, path).',
)
def _build_json_exists(json, path: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('JSON_EXISTS', json, Literal(path), alias=alias)


@register_function(
    name='json_query',
    clickhouse_name='JSON_QUERY',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSON_QUERY'],
    doc='Query JSON using JSONPath (SQL/JSON standard). Maps to JSON_QUERY(json, path).',
)
def _build_json_query(json, path: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('JSON_QUERY', json, Literal(path), alias=alias)


@register_function(
    name='json_value',
    clickhouse_name='JSON_VALUE',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSON_VALUE'],
    doc='Extract scalar value from JSON using JSONPath (SQL/JSON standard). Maps to JSON_VALUE(json, path).',
)
def _build_json_value(json, path: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('JSON_VALUE', json, Literal(path), alias=alias)


# ---------- simpleJSON functions (faster, limited JSON parsing) ----------


@register_function(
    name='simple_json_has',
    clickhouse_name='simpleJSONHas',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['simpleJSONHas', 'visitParamHas'],
    doc='Check if field exists in JSON (fast, limited). Maps to simpleJSONHas(json, field).',
)
def _build_simple_json_has(json, field: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('simpleJSONHas', json, Literal(field), alias=alias)


@register_function(
    name='simple_json_extract_string',
    clickhouse_name='simpleJSONExtractString',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['simpleJSONExtractString', 'visitParamExtractString'],
    doc='Extract string from JSON (fast, limited). Maps to simpleJSONExtractString(json, field).',
)
def _build_simple_json_extract_string(json, field: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('simpleJSONExtractString', json, Literal(field), alias=alias)


@register_function(
    name='simple_json_extract_int',
    clickhouse_name='simpleJSONExtractInt',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['simpleJSONExtractInt', 'visitParamExtractInt'],
    doc='Extract integer from JSON (fast, limited). Maps to simpleJSONExtractInt(json, field).',
)
def _build_simple_json_extract_int(json, field: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('simpleJSONExtractInt', json, Literal(field), alias=alias)


@register_function(
    name='simple_json_extract_uint',
    clickhouse_name='simpleJSONExtractUInt',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['simpleJSONExtractUInt', 'visitParamExtractUInt'],
    doc='Extract unsigned integer from JSON (fast, limited). Maps to simpleJSONExtractUInt(json, field).',
)
def _build_simple_json_extract_uint(json, field: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('simpleJSONExtractUInt', json, Literal(field), alias=alias)


@register_function(
    name='simple_json_extract_float',
    clickhouse_name='simpleJSONExtractFloat',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['simpleJSONExtractFloat', 'visitParamExtractFloat'],
    doc='Extract float from JSON (fast, limited). Maps to simpleJSONExtractFloat(json, field).',
)
def _build_simple_json_extract_float(json, field: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('simpleJSONExtractFloat', json, Literal(field), alias=alias)


@register_function(
    name='simple_json_extract_bool',
    clickhouse_name='simpleJSONExtractBool',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['simpleJSONExtractBool', 'visitParamExtractBool'],
    doc='Extract boolean from JSON (fast, limited). Maps to simpleJSONExtractBool(json, field).',
)
def _build_simple_json_extract_bool(json, field: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('simpleJSONExtractBool', json, Literal(field), alias=alias)


@register_function(
    name='simple_json_extract_raw',
    clickhouse_name='simpleJSONExtractRaw',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['simpleJSONExtractRaw', 'visitParamExtractRaw'],
    doc='Extract raw value from JSON (fast, limited). Maps to simpleJSONExtractRaw(json, field).',
)
def _build_simple_json_extract_raw(json, field: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('simpleJSONExtractRaw', json, Literal(field), alias=alias)


# ---------- Case-insensitive JSON functions ----------


@register_function(
    name='json_extract_string_ci',
    clickhouse_name='JSONExtractStringCaseInsensitive',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractStringCaseInsensitive'],
    doc='Extract string from JSON (case-insensitive). Maps to JSONExtractStringCaseInsensitive(json, path).',
)
def _build_json_extract_string_ci(json, path: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('JSONExtractStringCaseInsensitive', json, Literal(path), alias=alias)


@register_function(
    name='json_extract_int_ci',
    clickhouse_name='JSONExtractIntCaseInsensitive',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractIntCaseInsensitive'],
    doc='Extract integer from JSON (case-insensitive). Maps to JSONExtractIntCaseInsensitive(json, path).',
)
def _build_json_extract_int_ci(json, path: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('JSONExtractIntCaseInsensitive', json, Literal(path), alias=alias)


@register_function(
    name='json_extract_float_ci',
    clickhouse_name='JSONExtractFloatCaseInsensitive',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractFloatCaseInsensitive'],
    doc='Extract float from JSON (case-insensitive). Maps to JSONExtractFloatCaseInsensitive(json, path).',
)
def _build_json_extract_float_ci(json, path: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('JSONExtractFloatCaseInsensitive', json, Literal(path), alias=alias)


@register_function(
    name='json_extract_bool_ci',
    clickhouse_name='JSONExtractBoolCaseInsensitive',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractBoolCaseInsensitive'],
    doc='Extract boolean from JSON (case-insensitive). Maps to JSONExtractBoolCaseInsensitive(json, path).',
)
def _build_json_extract_bool_ci(json, path: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('JSONExtractBoolCaseInsensitive', json, Literal(path), alias=alias)


@register_function(
    name='json_extract_raw_ci',
    clickhouse_name='JSONExtractRawCaseInsensitive',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractRawCaseInsensitive'],
    doc='Extract raw value from JSON (case-insensitive). Maps to JSONExtractRawCaseInsensitive(json, path).',
)
def _build_json_extract_raw_ci(json, path: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('JSONExtractRawCaseInsensitive', json, Literal(path), alias=alias)


@register_function(
    name='json_extract_array_raw_ci',
    clickhouse_name='JSONExtractArrayRawCaseInsensitive',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractArrayRawCaseInsensitive'],
    doc='Extract array as raw strings (case-insensitive). Maps to JSONExtractArrayRawCaseInsensitive(json, path).',
)
def _build_json_extract_array_raw_ci(json, path: str = None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if path:
        return Function('JSONExtractArrayRawCaseInsensitive', json, Literal(path), alias=alias)
    return Function('JSONExtractArrayRawCaseInsensitive', json, alias=alias)


@register_function(
    name='json_extract_keys_ci',
    clickhouse_name='JSONExtractKeysCaseInsensitive',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractKeysCaseInsensitive'],
    doc='Extract keys from JSON (case-insensitive). Maps to JSONExtractKeysCaseInsensitive(json, path).',
)
def _build_json_extract_keys_ci(json, path: str = None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if path:
        return Function('JSONExtractKeysCaseInsensitive', json, Literal(path), alias=alias)
    return Function('JSONExtractKeysCaseInsensitive', json, alias=alias)


# =============================================================================
# URL FUNCTIONS
# =============================================================================


@register_function(
    name='domain',
    clickhouse_name='domain',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.URL,
    doc='Extract domain from URL. Maps to domain(url).',
)
def _build_domain(expr, alias=None):
    from .functions import Function

    return Function('domain', expr, alias=alias)


@register_function(
    name='domain_without_www',
    clickhouse_name='domainWithoutWWW',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.URL,
    aliases=['domainWithoutWWW'],
    doc='Extract domain without www. Maps to domainWithoutWWW(url).',
)
def _build_domain_without_www(expr, alias=None):
    from .functions import Function

    return Function('domainWithoutWWW', expr, alias=alias)


@register_function(
    name='top_level_domain',
    clickhouse_name='topLevelDomain',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.URL,
    aliases=['topLevelDomain', 'tld'],
    doc='Extract top-level domain. Maps to topLevelDomain(url).',
)
def _build_top_level_domain(expr, alias=None):
    from .functions import Function

    return Function('topLevelDomain', expr, alias=alias)


@register_function(
    name='protocol',
    clickhouse_name='protocol',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.URL,
    aliases=['scheme'],
    doc='Extract protocol/scheme. Maps to protocol(url).',
)
def _build_protocol(expr, alias=None):
    from .functions import Function

    return Function('protocol', expr, alias=alias)


@register_function(
    name='url_path',
    clickhouse_name='path',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.URL,
    aliases=['path'],
    doc='Extract path from URL. Maps to path(url).',
)
def _build_url_path(expr, alias=None):
    from .functions import Function

    return Function('path', expr, alias=alias)


@register_function(
    name='path_full',
    clickhouse_name='pathFull',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.URL,
    aliases=['pathFull'],
    doc='Extract full path including query. Maps to pathFull(url).',
)
def _build_path_full(expr, alias=None):
    from .functions import Function

    return Function('pathFull', expr, alias=alias)


@register_function(
    name='query_string',
    clickhouse_name='queryString',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.URL,
    aliases=['queryString'],
    doc='Extract query string. Maps to queryString(url).',
)
def _build_query_string(expr, alias=None):
    from .functions import Function

    return Function('queryString', expr, alias=alias)


@register_function(
    name='fragment',
    clickhouse_name='fragment',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.URL,
    doc='Extract fragment/anchor. Maps to fragment(url).',
)
def _build_fragment(expr, alias=None):
    from .functions import Function

    return Function('fragment', expr, alias=alias)


@register_function(
    name='url_port',
    clickhouse_name='port',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.URL,
    aliases=['port'],
    doc='Extract port from URL. Maps to port(url).',
)
def _build_url_port(expr, alias=None):
    from .functions import Function

    return Function('port', expr, alias=alias)


@register_function(
    name='extract_url_parameter',
    clickhouse_name='extractURLParameter',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.URL,
    aliases=['extractURLParameter', 'get_param'],
    doc='Extract URL parameter value. Maps to extractURLParameter(url, name).',
)
def _build_extract_url_parameter(expr, name: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('extractURLParameter', expr, Literal(name), alias=alias)


@register_function(
    name='extract_url_parameters',
    clickhouse_name='extractURLParameters',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.URL,
    aliases=['extractURLParameters', 'get_params'],
    doc='Extract all URL parameters as array. Maps to extractURLParameters(url).',
)
def _build_extract_url_parameters(expr, alias=None):
    from .functions import Function

    return Function('extractURLParameters', expr, alias=alias)


@register_function(
    name='extract_url_parameter_names',
    clickhouse_name='extractURLParameterNames',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.URL,
    aliases=['extractURLParameterNames', 'get_param_names'],
    doc='Extract URL parameter names. Maps to extractURLParameterNames(url).',
)
def _build_extract_url_parameter_names(expr, alias=None):
    from .functions import Function

    return Function('extractURLParameterNames', expr, alias=alias)


@register_function(
    name='cut_url_parameter',
    clickhouse_name='cutURLParameter',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.URL,
    aliases=['cutURLParameter'],
    doc='Remove URL parameter. Maps to cutURLParameter(url, name).',
)
def _build_cut_url_parameter(expr, name: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('cutURLParameter', expr, Literal(name), alias=alias)


@register_function(
    name='decode_url_component',
    clickhouse_name='decodeURLComponent',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.URL,
    aliases=['decodeURLComponent', 'url_decode'],
    doc='Decode URL component. Maps to decodeURLComponent(str).',
)
def _build_decode_url_component(expr, alias=None):
    from .functions import Function

    return Function('decodeURLComponent', expr, alias=alias)


@register_function(
    name='encode_url_component',
    clickhouse_name='encodeURLComponent',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.URL,
    aliases=['encodeURLComponent', 'url_encode'],
    doc='Encode URL component. Maps to encodeURLComponent(str).',
)
def _build_encode_url_component(expr, alias=None):
    from .functions import Function

    return Function('encodeURLComponent', expr, alias=alias)


# =============================================================================
# IP ADDRESS FUNCTIONS
# =============================================================================


@register_function(
    name='to_ipv4',
    clickhouse_name='toIPv4',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.IP,
    aliases=['toIPv4'],
    doc='Convert string to IPv4. Maps to toIPv4(str).',
)
def _build_to_ipv4(expr, alias=None):
    from .functions import Function

    return Function('toIPv4', expr, alias=alias)


@register_function(
    name='to_ipv6',
    clickhouse_name='toIPv6',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.IP,
    aliases=['toIPv6'],
    doc='Convert string to IPv6. Maps to toIPv6(str).',
)
def _build_to_ipv6(expr, alias=None):
    from .functions import Function

    return Function('toIPv6', expr, alias=alias)


@register_function(
    name='ipv4_num_to_string',
    clickhouse_name='IPv4NumToString',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.IP,
    aliases=['IPv4NumToString'],
    doc='Convert IPv4 number to string. Maps to IPv4NumToString(num).',
)
def _build_ipv4_num_to_string(expr, alias=None):
    from .functions import Function

    return Function('IPv4NumToString', expr, alias=alias)


@register_function(
    name='ipv4_string_to_num',
    clickhouse_name='IPv4StringToNum',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.IP,
    aliases=['IPv4StringToNum'],
    doc='Convert IPv4 string to number. Maps to IPv4StringToNum(str).',
)
def _build_ipv4_string_to_num(expr, alias=None):
    from .functions import Function

    return Function('IPv4StringToNum', expr, alias=alias)


@register_function(
    name='ipv6_num_to_string',
    clickhouse_name='IPv6NumToString',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.IP,
    aliases=['IPv6NumToString'],
    doc='Convert IPv6 number to string. Maps to IPv6NumToString(num).',
)
def _build_ipv6_num_to_string(expr, alias=None):
    from .functions import Function

    return Function('IPv6NumToString', expr, alias=alias)


@register_function(
    name='ipv4_to_ipv6',
    clickhouse_name='IPv4ToIPv6',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.IP,
    aliases=['IPv4ToIPv6'],
    doc='Convert IPv4 to IPv6. Maps to IPv4ToIPv6(ip).',
)
def _build_ipv4_to_ipv6(expr, alias=None):
    from .functions import Function

    return Function('IPv4ToIPv6', expr, alias=alias)


@register_function(
    name='is_ipv4_string',
    clickhouse_name='isIPv4String',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.IP,
    aliases=['isIPv4String'],
    doc='Check if string is valid IPv4. Maps to isIPv4String(str).',
)
def _build_is_ipv4_string(expr, alias=None):
    from .functions import Function

    return Function('isIPv4String', expr, alias=alias)


@register_function(
    name='is_ipv6_string',
    clickhouse_name='isIPv6String',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.IP,
    aliases=['isIPv6String'],
    doc='Check if string is valid IPv6. Maps to isIPv6String(str).',
)
def _build_is_ipv6_string(expr, alias=None):
    from .functions import Function

    return Function('isIPv6String', expr, alias=alias)


@register_function(
    name='ipv4_cidr_to_range',
    clickhouse_name='IPv4CIDRToRange',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.IP,
    aliases=['IPv4CIDRToRange'],
    doc='Convert CIDR to IP range. Maps to IPv4CIDRToRange(ip, cidr).',
)
def _build_ipv4_cidr_to_range(expr, cidr: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('IPv4CIDRToRange', expr, Literal(cidr), alias=alias)


# =============================================================================
# GEO/DISTANCE FUNCTIONS
# =============================================================================


@register_function(
    name='great_circle_distance',
    clickhouse_name='greatCircleDistance',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['greatCircleDistance'],
    doc='Calculate great circle distance. Maps to greatCircleDistance(lon1, lat1, lon2, lat2).',
)
def _build_great_circle_distance(lon1, lat1, lon2, lat2, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function(
        'greatCircleDistance',
        Expression.wrap(lon1),
        Expression.wrap(lat1),
        Expression.wrap(lon2),
        Expression.wrap(lat2),
        alias=alias,
    )


@register_function(
    name='geo_distance',
    clickhouse_name='geoDistance',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['geoDistance'],
    doc='Calculate geographic distance (WGS-84). Maps to geoDistance(lon1, lat1, lon2, lat2).',
)
def _build_geo_distance(lon1, lat1, lon2, lat2, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function(
        'geoDistance',
        Expression.wrap(lon1),
        Expression.wrap(lat1),
        Expression.wrap(lon2),
        Expression.wrap(lat2),
        alias=alias,
    )


@register_function(
    name='point_in_ellipses',
    clickhouse_name='pointInEllipses',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['pointInEllipses'],
    doc='Check if point is inside ellipses. Maps to pointInEllipses(x, y, ...).',
)
def _build_point_in_ellipses(*args, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('pointInEllipses', *[Expression.wrap(a) for a in args], alias=alias)


@register_function(
    name='point_in_polygon',
    clickhouse_name='pointInPolygon',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['pointInPolygon'],
    doc='Check if point is inside polygon. Maps to pointInPolygon((x, y), polygon).',
)
def _build_point_in_polygon(point, polygon, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('pointInPolygon', Expression.wrap(point), Expression.wrap(polygon), alias=alias)


@register_function(
    name='geo_to_h3',
    clickhouse_name='geoToH3',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['geoToH3'],
    doc='Convert geo coordinates to H3 index. Maps to geoToH3(lon, lat, resolution).',
)
def _build_geo_to_h3(lon, lat, resolution: int, alias=None):
    from .functions import Function
    from .expressions import Expression, Literal

    return Function('geoToH3', Expression.wrap(lon), Expression.wrap(lat), Literal(resolution), alias=alias)


@register_function(
    name='h3_to_geo',
    clickhouse_name='h3ToGeo',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['h3ToGeo'],
    doc='Convert H3 index to geo coordinates. Maps to h3ToGeo(h3index).',
)
def _build_h3_to_geo(expr, alias=None):
    from .functions import Function

    return Function('h3ToGeo', expr, alias=alias)


@register_function(
    name='l1_distance',
    clickhouse_name='L1Distance',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['L1Distance', 'manhattan_distance'],
    doc='Calculate L1 (Manhattan) distance. Maps to L1Distance(vec1, vec2).',
)
def _build_l1_distance(vec1, vec2, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('L1Distance', Expression.wrap(vec1), Expression.wrap(vec2), alias=alias)


@register_function(
    name='l2_distance',
    clickhouse_name='L2Distance',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['L2Distance', 'euclidean_distance'],
    doc='Calculate L2 (Euclidean) distance. Maps to L2Distance(vec1, vec2).',
)
def _build_l2_distance(vec1, vec2, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('L2Distance', Expression.wrap(vec1), Expression.wrap(vec2), alias=alias)


@register_function(
    name='l2_squared_distance',
    clickhouse_name='L2SquaredDistance',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['L2SquaredDistance'],
    doc='Calculate squared L2 distance. Maps to L2SquaredDistance(vec1, vec2).',
)
def _build_l2_squared_distance(vec1, vec2, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('L2SquaredDistance', Expression.wrap(vec1), Expression.wrap(vec2), alias=alias)


@register_function(
    name='linf_distance',
    clickhouse_name='LinfDistance',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['LinfDistance', 'chebyshev_distance'],
    doc='Calculate Linf (Chebyshev) distance. Maps to LinfDistance(vec1, vec2).',
)
def _build_linf_distance(vec1, vec2, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('LinfDistance', Expression.wrap(vec1), Expression.wrap(vec2), alias=alias)


@register_function(
    name='cosine_distance',
    clickhouse_name='cosineDistance',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['cosineDistance'],
    doc='Calculate cosine distance. Maps to cosineDistance(vec1, vec2).',
)
def _build_cosine_distance(vec1, vec2, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('cosineDistance', Expression.wrap(vec1), Expression.wrap(vec2), alias=alias)


@register_function(
    name='dot_product',
    clickhouse_name='dotProduct',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['dotProduct'],
    doc='Calculate dot product. Maps to dotProduct(vec1, vec2).',
)
def _build_dot_product(vec1, vec2, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('dotProduct', Expression.wrap(vec1), Expression.wrap(vec2), alias=alias)


@register_function(
    name='l2_norm',
    clickhouse_name='L2Norm',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['L2Norm', 'norm'],
    doc='Calculate L2 norm. Maps to L2Norm(vec).',
)
def _build_l2_norm(expr, alias=None):
    from .functions import Function

    return Function('L2Norm', expr, alias=alias)


@register_function(
    name='l2_normalize',
    clickhouse_name='L2Normalize',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['L2Normalize', 'normalize'],
    doc='Normalize vector to unit length. Maps to L2Normalize(vec).',
)
def _build_l2_normalize(expr, alias=None):
    from .functions import Function

    return Function('L2Normalize', expr, alias=alias)


# =============================================================================
# MORE AGGREGATE FUNCTIONS
# =============================================================================


@register_function(
    name='count_if',
    clickhouse_name='countIf',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['countIf'],
    doc='Count rows matching condition. Maps to countIf(condition).',
)
def _build_count_if(condition, alias=None):
    from .functions import AggregateFunction
    from .expressions import Expression

    return AggregateFunction('countIf', Expression.wrap(condition), alias=alias)


@register_function(
    name='sum_if',
    clickhouse_name='sumIf',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['sumIf'],
    doc='Sum values matching condition. Maps to sumIf(x, condition).',
)
def _build_sum_if(expr, condition, alias=None):
    from .functions import AggregateFunction
    from .expressions import Expression

    return AggregateFunction('sumIf', Expression.wrap(expr), Expression.wrap(condition), alias=alias)


@register_function(
    name='avg_if',
    clickhouse_name='avgIf',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['avgIf'],
    doc='Average values matching condition. Maps to avgIf(x, condition).',
)
def _build_avg_if(expr, condition, alias=None):
    from .functions import AggregateFunction
    from .expressions import Expression

    return AggregateFunction('avgIf', Expression.wrap(expr), Expression.wrap(condition), alias=alias)


@register_function(
    name='min_if',
    clickhouse_name='minIf',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['minIf'],
    doc='Minimum matching condition. Maps to minIf(x, condition).',
)
def _build_min_if(expr, condition, alias=None):
    from .functions import AggregateFunction
    from .expressions import Expression

    return AggregateFunction('minIf', Expression.wrap(expr), Expression.wrap(condition), alias=alias)


@register_function(
    name='max_if',
    clickhouse_name='maxIf',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['maxIf'],
    doc='Maximum matching condition. Maps to maxIf(x, condition).',
)
def _build_max_if(expr, condition, alias=None):
    from .functions import AggregateFunction
    from .expressions import Expression

    return AggregateFunction('maxIf', Expression.wrap(expr), Expression.wrap(condition), alias=alias)


@register_function(
    name='quantile',
    clickhouse_name='quantile',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['percentile'],
    doc='Calculate quantile. Maps to quantile(level)(x).',
)
def _build_quantile(expr, level: float = 0.5, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction(f'quantile({level})', expr, alias=alias)


@register_function(
    name='quantiles',
    clickhouse_name='quantiles',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['percentiles'],
    doc='Calculate multiple quantiles. Maps to quantiles(levels...)(x).',
)
def _build_quantiles(expr, *levels, alias=None):
    from .functions import AggregateFunction

    levels_str = ', '.join(str(l) for l in levels)
    return AggregateFunction(f'quantiles({levels_str})', expr, alias=alias)


@register_function(
    name='top_k',
    clickhouse_name='topK',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['topK'],
    doc='Get top K frequent values. Maps to topK(k)(x).',
)
def _build_top_k(expr, k: int = 10, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction(f'topK({k})', expr, alias=alias)


@register_function(
    name='top_k_weighted',
    clickhouse_name='topKWeighted',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['topKWeighted'],
    doc='Get top K weighted values. Maps to topKWeighted(k)(x, weight).',
)
def _build_top_k_weighted(expr, weight, k: int = 10, alias=None):
    from .functions import AggregateFunction
    from .expressions import Expression

    return AggregateFunction(f'topKWeighted({k})', Expression.wrap(expr), Expression.wrap(weight), alias=alias)


@register_function(
    name='histogram',
    clickhouse_name='histogram',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    doc='Build histogram. Maps to histogram(num_bins)(x).',
)
def _build_histogram(expr, num_bins: int = 10, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction(f'histogram({num_bins})', expr, alias=alias)


@register_function(
    name='uniq_exact',
    clickhouse_name='uniqExact',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['uniqExact', 'count_distinct_exact'],
    doc='Exact count distinct. Maps to uniqExact(x).',
)
def _build_uniq_exact(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('uniqExact', expr, alias=alias)


@register_function(
    name='uniq_combined',
    clickhouse_name='uniqCombined',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['uniqCombined'],
    doc='Approximate count distinct (HyperLogLog++). Maps to uniqCombined(x).',
)
def _build_uniq_combined(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('uniqCombined', expr, alias=alias)


@register_function(
    name='avg_weighted',
    clickhouse_name='avgWeighted',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['avgWeighted', 'weighted_avg'],
    doc='Weighted average. Maps to avgWeighted(x, weight).',
)
def _build_avg_weighted(expr, weight, alias=None):
    from .functions import AggregateFunction
    from .expressions import Expression

    return AggregateFunction('avgWeighted', Expression.wrap(expr), Expression.wrap(weight), alias=alias)


@register_function(
    name='group_concat',
    clickhouse_name='groupConcat',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['groupConcat', 'string_agg'],
    doc='Concatenate values with separator. Maps to groupConcat(sep)(x).',
)
def _build_group_concat(expr, sep: str = ',', alias=None):
    from .functions import AggregateFunction
    from .expressions import Literal

    return AggregateFunction('groupConcat', expr, Literal(sep), alias=alias)


@register_function(
    name='group_bit_and',
    clickhouse_name='groupBitAnd',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['groupBitAnd'],
    doc='Bitwise AND of all values. Maps to groupBitAnd(x).',
)
def _build_group_bit_and(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('groupBitAnd', expr, alias=alias)


@register_function(
    name='group_bit_or',
    clickhouse_name='groupBitOr',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['groupBitOr'],
    doc='Bitwise OR of all values. Maps to groupBitOr(x).',
)
def _build_group_bit_or(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('groupBitOr', expr, alias=alias)


@register_function(
    name='group_bit_xor',
    clickhouse_name='groupBitXor',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['groupBitXor'],
    doc='Bitwise XOR of all values. Maps to groupBitXor(x).',
)
def _build_group_bit_xor(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('groupBitXor', expr, alias=alias)


@register_function(
    name='entropy',
    clickhouse_name='entropy',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    doc='Calculate entropy. Maps to entropy(x).',
)
def _build_entropy(expr, alias=None):
    from .functions import AggregateFunction

    return AggregateFunction('entropy', expr, alias=alias)


@register_function(
    name='simple_linear_regression',
    clickhouse_name='simpleLinearRegression',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['simpleLinearRegression', 'linear_regression'],
    doc='Simple linear regression. Maps to simpleLinearRegression(x, y).',
)
def _build_simple_linear_regression(x, y, alias=None):
    from .functions import AggregateFunction
    from .expressions import Expression

    return AggregateFunction('simpleLinearRegression', Expression.wrap(x), Expression.wrap(y), alias=alias)


@register_function(
    name='stochastic_linear_regression',
    clickhouse_name='stochasticLinearRegression',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['stochasticLinearRegression'],
    doc='Stochastic linear regression. Maps to stochasticLinearRegression(...).',
)
def _build_stochastic_linear_regression(*args, alias=None):
    from .functions import AggregateFunction
    from .expressions import Expression

    return AggregateFunction('stochasticLinearRegression', *[Expression.wrap(a) for a in args], alias=alias)


# =============================================================================
# MORE WINDOW FUNCTIONS
# =============================================================================


@register_function(
    name='percent_rank',
    clickhouse_name='percent_rank',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    aliases=['percentRank'],
    doc='Relative rank (0-1). Requires OVER clause.',
    supports_over=True,
)
def _build_percent_rank(alias=None):
    from .functions import WindowFunction

    return WindowFunction('percent_rank', alias=alias)


@register_function(
    name='cume_dist',
    clickhouse_name='cume_dist',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    aliases=['cumeDist'],
    doc='Cumulative distribution. Requires OVER clause.',
    supports_over=True,
)
def _build_cume_dist(alias=None):
    from .functions import WindowFunction

    return WindowFunction('cume_dist', alias=alias)


@register_function(
    name='nth_value',
    clickhouse_name='nth_value',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    aliases=['nthValue'],
    doc='Nth value in window frame. Requires OVER clause.',
    supports_over=True,
)
def _build_nth_value(expr, n: int, alias=None):
    from .functions import WindowFunction
    from .expressions import Expression, Literal

    return WindowFunction('nth_value', Expression.wrap(expr), Literal(n), alias=alias)


# =============================================================================
# ENCODING FUNCTIONS
# =============================================================================


@register_function(
    name='hex',
    clickhouse_name='hex',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ENCODING,
    doc='Convert to hexadecimal. Maps to hex(x).',
)
def _build_hex(expr, alias=None):
    from .functions import Function

    return Function('hex', expr, alias=alias)


@register_function(
    name='unhex',
    clickhouse_name='unhex',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ENCODING,
    doc='Convert from hexadecimal. Maps to unhex(str).',
)
def _build_unhex(expr, alias=None):
    from .functions import Function

    return Function('unhex', expr, alias=alias)


@register_function(
    name='bin',
    clickhouse_name='bin',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ENCODING,
    aliases=['binary'],
    doc='Convert to binary string. Maps to bin(x).',
)
def _build_bin(expr, alias=None):
    from .functions import Function

    return Function('bin', expr, alias=alias)


@register_function(
    name='unbin',
    clickhouse_name='unbin',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ENCODING,
    doc='Convert from binary string. Maps to unbin(str).',
)
def _build_unbin(expr, alias=None):
    from .functions import Function

    return Function('unbin', expr, alias=alias)


@register_function(
    name='base64_encode',
    clickhouse_name='base64Encode',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ENCODING,
    aliases=['base64Encode', 'to_base64'],
    doc='Encode to Base64. Maps to base64Encode(str).',
)
def _build_base64_encode(expr, alias=None):
    from .functions import Function

    return Function('base64Encode', expr, alias=alias)


@register_function(
    name='base64_decode',
    clickhouse_name='base64Decode',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ENCODING,
    aliases=['base64Decode', 'from_base64'],
    doc='Decode from Base64. Maps to base64Decode(str).',
)
def _build_base64_decode(expr, alias=None):
    from .functions import Function

    return Function('base64Decode', expr, alias=alias)


@register_function(
    name='bit_count',
    clickhouse_name='bitCount',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ENCODING,
    aliases=['bitCount', 'popcount'],
    doc='Count set bits. Maps to bitCount(x).',
)
def _build_bit_count(expr, alias=None):
    from .functions import Function

    return Function('bitCount', expr, alias=alias)


# =============================================================================
# UUID FUNCTIONS
# =============================================================================


@register_function(
    name='generate_uuid_v4',
    clickhouse_name='generateUUIDv4',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.UUID,
    aliases=['generateUUIDv4', 'uuid4', 'new_uuid'],
    doc='Generate random UUID v4. Maps to generateUUIDv4().',
)
def _build_generate_uuid_v4(alias=None):
    from .functions import Function

    return Function('generateUUIDv4', alias=alias)


@register_function(
    name='generate_uuid_v7',
    clickhouse_name='generateUUIDv7',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.UUID,
    aliases=['generateUUIDv7', 'uuid7'],
    doc='Generate time-ordered UUID v7. Maps to generateUUIDv7().',
)
def _build_generate_uuid_v7(alias=None):
    from .functions import Function

    return Function('generateUUIDv7', alias=alias)


@register_function(
    name='to_uuid',
    clickhouse_name='toUUID',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.UUID,
    aliases=['toUUID'],
    doc='Convert string to UUID. Maps to toUUID(str).',
)
def _build_to_uuid(expr, alias=None):
    from .functions import Function

    return Function('toUUID', expr, alias=alias)


@register_function(
    name='uuid_to_num',
    clickhouse_name='UUIDToNum',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.UUID,
    aliases=['UUIDToNum'],
    doc='Convert UUID to FixedString(16). Maps to UUIDToNum(uuid).',
)
def _build_uuid_to_num(expr, alias=None):
    from .functions import Function

    return Function('UUIDToNum', expr, alias=alias)


# =============================================================================
# MORE MATH FUNCTIONS
# =============================================================================


@register_function(
    name='asin',
    clickhouse_name='asin',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['arcsin'],
    doc='Arc sine. Maps to asin(x).',
)
def _build_asin(expr, alias=None):
    from .functions import Function

    return Function('asin', expr, alias=alias)


@register_function(
    name='acos',
    clickhouse_name='acos',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['arccos'],
    doc='Arc cosine. Maps to acos(x).',
)
def _build_acos(expr, alias=None):
    from .functions import Function

    return Function('acos', expr, alias=alias)


@register_function(
    name='atan',
    clickhouse_name='atan',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['arctan'],
    doc='Arc tangent. Maps to atan(x).',
)
def _build_atan(expr, alias=None):
    from .functions import Function

    return Function('atan', expr, alias=alias)


@register_function(
    name='atan2',
    clickhouse_name='atan2',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Arc tangent of y/x. Maps to atan2(y, x).',
)
def _build_atan2(y, x, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('atan2', Expression.wrap(y), Expression.wrap(x), alias=alias)


@register_function(
    name='sinh',
    clickhouse_name='sinh',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Hyperbolic sine. Maps to sinh(x).',
)
def _build_sinh(expr, alias=None):
    from .functions import Function

    return Function('sinh', expr, alias=alias)


@register_function(
    name='cosh',
    clickhouse_name='cosh',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Hyperbolic cosine. Maps to cosh(x).',
)
def _build_cosh(expr, alias=None):
    from .functions import Function

    return Function('cosh', expr, alias=alias)


@register_function(
    name='tanh',
    clickhouse_name='tanh',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Hyperbolic tangent. Maps to tanh(x).',
)
def _build_tanh(expr, alias=None):
    from .functions import Function

    return Function('tanh', expr, alias=alias)


@register_function(
    name='degrees',
    clickhouse_name='degrees',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['toDegrees'],
    doc='Convert radians to degrees. Maps to degrees(x).',
)
def _build_degrees(expr, alias=None):
    from .functions import Function

    return Function('degrees', expr, alias=alias)


@register_function(
    name='radians',
    clickhouse_name='radians',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['toRadians'],
    doc='Convert degrees to radians. Maps to radians(x).',
)
def _build_radians(expr, alias=None):
    from .functions import Function

    return Function('radians', expr, alias=alias)


@register_function(
    name='cbrt',
    clickhouse_name='cbrt',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Cube root. Maps to cbrt(x).',
)
def _build_cbrt(expr, alias=None):
    from .functions import Function

    return Function('cbrt', expr, alias=alias)


@register_function(
    name='erf',
    clickhouse_name='erf',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Error function. Maps to erf(x).',
)
def _build_erf(expr, alias=None):
    from .functions import Function

    return Function('erf', expr, alias=alias)


@register_function(
    name='erfc',
    clickhouse_name='erfc',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Complementary error function. Maps to erfc(x).',
)
def _build_erfc(expr, alias=None):
    from .functions import Function

    return Function('erfc', expr, alias=alias)


@register_function(
    name='lgamma',
    clickhouse_name='lgamma',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Log-gamma function. Maps to lgamma(x).',
)
def _build_lgamma(expr, alias=None):
    from .functions import Function

    return Function('lgamma', expr, alias=alias)


@register_function(
    name='tgamma',
    clickhouse_name='tgamma',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['gamma'],
    doc='Gamma function. Maps to tgamma(x).',
)
def _build_tgamma(expr, alias=None):
    from .functions import Function

    return Function('tgamma', expr, alias=alias)


@register_function(
    name='greatest',
    clickhouse_name='greatest',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Return greatest value. Maps to greatest(a, b, ...).',
    min_args=2,
    max_args=-1,
)
def _build_greatest(*args, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('greatest', *[Expression.wrap(a) for a in args], alias=alias)


@register_function(
    name='least',
    clickhouse_name='least',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    doc='Return least value. Maps to least(a, b, ...).',
    min_args=2,
    max_args=-1,
)
def _build_least(*args, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('least', *[Expression.wrap(a) for a in args], alias=alias)


@register_function(
    name='rand',
    clickhouse_name='rand',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['random'],
    doc='Generate random UInt32. Maps to rand().',
)
def _build_rand(alias=None):
    from .functions import Function

    return Function('rand', alias=alias)


@register_function(
    name='rand64',
    clickhouse_name='rand64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['random64'],
    doc='Generate random UInt64. Maps to rand64().',
)
def _build_rand64(alias=None):
    from .functions import Function

    return Function('rand64', alias=alias)


@register_function(
    name='rand_uniform',
    clickhouse_name='randUniform',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['randUniform'],
    doc='Generate uniform random in [min, max]. Maps to randUniform(min, max).',
)
def _build_rand_uniform(min_val, max_val, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('randUniform', Expression.wrap(min_val), Expression.wrap(max_val), alias=alias)


@register_function(
    name='rand_normal',
    clickhouse_name='randNormal',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['randNormal', 'rand_gaussian'],
    doc='Generate normal random. Maps to randNormal(mean, stddev).',
)
def _build_rand_normal(mean, stddev, alias=None):
    from .functions import Function
    from .expressions import Expression

    return Function('randNormal', Expression.wrap(mean), Expression.wrap(stddev), alias=alias)


# =============================================================================
# MORE STRING FUNCTIONS (Pandas .str compatibility)
# =============================================================================


@register_function(
    name='fullmatch',
    clickhouse_name='match',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Full regex match (anchored). Maps to match(s, "^pattern$").',
)
def _build_fullmatch(expr, pattern: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    # Add anchors for full match
    full_pattern = f'^{pattern}$'
    return Function('match', expr, Literal(full_pattern), alias=alias)


@register_function(
    name='findall',
    clickhouse_name='extractAll',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['extractAll'],
    doc='Find all regex matches. Maps to extractAll(s, pattern).',
)
def _build_findall(expr, pattern: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('extractAll', expr, Literal(pattern), alias=alias)


@register_function(
    name='get_char',
    clickhouse_name='substring',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Get character at index. Maps to substring(s, i, 1).',
)
def _build_get_char(expr, i: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    # Pandas uses 0-based, ClickHouse uses 1-based
    return Function('substring', expr, Literal(i + 1), Literal(1), alias=alias)


@register_function(
    name='istitle',
    clickhouse_name='match',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Check if string is titlecased.',
)
def _build_istitle(expr, alias=None):
    from .functions import Function
    from .expressions import Literal

    # Check if each word starts with uppercase
    return Function('match', expr, Literal('^([A-Z][a-z]*\\s*)+$'), alias=alias)


@register_function(
    name='casefold',
    clickhouse_name='lower',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Casefold string (aggressive lowercase). Maps to lower(s).',
)
def _build_casefold(expr, alias=None):
    from .functions import Function

    # ClickHouse doesn't have casefold, use lower as approximation
    return Function('lower', expr, alias=alias)


@register_function(
    name='rsplit',
    clickhouse_name='splitByString',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Split string from right. Maps to reverse of splitByString.',
)
def _build_rsplit(expr, sep: str = ' ', maxsplit: int = -1, alias=None):
    from .functions import Function
    from .expressions import Literal

    # ClickHouse doesn't have rsplit, use regular split
    return Function('splitByString', Literal(sep), expr, alias=alias)


@register_function(
    name='str_get',
    clickhouse_name='substring',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=[],  # Removed 'get' to avoid conflict with pandas Series.get()
    doc='Get character at index. Maps to substring(s, i+1, 1).',
)
def _build_str_get(expr, i: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    # Handle negative indices
    if i >= 0:
        return Function('substring', expr, Literal(i + 1), Literal(1), alias=alias)
    else:
        # For negative index, use length + i + 1
        return Function(
            'substring', expr, Function('plus', Function('length', expr), Literal(i + 1)), Literal(1), alias=alias
        )


@register_function(
    name='str_count',
    clickhouse_name='countSubstrings',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Count occurrences of pattern. Maps to countSubstrings(s, pattern).',
)
def _build_str_count(expr, pattern: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('countSubstrings', expr, Literal(pattern), alias=alias)


@register_function(
    name='slice_replace',
    clickhouse_name='concat',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Replace a slice of string with another string.',
)
def _build_slice_replace(expr, start: int = None, stop: int = None, repl: str = '', alias=None):
    from .functions import Function
    from .expressions import Literal

    # slice_replace(s, start, stop, repl) = left(s, start) + repl + substring(s, stop+1)
    if start is None:
        start = 0
    if stop is None:
        # Replace from start to end
        return Function('concat', Function('left', expr, Literal(start)), Literal(repl), alias=alias)
    return Function(
        'concat',
        Function('left', expr, Literal(start)),
        Literal(repl),
        Function('substring', expr, Literal(stop + 1)),
        alias=alias,
    )


@register_function(
    name='str_join',
    clickhouse_name='arrayStringConcat',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Join elements of array column with separator.',
)
def _build_str_join(expr, sep: str = '', alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('arrayStringConcat', expr, Literal(sep), alias=alias)


@register_function(
    name='translate',
    clickhouse_name='translate',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Translate characters using mapping. Maps to translate(s, from, to).',
)
def _build_translate(expr, from_chars: str, to_chars: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('translate', expr, Literal(from_chars), Literal(to_chars), alias=alias)


# =============================================================================
# MORE DATETIME FUNCTIONS (Pandas .dt compatibility)
# =============================================================================


@register_function(
    name='weekday_num',
    clickhouse_name='toDayOfWeek',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['weekday'],
    doc='Day of week (0=Monday, 6=Sunday). Maps to toDayOfWeek(dt) - 1.',
    accessor_only=True,
)
def _build_weekday_num(expr, alias=None):
    from .functions import Function
    from .expressions import Literal

    # Pandas weekday is 0=Monday, ClickHouse toDayOfWeek is 1=Monday
    return Function('minus', Function('toDayOfWeek', expr), Literal(1), alias=alias)


@register_function(
    name='normalize_dt',
    clickhouse_name='toStartOfDay',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    doc='Normalize datetime to midnight. Maps to toStartOfDay(dt).',
    accessor_only=True,
)
def _build_normalize_dt(expr, alias=None):
    from .functions import Function

    return Function('toStartOfDay', expr, alias=alias)


@register_function(
    name='floor_datetime',
    clickhouse_name='toStartOfInterval',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    doc='Floor datetime to frequency. Maps to toStartOfInterval(dt, INTERVAL).',
)
def _build_floor_datetime(expr, freq: str = 'D', alias=None):
    from .functions import Function
    from .expressions import Literal

    freq_map = {
        'D': 'toStartOfDay',
        'H': 'toStartOfHour',
        'T': 'toStartOfMinute',
        'min': 'toStartOfMinute',
        'S': 'toStartOfSecond',
        'W': 'toStartOfWeek',
        'M': 'toStartOfMonth',
        'Q': 'toStartOfQuarter',
        'Y': 'toStartOfYear',
    }
    func_name = freq_map.get(freq.upper(), 'toStartOfDay')
    return Function(func_name, expr, alias=alias)


@register_function(
    name='ceil_datetime',
    clickhouse_name='toStartOfInterval',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    doc='Ceil datetime to frequency.',
)
def _build_ceil_datetime(expr, freq: str = 'D', alias=None):
    from .functions import Function
    from .expressions import Literal

    # Ceil = floor + 1 unit
    freq_map = {
        'D': ('toStartOfDay', 'addDays', 1),
        'H': ('toStartOfHour', 'addHours', 1),
        'T': ('toStartOfMinute', 'addMinutes', 1),
        'min': ('toStartOfMinute', 'addMinutes', 1),
        'S': ('toStartOfSecond', 'addSeconds', 1),
        'W': ('toStartOfWeek', 'addWeeks', 1),
        'M': ('toStartOfMonth', 'addMonths', 1),
        'Q': ('toStartOfQuarter', 'addQuarters', 1),
        'Y': ('toStartOfYear', 'addYears', 1),
    }
    floor_func, add_func, amount = freq_map.get(freq.upper(), ('toStartOfDay', 'addDays', 1))
    return Function(add_func, Function(floor_func, expr), Literal(amount), alias=alias)


@register_function(
    name='round_datetime',
    clickhouse_name='toStartOfInterval',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    doc='Round datetime to nearest frequency.',
)
def _build_round_datetime(expr, freq: str = 'D', alias=None):
    from .functions import Function

    # For simplicity, use floor (proper rounding would require more complex logic)
    freq_map = {
        'D': 'toStartOfDay',
        'H': 'toStartOfHour',
        'T': 'toStartOfMinute',
        'min': 'toStartOfMinute',
        'S': 'toStartOfSecond',
    }
    func_name = freq_map.get(freq.upper(), 'toStartOfDay')
    return Function(func_name, expr, alias=alias)


@register_function(
    name='iso_calendar',
    clickhouse_name='toISOYear',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['isocalendar'],
    doc='ISO year. Maps to toISOYear(dt).',
    accessor_only=True,
)
def _build_iso_calendar(expr, alias=None):
    from .functions import Function

    return Function('toISOYear', expr, alias=alias)


@register_function(
    name='iso_week',
    clickhouse_name='toISOWeek',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toISOWeek'],
    doc='ISO week number. Maps to toISOWeek(dt).',
    accessor_only=True,
)
def _build_iso_week(expr, alias=None):
    from .functions import Function

    return Function('toISOWeek', expr, alias=alias)


# =============================================================================
# HASH FUNCTIONS (Extended)
# =============================================================================


@register_function(
    name='md5',
    clickhouse_name='MD5',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['MD5'],
    doc='Calculate MD5 hash. Maps to MD5(s).',
)
def _build_md5(expr, alias=None):
    from .functions import Function

    return Function('MD5', expr, alias=alias)


@register_function(
    name='sha1',
    clickhouse_name='SHA1',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['SHA1'],
    doc='Calculate SHA1 hash. Maps to SHA1(s).',
)
def _build_sha1(expr, alias=None):
    from .functions import Function

    return Function('SHA1', expr, alias=alias)


@register_function(
    name='sha224',
    clickhouse_name='SHA224',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['SHA224'],
    doc='Calculate SHA224 hash. Maps to SHA224(s).',
)
def _build_sha224(expr, alias=None):
    from .functions import Function

    return Function('SHA224', expr, alias=alias)


@register_function(
    name='sha256',
    clickhouse_name='SHA256',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['SHA256'],
    doc='Calculate SHA256 hash. Maps to SHA256(s).',
)
def _build_sha256(expr, alias=None):
    from .functions import Function

    return Function('SHA256', expr, alias=alias)


@register_function(
    name='sha384',
    clickhouse_name='SHA384',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['SHA384'],
    doc='Calculate SHA384 hash. Maps to SHA384(s).',
)
def _build_sha384(expr, alias=None):
    from .functions import Function

    return Function('SHA384', expr, alias=alias)


@register_function(
    name='sha512',
    clickhouse_name='SHA512',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['SHA512'],
    doc='Calculate SHA512 hash. Maps to SHA512(s).',
)
def _build_sha512(expr, alias=None):
    from .functions import Function

    return Function('SHA512', expr, alias=alias)


@register_function(
    name='xxhash32',
    clickhouse_name='xxHash32',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['xxHash32'],
    doc='Calculate xxHash32. Maps to xxHash32(s).',
)
def _build_xxhash32(expr, alias=None):
    from .functions import Function

    return Function('xxHash32', expr, alias=alias)


@register_function(
    name='xxhash64',
    clickhouse_name='xxHash64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['xxHash64'],
    doc='Calculate xxHash64. Maps to xxHash64(s).',
)
def _build_xxhash64(expr, alias=None):
    from .functions import Function

    return Function('xxHash64', expr, alias=alias)


@register_function(
    name='siphash64',
    clickhouse_name='sipHash64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['sipHash64'],
    doc='Calculate sipHash64. Maps to sipHash64(s).',
)
def _build_siphash64(expr, alias=None):
    from .functions import Function

    return Function('sipHash64', expr, alias=alias)


@register_function(
    name='siphash128',
    clickhouse_name='sipHash128',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['sipHash128'],
    doc='Calculate sipHash128. Maps to sipHash128(s).',
)
def _build_siphash128(expr, alias=None):
    from .functions import Function

    return Function('sipHash128', expr, alias=alias)


@register_function(
    name='cityhash64',
    clickhouse_name='cityHash64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['cityHash64'],
    doc='Calculate cityHash64. Maps to cityHash64(s).',
)
def _build_cityhash64(expr, alias=None):
    from .functions import Function

    return Function('cityHash64', expr, alias=alias)


@register_function(
    name='farmhash64',
    clickhouse_name='farmHash64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['farmHash64'],
    doc='Calculate farmHash64. Maps to farmHash64(s).',
)
def _build_farmhash64(expr, alias=None):
    from .functions import Function

    return Function('farmHash64', expr, alias=alias)


@register_function(
    name='metrohash64',
    clickhouse_name='metroHash64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['metroHash64'],
    doc='Calculate metroHash64. Maps to metroHash64(s).',
)
def _build_metrohash64(expr, alias=None):
    from .functions import Function

    return Function('metroHash64', expr, alias=alias)


@register_function(
    name='murmurhash2_32',
    clickhouse_name='murmurHash2_32',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['murmurHash2_32'],
    doc='Calculate murmurHash2 32-bit. Maps to murmurHash2_32(s).',
)
def _build_murmurhash2_32(expr, alias=None):
    from .functions import Function

    return Function('murmurHash2_32', expr, alias=alias)


@register_function(
    name='murmurhash2_64',
    clickhouse_name='murmurHash2_64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['murmurHash2_64'],
    doc='Calculate murmurHash2 64-bit. Maps to murmurHash2_64(s).',
)
def _build_murmurhash2_64(expr, alias=None):
    from .functions import Function

    return Function('murmurHash2_64', expr, alias=alias)


@register_function(
    name='murmurhash3_32',
    clickhouse_name='murmurHash3_32',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['murmurHash3_32'],
    doc='Calculate murmurHash3 32-bit. Maps to murmurHash3_32(s).',
)
def _build_murmurhash3_32(expr, alias=None):
    from .functions import Function

    return Function('murmurHash3_32', expr, alias=alias)


@register_function(
    name='murmurhash3_64',
    clickhouse_name='murmurHash3_64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['murmurHash3_64'],
    doc='Calculate murmurHash3 64-bit. Maps to murmurHash3_64(s).',
)
def _build_murmurhash3_64(expr, alias=None):
    from .functions import Function

    return Function('murmurHash3_64', expr, alias=alias)


@register_function(
    name='murmurhash3_128',
    clickhouse_name='murmurHash3_128',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['murmurHash3_128'],
    doc='Calculate murmurHash3 128-bit. Maps to murmurHash3_128(s).',
)
def _build_murmurhash3_128(expr, alias=None):
    from .functions import Function

    return Function('murmurHash3_128', expr, alias=alias)


@register_function(
    name='javahash',
    clickhouse_name='javaHash',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['javaHash'],
    doc='Calculate Java hash. Maps to javaHash(s).',
)
def _build_javahash(expr, alias=None):
    from .functions import Function

    return Function('javaHash', expr, alias=alias)


@register_function(
    name='inthash32',
    clickhouse_name='intHash32',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['intHash32'],
    doc='Hash integer to 32-bit. Maps to intHash32(n).',
)
def _build_inthash32(expr, alias=None):
    from .functions import Function

    return Function('intHash32', expr, alias=alias)


@register_function(
    name='inthash64',
    clickhouse_name='intHash64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['intHash64'],
    doc='Hash integer to 64-bit. Maps to intHash64(n).',
)
def _build_inthash64(expr, alias=None):
    from .functions import Function

    return Function('intHash64', expr, alias=alias)


@register_function(
    name='halfmd5',
    clickhouse_name='halfMD5',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['halfMD5'],
    doc='Calculate half MD5 hash. Maps to halfMD5(s).',
)
def _build_halfmd5(expr, alias=None):
    from .functions import Function

    return Function('halfMD5', expr, alias=alias)


@register_function(
    name='crc32',
    clickhouse_name='CRC32',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['CRC32'],
    doc='Calculate CRC32. Maps to CRC32(s).',
)
def _build_crc32(expr, alias=None):
    from .functions import Function

    return Function('CRC32', expr, alias=alias)


@register_function(
    name='crc64',
    clickhouse_name='CRC64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['CRC64'],
    doc='Calculate CRC64. Maps to CRC64(s).',
)
def _build_crc64(expr, alias=None):
    from .functions import Function

    return Function('CRC64', expr, alias=alias)


@register_function(
    name='wyhash64',
    clickhouse_name='wyHash64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['wyHash64'],
    doc='Calculate wyHash64. Maps to wyHash64(s).',
)
def _build_wyhash64(expr, alias=None):
    from .functions import Function

    return Function('wyHash64', expr, alias=alias)


@register_function(
    name='urlhash',
    clickhouse_name='URLHash',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['URLHash'],
    doc='Calculate URL hash. Maps to URLHash(url, n).',
)
def _build_urlhash(url, n: int = 0, alias=None):
    from .functions import Function
    from .expressions import Literal

    if n:
        return Function('URLHash', url, Literal(n), alias=alias)
    return Function('URLHash', url, alias=alias)


# =============================================================================
# MORE ARRAY FUNCTIONS
# =============================================================================


@register_function(
    name='array_zip',
    clickhouse_name='arrayZip',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayZip'],
    doc='Zip multiple arrays. Maps to arrayZip(arr1, arr2, ...).',
)
def _build_array_zip(*arrays, alias=None):
    from .functions import Function

    return Function('arrayZip', *arrays, alias=alias)


@register_function(
    name='array_uniq',
    clickhouse_name='arrayUniq',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayUniq'],
    doc='Count unique elements in array. Maps to arrayUniq(arr).',
)
def _build_array_uniq(arr, alias=None):
    from .functions import Function

    return Function('arrayUniq', arr, alias=alias)


@register_function(
    name='array_reduce',
    clickhouse_name='arrayReduce',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayReduce'],
    doc='Apply aggregate function to array. Maps to arrayReduce(func, arr).',
)
def _build_array_reduce(func: str, arr, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('arrayReduce', Literal(func), arr, alias=alias)


@register_function(
    name='array_fold',
    clickhouse_name='arrayFold',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayFold'],
    doc='Fold array with lambda. Maps to arrayFold(lambda, arr, init).',
)
def _build_array_fold(lambda_expr, arr, init, alias=None):
    from .functions import Function

    return Function('arrayFold', lambda_expr, arr, init, alias=alias)


@register_function(
    name='array_difference',
    clickhouse_name='arrayDifference',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayDifference'],
    doc='Calculate differences between consecutive elements. Maps to arrayDifference(arr).',
)
def _build_array_difference(arr, alias=None):
    from .functions import Function

    return Function('arrayDifference', arr, alias=alias)


@register_function(
    name='array_cum_sum',
    clickhouse_name='arrayCumSum',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayCumSum'],
    doc='Calculate cumulative sum. Maps to arrayCumSum(arr).',
)
def _build_array_cum_sum(arr, alias=None):
    from .functions import Function

    return Function('arrayCumSum', arr, alias=alias)


@register_function(
    name='array_cum_sum_non_negative',
    clickhouse_name='arrayCumSumNonNegative',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayCumSumNonNegative'],
    doc='Calculate cumulative sum (non-negative). Maps to arrayCumSumNonNegative(arr).',
)
def _build_array_cum_sum_non_negative(arr, alias=None):
    from .functions import Function

    return Function('arrayCumSumNonNegative', arr, alias=alias)


@register_function(
    name='array_product',
    clickhouse_name='arrayProduct',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayProduct'],
    doc='Calculate product of array elements. Maps to arrayProduct(arr).',
)
def _build_array_product(arr, alias=None):
    from .functions import Function

    return Function('arrayProduct', arr, alias=alias)


@register_function(
    name='array_avg',
    clickhouse_name='arrayAvg',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayAvg'],
    doc='Calculate average of array elements. Maps to arrayAvg(arr).',
)
def _build_array_avg(arr, alias=None):
    from .functions import Function

    return Function('arrayAvg', arr, alias=alias)


@register_function(
    name='array_min',
    clickhouse_name='arrayMin',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayMin'],
    doc='Get minimum of array. Maps to arrayMin(arr).',
)
def _build_array_min(arr, alias=None):
    from .functions import Function

    return Function('arrayMin', arr, alias=alias)


@register_function(
    name='array_max',
    clickhouse_name='arrayMax',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayMax'],
    doc='Get maximum of array. Maps to arrayMax(arr).',
)
def _build_array_max(arr, alias=None):
    from .functions import Function

    return Function('arrayMax', arr, alias=alias)


@register_function(
    name='array_sum',
    clickhouse_name='arraySum',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arraySum'],
    doc='Calculate sum of array elements. Maps to arraySum(arr).',
)
def _build_array_sum(arr, alias=None):
    from .functions import Function

    return Function('arraySum', arr, alias=alias)


@register_function(
    name='array_count',
    clickhouse_name='arrayCount',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayCount'],
    doc='Count elements matching condition. Maps to arrayCount(lambda, arr).',
)
def _build_array_count(arr, lambda_expr=None, alias=None):
    from .functions import Function

    if lambda_expr:
        return Function('arrayCount', lambda_expr, arr, alias=alias)
    return Function('arrayCount', arr, alias=alias)


@register_function(
    name='array_exists',
    clickhouse_name='arrayExists',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayExists'],
    doc='Check if any element matches condition. Maps to arrayExists(lambda, arr).',
)
def _build_array_exists(lambda_expr, arr, alias=None):
    from .functions import Function

    return Function('arrayExists', lambda_expr, arr, alias=alias)


@register_function(
    name='array_all',
    clickhouse_name='arrayAll',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayAll'],
    doc='Check if all elements match condition. Maps to arrayAll(lambda, arr).',
)
def _build_array_all(lambda_expr, arr, alias=None):
    from .functions import Function

    return Function('arrayAll', lambda_expr, arr, alias=alias)


@register_function(
    name='array_first',
    clickhouse_name='arrayFirst',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayFirst'],
    doc='Get first element matching condition. Maps to arrayFirst(lambda, arr).',
)
def _build_array_first(lambda_expr, arr, alias=None):
    from .functions import Function

    return Function('arrayFirst', lambda_expr, arr, alias=alias)


@register_function(
    name='array_last',
    clickhouse_name='arrayLast',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayLast'],
    doc='Get last element matching condition. Maps to arrayLast(lambda, arr).',
)
def _build_array_last(lambda_expr, arr, alias=None):
    from .functions import Function

    return Function('arrayLast', lambda_expr, arr, alias=alias)


@register_function(
    name='array_first_index',
    clickhouse_name='arrayFirstIndex',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayFirstIndex'],
    doc='Get index of first element matching condition. Maps to arrayFirstIndex(lambda, arr).',
)
def _build_array_first_index(lambda_expr, arr, alias=None):
    from .functions import Function

    return Function('arrayFirstIndex', lambda_expr, arr, alias=alias)


@register_function(
    name='array_last_index',
    clickhouse_name='arrayLastIndex',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayLastIndex'],
    doc='Get index of last element matching condition. Maps to arrayLastIndex(lambda, arr).',
)
def _build_array_last_index(lambda_expr, arr, alias=None):
    from .functions import Function

    return Function('arrayLastIndex', lambda_expr, arr, alias=alias)


@register_function(
    name='array_fill',
    clickhouse_name='arrayFill',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayFill'],
    doc='Fill array with forward fill. Maps to arrayFill(lambda, arr).',
)
def _build_array_fill(lambda_expr, arr, alias=None):
    from .functions import Function

    return Function('arrayFill', lambda_expr, arr, alias=alias)


@register_function(
    name='array_reverse_fill',
    clickhouse_name='arrayReverseFill',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayReverseFill'],
    doc='Fill array with backward fill. Maps to arrayReverseFill(lambda, arr).',
)
def _build_array_reverse_fill(lambda_expr, arr, alias=None):
    from .functions import Function

    return Function('arrayReverseFill', lambda_expr, arr, alias=alias)


@register_function(
    name='array_split',
    clickhouse_name='arraySplit',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arraySplit'],
    doc='Split array by condition. Maps to arraySplit(lambda, arr).',
)
def _build_array_split(lambda_expr, arr, alias=None):
    from .functions import Function

    return Function('arraySplit', lambda_expr, arr, alias=alias)


@register_function(
    name='array_reverse_split',
    clickhouse_name='arrayReverseSplit',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayReverseSplit'],
    doc='Split array by condition (reverse). Maps to arrayReverseSplit(lambda, arr).',
)
def _build_array_reverse_split(lambda_expr, arr, alias=None):
    from .functions import Function

    return Function('arrayReverseSplit', lambda_expr, arr, alias=alias)


@register_function(
    name='array_compact',
    clickhouse_name='arrayCompact',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayCompact'],
    doc='Remove consecutive duplicates. Maps to arrayCompact(arr).',
)
def _build_array_compact(arr, alias=None):
    from .functions import Function

    return Function('arrayCompact', arr, alias=alias)


@register_function(
    name='array_flatten',
    clickhouse_name='arrayFlatten',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayFlatten'],
    doc='Flatten nested arrays. Maps to arrayFlatten(arr).',
)
def _build_array_flatten(arr, alias=None):
    from .functions import Function

    return Function('arrayFlatten', arr, alias=alias)


@register_function(
    name='array_pop_back',
    clickhouse_name='arrayPopBack',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayPopBack'],
    doc='Remove last element. Maps to arrayPopBack(arr).',
)
def _build_array_pop_back(arr, alias=None):
    from .functions import Function

    return Function('arrayPopBack', arr, alias=alias)


@register_function(
    name='array_pop_front',
    clickhouse_name='arrayPopFront',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayPopFront'],
    doc='Remove first element. Maps to arrayPopFront(arr).',
)
def _build_array_pop_front(arr, alias=None):
    from .functions import Function

    return Function('arrayPopFront', arr, alias=alias)


@register_function(
    name='array_push_back',
    clickhouse_name='arrayPushBack',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayPushBack'],
    doc='Add element to end. Maps to arrayPushBack(arr, elem).',
)
def _build_array_push_back(arr, elem, alias=None):
    from .functions import Function

    return Function('arrayPushBack', arr, elem, alias=alias)


@register_function(
    name='array_push_front',
    clickhouse_name='arrayPushFront',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayPushFront'],
    doc='Add element to front. Maps to arrayPushFront(arr, elem).',
)
def _build_array_push_front(arr, elem, alias=None):
    from .functions import Function

    return Function('arrayPushFront', arr, elem, alias=alias)


@register_function(
    name='array_resize',
    clickhouse_name='arrayResize',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayResize'],
    doc='Resize array. Maps to arrayResize(arr, size, default).',
)
def _build_array_resize(arr, size: int, default=None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if default is not None:
        return Function('arrayResize', arr, Literal(size), default, alias=alias)
    return Function('arrayResize', arr, Literal(size), alias=alias)


@register_function(
    name='array_shuffle',
    clickhouse_name='arrayShuffle',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayShuffle'],
    doc='Shuffle array randomly. Maps to arrayShuffle(arr).',
)
def _build_array_shuffle(arr, alias=None):
    from .functions import Function

    return Function('arrayShuffle', arr, alias=alias)


@register_function(
    name='array_partial_shuffle',
    clickhouse_name='arrayPartialShuffle',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayPartialShuffle'],
    doc='Partially shuffle array. Maps to arrayPartialShuffle(arr, limit).',
)
def _build_array_partial_shuffle(arr, limit: int = None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if limit is not None:
        return Function('arrayPartialShuffle', arr, Literal(limit), alias=alias)
    return Function('arrayPartialShuffle', arr, alias=alias)


# =============================================================================
# MORE AGGREGATE FUNCTIONS
# =============================================================================


@register_function(
    name='any_heavy',
    clickhouse_name='anyHeavy',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['anyHeavy'],
    doc='Get frequently occurring value. Maps to anyHeavy(x).',
)
def _build_any_heavy(expr, alias=None):
    from .functions import Function

    return Function('anyHeavy', expr, alias=alias)


@register_function(
    name='any_last',
    clickhouse_name='anyLast',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['anyLast'],
    doc='Get last value. Maps to anyLast(x).',
)
def _build_any_last(expr, alias=None):
    from .functions import Function

    return Function('anyLast', expr, alias=alias)


@register_function(
    name='arg_min',
    clickhouse_name='argMin',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['argMin'],
    doc='Get arg at minimum. Maps to argMin(arg, val).',
)
def _build_arg_min(arg, val, alias=None):
    from .functions import Function

    return Function('argMin', arg, val, alias=alias)


@register_function(
    name='arg_max',
    clickhouse_name='argMax',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['argMax'],
    doc='Get arg at maximum. Maps to argMax(arg, val).',
)
def _build_arg_max(arg, val, alias=None):
    from .functions import Function

    return Function('argMax', arg, val, alias=alias)


@register_function(
    name='top_k',
    clickhouse_name='topK',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['topK'],
    doc='Get top K frequent values. Maps to topK(k)(x).',
)
def _build_top_k(expr, k: int = 10, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function(f'topK({k})', expr, alias=alias)


@register_function(
    name='top_k_weighted',
    clickhouse_name='topKWeighted',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['topKWeighted'],
    doc='Get top K weighted values. Maps to topKWeighted(k)(x, weight).',
)
def _build_top_k_weighted(expr, weight, k: int = 10, alias=None):
    from .functions import Function

    return Function(f'topKWeighted({k})', expr, weight, alias=alias)


@register_function(
    name='group_array',
    clickhouse_name='groupArray',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['groupArray'],
    doc='Collect values into array. Maps to groupArray(x).',
)
def _build_group_array(expr, max_size: int = None, alias=None):
    from .functions import Function

    if max_size:
        return Function(f'groupArray({max_size})', expr, alias=alias)
    return Function('groupArray', expr, alias=alias)


@register_function(
    name='group_uniq_array',
    clickhouse_name='groupUniqArray',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['groupUniqArray'],
    doc='Collect unique values into array. Maps to groupUniqArray(x).',
)
def _build_group_uniq_array(expr, max_size: int = None, alias=None):
    from .functions import Function

    if max_size:
        return Function(f'groupUniqArray({max_size})', expr, alias=alias)
    return Function('groupUniqArray', expr, alias=alias)


@register_function(
    name='group_array_insert_at',
    clickhouse_name='groupArrayInsertAt',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['groupArrayInsertAt'],
    doc='Insert values at positions. Maps to groupArrayInsertAt(x, pos).',
)
def _build_group_array_insert_at(value, pos, default=None, alias=None):
    from .functions import Function

    if default is not None:
        return Function('groupArrayInsertAt', value, pos, default, alias=alias)
    return Function('groupArrayInsertAt', value, pos, alias=alias)


@register_function(
    name='group_bitmap',
    clickhouse_name='groupBitmap',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['groupBitmap'],
    doc='Create bitmap from values. Maps to groupBitmap(x).',
)
def _build_group_bitmap(expr, alias=None):
    from .functions import Function

    return Function('groupBitmap', expr, alias=alias)


@register_function(
    name='simplelinear_regression',
    clickhouse_name='simpleLinearRegression',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['simpleLinearRegression'],
    doc='Simple linear regression. Maps to simpleLinearRegression(x, y).',
)
def _build_simplelinear_regression(x, y, alias=None):
    from .functions import Function

    return Function('simpleLinearRegression', x, y, alias=alias)


@register_function(
    name='stochasticlinear_regression',
    clickhouse_name='stochasticLinearRegression',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['stochasticLinearRegression'],
    doc='Stochastic linear regression. Maps to stochasticLinearRegression(...).',
)
def _build_stochasticlinear_regression(*args, alias=None):
    from .functions import Function

    return Function('stochasticLinearRegression', *args, alias=alias)


@register_function(
    name='stochasticlogistic_regression',
    clickhouse_name='stochasticLogisticRegression',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['stochasticLogisticRegression'],
    doc='Stochastic logistic regression. Maps to stochasticLogisticRegression(...).',
)
def _build_stochasticlogistic_regression(*args, alias=None):
    from .functions import Function

    return Function('stochasticLogisticRegression', *args, alias=alias)


@register_function(
    name='corr',
    clickhouse_name='corr',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['corrStable'],
    doc='Calculate correlation. Maps to corr(x, y).',
)
def _build_corr(x, y, alias=None):
    from .functions import Function

    return Function('corr', x, y, alias=alias)


@register_function(
    name='covar_samp',
    clickhouse_name='covarSamp',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['covarSamp', 'covarSampStable'],
    doc='Calculate sample covariance. Maps to covarSamp(x, y).',
)
def _build_covar_samp(x, y, alias=None):
    from .functions import Function

    return Function('covarSamp', x, y, alias=alias)


@register_function(
    name='covar_pop',
    clickhouse_name='covarPop',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['covarPop', 'covarPopStable'],
    doc='Calculate population covariance. Maps to covarPop(x, y).',
)
def _build_covar_pop(x, y, alias=None):
    from .functions import Function

    return Function('covarPop', x, y, alias=alias)


@register_function(
    name='entropy',
    clickhouse_name='entropy',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['entropyStable'],
    doc='Calculate Shannon entropy. Maps to entropy(x).',
)
def _build_entropy(expr, alias=None):
    from .functions import Function

    return Function('entropy', expr, alias=alias)


@register_function(
    name='kurtosis',
    clickhouse_name='kurtSamp',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['kurtSamp', 'kurtosis'],
    doc='Calculate sample kurtosis. Maps to kurtSamp(x).',
)
def _build_kurtosis(expr, alias=None):
    from .functions import Function

    return Function('kurtSamp', expr, alias=alias)


@register_function(
    name='skewness',
    clickhouse_name='skewSamp',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['skewSamp', 'skewness'],
    doc='Calculate sample skewness. Maps to skewSamp(x).',
)
def _build_skewness(expr, alias=None):
    from .functions import Function

    return Function('skewSamp', expr, alias=alias)


@register_function(
    name='uniq_exact',
    clickhouse_name='uniqExact',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['uniqExact'],
    doc='Exact unique count. Maps to uniqExact(x).',
)
def _build_uniq_exact(expr, alias=None):
    from .functions import Function

    return Function('uniqExact', expr, alias=alias)


@register_function(
    name='uniq_combined',
    clickhouse_name='uniqCombined',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['uniqCombined'],
    doc='Approximate unique count (combined). Maps to uniqCombined(x).',
)
def _build_uniq_combined(expr, alias=None):
    from .functions import Function

    return Function('uniqCombined', expr, alias=alias)


@register_function(
    name='uniq_combined64',
    clickhouse_name='uniqCombined64',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['uniqCombined64'],
    doc='Approximate unique count (64-bit). Maps to uniqCombined64(x).',
)
def _build_uniq_combined64(expr, alias=None):
    from .functions import Function

    return Function('uniqCombined64', expr, alias=alias)


@register_function(
    name='uniq_hll12',
    clickhouse_name='uniqHLL12',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['uniqHLL12'],
    doc='Approximate unique count (HyperLogLog). Maps to uniqHLL12(x).',
)
def _build_uniq_hll12(expr, alias=None):
    from .functions import Function

    return Function('uniqHLL12', expr, alias=alias)


@register_function(
    name='uniq_theta',
    clickhouse_name='uniqTheta',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['uniqTheta'],
    doc='Approximate unique count (Theta sketch). Maps to uniqTheta(x).',
)
def _build_uniq_theta(expr, alias=None):
    from .functions import Function

    return Function('uniqTheta', expr, alias=alias)


@register_function(
    name='histogram',
    clickhouse_name='histogram',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['histogramAgg'],
    doc='Build histogram. Maps to histogram(num_bins)(x).',
)
def _build_histogram(expr, num_bins: int = 10, alias=None):
    from .functions import Function

    return Function(f'histogram({num_bins})', expr, alias=alias)


@register_function(
    name='quantile_timing',
    clickhouse_name='quantileTiming',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['quantileTiming'],
    doc='Quantile for timing data. Maps to quantileTiming(level)(x).',
)
def _build_quantile_timing(expr, level: float = 0.5, alias=None):
    from .functions import Function

    return Function(f'quantileTiming({level})', expr, alias=alias)


@register_function(
    name='quantile_exact',
    clickhouse_name='quantileExact',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['quantileExact'],
    doc='Exact quantile. Maps to quantileExact(level)(x).',
)
def _build_quantile_exact(expr, level: float = 0.5, alias=None):
    from .functions import Function

    return Function(f'quantileExact({level})', expr, alias=alias)


@register_function(
    name='quantile_exact_weighted',
    clickhouse_name='quantileExactWeighted',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['quantileExactWeighted'],
    doc='Weighted exact quantile. Maps to quantileExactWeighted(level)(x, weight).',
)
def _build_quantile_exact_weighted(expr, weight, level: float = 0.5, alias=None):
    from .functions import Function

    return Function(f'quantileExactWeighted({level})', expr, weight, alias=alias)


@register_function(
    name='quantile_tdigest',
    clickhouse_name='quantileTDigest',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['quantileTDigest'],
    doc='Quantile using t-digest. Maps to quantileTDigest(level)(x).',
)
def _build_quantile_tdigest(expr, level: float = 0.5, alias=None):
    from .functions import Function

    return Function(f'quantileTDigest({level})', expr, alias=alias)


@register_function(
    name='quantile_bfloat16',
    clickhouse_name='quantileBFloat16',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['quantileBFloat16'],
    doc='Quantile using bfloat16. Maps to quantileBFloat16(level)(x).',
)
def _build_quantile_bfloat16(expr, level: float = 0.5, alias=None):
    from .functions import Function

    return Function(f'quantileBFloat16({level})', expr, alias=alias)


# =============================================================================
# MATHEMATICAL FUNCTIONS (Extended)
# =============================================================================


@register_function(
    name='e',
    clickhouse_name='e',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['euler'],
    doc='Euler number e. Maps to e().',
)
def _build_e(alias=None):
    from .functions import Function

    return Function('e', alias=alias)


@register_function(
    name='pi',
    clickhouse_name='pi',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['pi_value'],
    doc='Pi constant. Maps to pi().',
)
def _build_pi(alias=None):
    from .functions import Function

    return Function('pi', alias=alias)


@register_function(
    name='exp2',
    clickhouse_name='exp2',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['exp2'],
    doc='2^x. Maps to exp2(x).',
)
def _build_exp2(expr, alias=None):
    from .functions import Function

    return Function('exp2', expr, alias=alias)


@register_function(
    name='exp10',
    clickhouse_name='exp10',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['exp10'],
    doc='10^x. Maps to exp10(x).',
)
def _build_exp10(expr, alias=None):
    from .functions import Function

    return Function('exp10', expr, alias=alias)


@register_function(
    name='log2',
    clickhouse_name='log2',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['log2'],
    doc='Log base 2. Maps to log2(x).',
)
def _build_log2(expr, alias=None):
    from .functions import Function

    return Function('log2', expr, alias=alias)


@register_function(
    name='log1p',
    clickhouse_name='log1p',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['log1p'],
    doc='log(1 + x). Maps to log1p(x).',
)
def _build_log1p(expr, alias=None):
    from .functions import Function

    return Function('log1p', expr, alias=alias)


@register_function(
    name='cbrt',
    clickhouse_name='cbrt',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['cbrt'],
    doc='Cube root. Maps to cbrt(x).',
)
def _build_cbrt(expr, alias=None):
    from .functions import Function

    return Function('cbrt', expr, alias=alias)


@register_function(
    name='erf',
    clickhouse_name='erf',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['erf'],
    doc='Error function. Maps to erf(x).',
)
def _build_erf(expr, alias=None):
    from .functions import Function

    return Function('erf', expr, alias=alias)


@register_function(
    name='erfc',
    clickhouse_name='erfc',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['erfc'],
    doc='Complementary error function. Maps to erfc(x).',
)
def _build_erfc(expr, alias=None):
    from .functions import Function

    return Function('erfc', expr, alias=alias)


@register_function(
    name='lgamma',
    clickhouse_name='lgamma',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['lgamma'],
    doc='Log gamma function. Maps to lgamma(x).',
)
def _build_lgamma(expr, alias=None):
    from .functions import Function

    return Function('lgamma', expr, alias=alias)


@register_function(
    name='tgamma',
    clickhouse_name='tgamma',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['tgamma', 'gamma'],
    doc='Gamma function. Maps to tgamma(x).',
)
def _build_tgamma(expr, alias=None):
    from .functions import Function

    return Function('tgamma', expr, alias=alias)


@register_function(
    name='asinh',
    clickhouse_name='asinh',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['asinh'],
    doc='Inverse hyperbolic sine. Maps to asinh(x).',
)
def _build_asinh(expr, alias=None):
    from .functions import Function

    return Function('asinh', expr, alias=alias)


@register_function(
    name='acosh',
    clickhouse_name='acosh',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['acosh'],
    doc='Inverse hyperbolic cosine. Maps to acosh(x).',
)
def _build_acosh(expr, alias=None):
    from .functions import Function

    return Function('acosh', expr, alias=alias)


@register_function(
    name='atanh',
    clickhouse_name='atanh',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['atanh'],
    doc='Inverse hyperbolic tangent. Maps to atanh(x).',
)
def _build_atanh(expr, alias=None):
    from .functions import Function

    return Function('atanh', expr, alias=alias)


@register_function(
    name='sinh',
    clickhouse_name='sinh',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['sinh'],
    doc='Hyperbolic sine. Maps to sinh(x).',
)
def _build_sinh(expr, alias=None):
    from .functions import Function

    return Function('sinh', expr, alias=alias)


@register_function(
    name='cosh',
    clickhouse_name='cosh',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['cosh'],
    doc='Hyperbolic cosine. Maps to cosh(x).',
)
def _build_cosh(expr, alias=None):
    from .functions import Function

    return Function('cosh', expr, alias=alias)


@register_function(
    name='tanh',
    clickhouse_name='tanh',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['tanh'],
    doc='Hyperbolic tangent. Maps to tanh(x).',
)
def _build_tanh(expr, alias=None):
    from .functions import Function

    return Function('tanh', expr, alias=alias)


@register_function(
    name='hypot',
    clickhouse_name='hypot',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['hypot'],
    doc='Hypotenuse. Maps to hypot(x, y).',
)
def _build_hypot(x, y, alias=None):
    from .functions import Function

    return Function('hypot', x, y, alias=alias)


@register_function(
    name='degrees',
    clickhouse_name='degrees',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['degrees', 'toDegrees'],
    doc='Radians to degrees. Maps to degrees(x).',
)
def _build_degrees(expr, alias=None):
    from .functions import Function

    return Function('degrees', expr, alias=alias)


@register_function(
    name='radians',
    clickhouse_name='radians',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['radians', 'toRadians'],
    doc='Degrees to radians. Maps to radians(x).',
)
def _build_radians(expr, alias=None):
    from .functions import Function

    return Function('radians', expr, alias=alias)


@register_function(
    name='gcd',
    clickhouse_name='gcd',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['gcd'],
    doc='Greatest common divisor. Maps to gcd(a, b).',
)
def _build_gcd(a, b, alias=None):
    from .functions import Function

    return Function('gcd', a, b, alias=alias)


@register_function(
    name='lcm',
    clickhouse_name='lcm',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['lcm'],
    doc='Least common multiple. Maps to lcm(a, b).',
)
def _build_lcm(a, b, alias=None):
    from .functions import Function

    return Function('lcm', a, b, alias=alias)


# =============================================================================
# TYPE CONVERSION FUNCTIONS (Extended)
# =============================================================================


@register_function(
    name='to_string',
    clickhouse_name='toString',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toString'],
    doc='Convert to string. Maps to toString(x).',
)
def _build_to_string(expr, alias=None):
    from .functions import Function

    return Function('toString', expr, alias=alias)


@register_function(
    name='to_fixed_string',
    clickhouse_name='toFixedString',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toFixedString'],
    doc='Convert to fixed-length string. Maps to toFixedString(s, n).',
)
def _build_to_fixed_string(expr, n: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('toFixedString', expr, Literal(n), alias=alias)


@register_function(
    name='to_int8',
    clickhouse_name='toInt8',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toInt8'],
    doc='Convert to Int8. Maps to toInt8(x).',
)
def _build_to_int8(expr, alias=None):
    from .functions import Function

    return Function('toInt8', expr, alias=alias)


@register_function(
    name='to_int16',
    clickhouse_name='toInt16',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toInt16'],
    doc='Convert to Int16. Maps to toInt16(x).',
)
def _build_to_int16(expr, alias=None):
    from .functions import Function

    return Function('toInt16', expr, alias=alias)


@register_function(
    name='to_int32',
    clickhouse_name='toInt32',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toInt32'],
    doc='Convert to Int32. Maps to toInt32(x).',
)
def _build_to_int32(expr, alias=None):
    from .functions import Function

    return Function('toInt32', expr, alias=alias)


@register_function(
    name='to_int64',
    clickhouse_name='toInt64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toInt64'],
    doc='Convert to Int64. Maps to toInt64(x).',
)
def _build_to_int64(expr, alias=None):
    from .functions import Function

    return Function('toInt64', expr, alias=alias)


@register_function(
    name='to_uint8',
    clickhouse_name='toUInt8',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toUInt8'],
    doc='Convert to UInt8. Maps to toUInt8(x).',
)
def _build_to_uint8(expr, alias=None):
    from .functions import Function

    return Function('toUInt8', expr, alias=alias)


@register_function(
    name='to_uint16',
    clickhouse_name='toUInt16',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toUInt16'],
    doc='Convert to UInt16. Maps to toUInt16(x).',
)
def _build_to_uint16(expr, alias=None):
    from .functions import Function

    return Function('toUInt16', expr, alias=alias)


@register_function(
    name='to_uint32',
    clickhouse_name='toUInt32',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toUInt32'],
    doc='Convert to UInt32. Maps to toUInt32(x).',
)
def _build_to_uint32(expr, alias=None):
    from .functions import Function

    return Function('toUInt32', expr, alias=alias)


@register_function(
    name='to_uint64',
    clickhouse_name='toUInt64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toUInt64'],
    doc='Convert to UInt64. Maps to toUInt64(x).',
)
def _build_to_uint64(expr, alias=None):
    from .functions import Function

    return Function('toUInt64', expr, alias=alias)


@register_function(
    name='to_float32',
    clickhouse_name='toFloat32',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toFloat32'],
    doc='Convert to Float32. Maps to toFloat32(x).',
)
def _build_to_float32(expr, alias=None):
    from .functions import Function

    return Function('toFloat32', expr, alias=alias)


@register_function(
    name='to_float64',
    clickhouse_name='toFloat64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toFloat64'],
    doc='Convert to Float64. Maps to toFloat64(x).',
)
def _build_to_float64(expr, alias=None):
    from .functions import Function

    return Function('toFloat64', expr, alias=alias)


@register_function(
    name='to_decimal32',
    clickhouse_name='toDecimal32',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toDecimal32'],
    doc='Convert to Decimal32. Maps to toDecimal32(x, scale).',
)
def _build_to_decimal32(expr, scale: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('toDecimal32', expr, Literal(scale), alias=alias)


@register_function(
    name='to_decimal64',
    clickhouse_name='toDecimal64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toDecimal64'],
    doc='Convert to Decimal64. Maps to toDecimal64(x, scale).',
)
def _build_to_decimal64(expr, scale: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('toDecimal64', expr, Literal(scale), alias=alias)


@register_function(
    name='to_decimal128',
    clickhouse_name='toDecimal128',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toDecimal128'],
    doc='Convert to Decimal128. Maps to toDecimal128(x, scale).',
)
def _build_to_decimal128(expr, scale: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('toDecimal128', expr, Literal(scale), alias=alias)


@register_function(
    name='reinterpret_as_string',
    clickhouse_name='reinterpretAsString',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpretAsString'],
    doc='Reinterpret as string. Maps to reinterpretAsString(x).',
)
def _build_reinterpret_as_string(expr, alias=None):
    from .functions import Function

    return Function('reinterpretAsString', expr, alias=alias)


@register_function(
    name='reinterpret_as_int64',
    clickhouse_name='reinterpretAsInt64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpretAsInt64'],
    doc='Reinterpret as Int64. Maps to reinterpretAsInt64(x).',
)
def _build_reinterpret_as_int64(expr, alias=None):
    from .functions import Function

    return Function('reinterpretAsInt64', expr, alias=alias)


# =============================================================================
# LOGICAL AND COMPARISON FUNCTIONS
# =============================================================================


@register_function(
    name='if_func',
    clickhouse_name='if',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['if_', 'ifFunc'],
    doc='Conditional. Maps to if(cond, then, else).',
)
def _build_if_func(cond, then_val, else_val, alias=None):
    from .functions import Function

    return Function('if', cond, then_val, else_val, alias=alias)


@register_function(
    name='multi_if',
    clickhouse_name='multiIf',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['multiIf'],
    doc='Multiple conditions. Maps to multiIf(cond1, val1, cond2, val2, ..., default).',
)
def _build_multi_if(*args, alias=None):
    from .functions import Function

    return Function('multiIf', *args, alias=alias)


@register_function(
    name='null_if',
    clickhouse_name='nullIf',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['nullIf'],
    doc='Return NULL if equal. Maps to nullIf(x, y).',
)
def _build_null_if(x, y, alias=None):
    from .functions import Function

    return Function('nullIf', x, y, alias=alias)


@register_function(
    name='if_null',
    clickhouse_name='ifNull',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['ifNull'],
    doc='Return value if not NULL. Maps to ifNull(x, alt).',
)
def _build_if_null(x, alt, alias=None):
    from .functions import Function

    return Function('ifNull', x, alt, alias=alias)


@register_function(
    name='is_null',
    clickhouse_name='isNull',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['isNull'],
    doc='Check if NULL. Maps to isNull(x).',
)
def _build_is_null(expr, alias=None):
    from .functions import Function

    return Function('isNull', expr, alias=alias)


@register_function(
    name='is_not_null',
    clickhouse_name='isNotNull',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['isNotNull'],
    doc='Check if not NULL. Maps to isNotNull(x).',
)
def _build_is_not_null(expr, alias=None):
    from .functions import Function

    return Function('isNotNull', expr, alias=alias)


@register_function(
    name='assume_not_null',
    clickhouse_name='assumeNotNull',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['assumeNotNull'],
    doc='Assume not NULL. Maps to assumeNotNull(x).',
)
def _build_assume_not_null(expr, alias=None):
    from .functions import Function

    return Function('assumeNotNull', expr, alias=alias)


@register_function(
    name='to_nullable',
    clickhouse_name='toNullable',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['toNullable'],
    doc='Convert to Nullable. Maps to toNullable(x).',
)
def _build_to_nullable(expr, alias=None):
    from .functions import Function

    return Function('toNullable', expr, alias=alias)


@register_function(
    name='greatest',
    clickhouse_name='greatest',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['greatest'],
    doc='Return greatest value. Maps to greatest(a, b, ...).',
)
def _build_greatest(*args, alias=None):
    from .functions import Function

    return Function('greatest', *args, alias=alias)


@register_function(
    name='least',
    clickhouse_name='least',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['least'],
    doc='Return least value. Maps to least(a, b, ...).',
)
def _build_least(*args, alias=None):
    from .functions import Function

    return Function('least', *args, alias=alias)


# =============================================================================
# BIT FUNCTIONS
# =============================================================================


@register_function(
    name='bit_and',
    clickhouse_name='bitAnd',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['bitAnd'],
    doc='Bitwise AND. Maps to bitAnd(a, b).',
)
def _build_bit_and(a, b, alias=None):
    from .functions import Function

    return Function('bitAnd', a, b, alias=alias)


@register_function(
    name='bit_or',
    clickhouse_name='bitOr',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['bitOr'],
    doc='Bitwise OR. Maps to bitOr(a, b).',
)
def _build_bit_or(a, b, alias=None):
    from .functions import Function

    return Function('bitOr', a, b, alias=alias)


@register_function(
    name='bit_xor',
    clickhouse_name='bitXor',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['bitXor'],
    doc='Bitwise XOR. Maps to bitXor(a, b).',
)
def _build_bit_xor(a, b, alias=None):
    from .functions import Function

    return Function('bitXor', a, b, alias=alias)


@register_function(
    name='bit_not',
    clickhouse_name='bitNot',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['bitNot'],
    doc='Bitwise NOT. Maps to bitNot(a).',
)
def _build_bit_not(a, alias=None):
    from .functions import Function

    return Function('bitNot', a, alias=alias)


@register_function(
    name='bit_shift_left',
    clickhouse_name='bitShiftLeft',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['bitShiftLeft'],
    doc='Bit shift left. Maps to bitShiftLeft(a, b).',
)
def _build_bit_shift_left(a, b, alias=None):
    from .functions import Function

    return Function('bitShiftLeft', a, b, alias=alias)


@register_function(
    name='bit_shift_right',
    clickhouse_name='bitShiftRight',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['bitShiftRight'],
    doc='Bit shift right. Maps to bitShiftRight(a, b).',
)
def _build_bit_shift_right(a, b, alias=None):
    from .functions import Function

    return Function('bitShiftRight', a, b, alias=alias)


@register_function(
    name='bit_rotate_left',
    clickhouse_name='bitRotateLeft',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['bitRotateLeft'],
    doc='Bit rotate left. Maps to bitRotateLeft(a, b).',
)
def _build_bit_rotate_left(a, b, alias=None):
    from .functions import Function

    return Function('bitRotateLeft', a, b, alias=alias)


@register_function(
    name='bit_rotate_right',
    clickhouse_name='bitRotateRight',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['bitRotateRight'],
    doc='Bit rotate right. Maps to bitRotateRight(a, b).',
)
def _build_bit_rotate_right(a, b, alias=None):
    from .functions import Function

    return Function('bitRotateRight', a, b, alias=alias)


@register_function(
    name='bit_test',
    clickhouse_name='bitTest',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['bitTest'],
    doc='Test bit at position. Maps to bitTest(a, pos).',
)
def _build_bit_test(a, pos, alias=None):
    from .functions import Function

    return Function('bitTest', a, pos, alias=alias)


@register_function(
    name='bit_count',
    clickhouse_name='bitCount',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['bitCount'],
    doc='Count set bits (popcount). Maps to bitCount(a).',
)
def _build_bit_count(a, alias=None):
    from .functions import Function

    return Function('bitCount', a, alias=alias)


@register_function(
    name='bit_hamming_distance',
    clickhouse_name='bitHammingDistance',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['bitHammingDistance'],
    doc='Hamming distance between bits. Maps to bitHammingDistance(a, b).',
)
def _build_bit_hamming_distance(a, b, alias=None):
    from .functions import Function

    return Function('bitHammingDistance', a, b, alias=alias)


# =============================================================================
# RANDOM FUNCTIONS
# =============================================================================


@register_function(
    name='rand',
    clickhouse_name='rand',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['rand', 'random'],
    doc='Random UInt32. Maps to rand().',
)
def _build_rand(alias=None):
    from .functions import Function

    return Function('rand', alias=alias)


@register_function(
    name='rand64',
    clickhouse_name='rand64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['rand64'],
    doc='Random UInt64. Maps to rand64().',
)
def _build_rand64(alias=None):
    from .functions import Function

    return Function('rand64', alias=alias)


@register_function(
    name='rand_constant',
    clickhouse_name='randConstant',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['randConstant'],
    doc='Random constant per block. Maps to randConstant().',
)
def _build_rand_constant(alias=None):
    from .functions import Function

    return Function('randConstant', alias=alias)


@register_function(
    name='rand_uniform',
    clickhouse_name='randUniform',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['randUniform'],
    doc='Random uniform distribution. Maps to randUniform(min, max).',
)
def _build_rand_uniform(min_val, max_val, alias=None):
    from .functions import Function

    return Function('randUniform', min_val, max_val, alias=alias)


@register_function(
    name='rand_normal',
    clickhouse_name='randNormal',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['randNormal'],
    doc='Random normal distribution. Maps to randNormal(mean, stddev).',
)
def _build_rand_normal(mean, stddev, alias=None):
    from .functions import Function

    return Function('randNormal', mean, stddev, alias=alias)


@register_function(
    name='rand_log_normal',
    clickhouse_name='randLogNormal',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['randLogNormal'],
    doc='Random log-normal distribution. Maps to randLogNormal(mean, stddev).',
)
def _build_rand_log_normal(mean, stddev, alias=None):
    from .functions import Function

    return Function('randLogNormal', mean, stddev, alias=alias)


@register_function(
    name='rand_exponential',
    clickhouse_name='randExponential',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['randExponential'],
    doc='Random exponential distribution. Maps to randExponential(lambda).',
)
def _build_rand_exponential(lam, alias=None):
    from .functions import Function

    return Function('randExponential', lam, alias=alias)


@register_function(
    name='rand_binomial',
    clickhouse_name='randBinomial',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['randBinomial'],
    doc='Random binomial distribution. Maps to randBinomial(n, p).',
)
def _build_rand_binomial(n, p, alias=None):
    from .functions import Function

    return Function('randBinomial', n, p, alias=alias)


@register_function(
    name='rand_poisson',
    clickhouse_name='randPoisson',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['randPoisson'],
    doc='Random Poisson distribution. Maps to randPoisson(lambda).',
)
def _build_rand_poisson(lam, alias=None):
    from .functions import Function

    return Function('randPoisson', lam, alias=alias)


@register_function(
    name='rand_bernoulli',
    clickhouse_name='randBernoulli',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['randBernoulli'],
    doc='Random Bernoulli distribution. Maps to randBernoulli(p).',
)
def _build_rand_bernoulli(p, alias=None):
    from .functions import Function

    return Function('randBernoulli', p, alias=alias)


# =============================================================================
# MAP FUNCTIONS
# =============================================================================


@register_function(
    name='map_func',
    clickhouse_name='map',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['map', 'createMap'],
    doc='Create map from keys and values. Maps to map(k1, v1, k2, v2, ...).',
)
def _build_map_func(*args, alias=None):
    from .functions import Function

    return Function('map', *args, alias=alias)


@register_function(
    name='map_keys',
    clickhouse_name='mapKeys',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['mapKeys'],
    doc='Get keys from map. Maps to mapKeys(map).',
)
def _build_map_keys(m, alias=None):
    from .functions import Function

    return Function('mapKeys', m, alias=alias)


@register_function(
    name='map_values',
    clickhouse_name='mapValues',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['mapValues'],
    doc='Get values from map. Maps to mapValues(map).',
)
def _build_map_values(m, alias=None):
    from .functions import Function

    return Function('mapValues', m, alias=alias)


@register_function(
    name='map_contains_key',
    clickhouse_name='mapContainsKey',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['mapContainsKey', 'mapContains'],
    doc='Check if map contains key. Maps to mapContainsKey(map, key).',
)
def _build_map_contains_key(m, key, alias=None):
    from .functions import Function

    return Function('mapContainsKey', m, key, alias=alias)


@register_function(
    name='map_concat',
    clickhouse_name='mapConcat',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['mapConcat'],
    doc='Concatenate maps. Maps to mapConcat(map1, map2, ...).',
)
def _build_map_concat(*maps, alias=None):
    from .functions import Function

    return Function('mapConcat', *maps, alias=alias)


@register_function(
    name='map_exists',
    clickhouse_name='mapExists',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['mapExists'],
    doc='Check if any key-value pair matches. Maps to mapExists(lambda, map).',
)
def _build_map_exists(lambda_expr, m, alias=None):
    from .functions import Function

    return Function('mapExists', lambda_expr, m, alias=alias)


@register_function(
    name='map_all',
    clickhouse_name='mapAll',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['mapAll'],
    doc='Check if all key-value pairs match. Maps to mapAll(lambda, map).',
)
def _build_map_all(lambda_expr, m, alias=None):
    from .functions import Function

    return Function('mapAll', lambda_expr, m, alias=alias)


@register_function(
    name='map_filter',
    clickhouse_name='mapFilter',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['mapFilter'],
    doc='Filter map by condition. Maps to mapFilter(lambda, map).',
)
def _build_map_filter(lambda_expr, m, alias=None):
    from .functions import Function

    return Function('mapFilter', lambda_expr, m, alias=alias)


@register_function(
    name='map_apply',
    clickhouse_name='mapApply',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['mapApply'],
    doc='Apply function to map. Maps to mapApply(lambda, map).',
)
def _build_map_apply(lambda_expr, m, alias=None):
    from .functions import Function

    return Function('mapApply', lambda_expr, m, alias=alias)


@register_function(
    name='map_add',
    clickhouse_name='mapAdd',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['mapAdd'],
    doc='Add values for same keys. Maps to mapAdd(map1, map2).',
)
def _build_map_add(m1, m2, alias=None):
    from .functions import Function

    return Function('mapAdd', m1, m2, alias=alias)


@register_function(
    name='map_subtract',
    clickhouse_name='mapSubtract',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['mapSubtract'],
    doc='Subtract values for same keys. Maps to mapSubtract(map1, map2).',
)
def _build_map_subtract(m1, m2, alias=None):
    from .functions import Function

    return Function('mapSubtract', m1, m2, alias=alias)


# =============================================================================
# TUPLE FUNCTIONS
# =============================================================================


@register_function(
    name='tuple_func',
    clickhouse_name='tuple',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['tuple', 'makeTuple'],
    doc='Create tuple. Maps to tuple(v1, v2, ...).',
)
def _build_tuple_func(*args, alias=None):
    from .functions import Function

    return Function('tuple', *args, alias=alias)


@register_function(
    name='tuple_element',
    clickhouse_name='tupleElement',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['tupleElement'],
    doc='Get element from tuple. Maps to tupleElement(tuple, n).',
)
def _build_tuple_element(t, n, alias=None):
    from .functions import Function

    return Function('tupleElement', t, n, alias=alias)


@register_function(
    name='tuple_plus',
    clickhouse_name='tuplePlus',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['tuplePlus'],
    doc='Add tuples element-wise. Maps to tuplePlus(t1, t2).',
)
def _build_tuple_plus(t1, t2, alias=None):
    from .functions import Function

    return Function('tuplePlus', t1, t2, alias=alias)


@register_function(
    name='tuple_minus',
    clickhouse_name='tupleMinus',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['tupleMinus'],
    doc='Subtract tuples element-wise. Maps to tupleMinus(t1, t2).',
)
def _build_tuple_minus(t1, t2, alias=None):
    from .functions import Function

    return Function('tupleMinus', t1, t2, alias=alias)


@register_function(
    name='tuple_multiply',
    clickhouse_name='tupleMultiply',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['tupleMultiply'],
    doc='Multiply tuples element-wise. Maps to tupleMultiply(t1, t2).',
)
def _build_tuple_multiply(t1, t2, alias=None):
    from .functions import Function

    return Function('tupleMultiply', t1, t2, alias=alias)


@register_function(
    name='tuple_divide',
    clickhouse_name='tupleDivide',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['tupleDivide'],
    doc='Divide tuples element-wise. Maps to tupleDivide(t1, t2).',
)
def _build_tuple_divide(t1, t2, alias=None):
    from .functions import Function

    return Function('tupleDivide', t1, t2, alias=alias)


# =============================================================================
# MORE DATETIME FUNCTIONS
# =============================================================================


@register_function(
    name='add_seconds',
    clickhouse_name='addSeconds',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['addSeconds'],
    doc='Add seconds to datetime. Maps to addSeconds(dt, n).',
)
def _build_add_seconds(dt, n, alias=None):
    from .functions import Function

    return Function('addSeconds', dt, n, alias=alias)


@register_function(
    name='add_minutes',
    clickhouse_name='addMinutes',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['addMinutes'],
    doc='Add minutes to datetime. Maps to addMinutes(dt, n).',
)
def _build_add_minutes(dt, n, alias=None):
    from .functions import Function

    return Function('addMinutes', dt, n, alias=alias)


@register_function(
    name='add_hours',
    clickhouse_name='addHours',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['addHours'],
    doc='Add hours to datetime. Maps to addHours(dt, n).',
)
def _build_add_hours(dt, n, alias=None):
    from .functions import Function

    return Function('addHours', dt, n, alias=alias)


@register_function(
    name='add_days',
    clickhouse_name='addDays',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['addDays'],
    doc='Add days to datetime. Maps to addDays(dt, n).',
)
def _build_add_days(dt, n, alias=None):
    from .functions import Function

    return Function('addDays', dt, n, alias=alias)


@register_function(
    name='add_weeks',
    clickhouse_name='addWeeks',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['addWeeks'],
    doc='Add weeks to datetime. Maps to addWeeks(dt, n).',
)
def _build_add_weeks(dt, n, alias=None):
    from .functions import Function

    return Function('addWeeks', dt, n, alias=alias)


@register_function(
    name='add_months',
    clickhouse_name='addMonths',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['addMonths'],
    doc='Add months to datetime. Maps to addMonths(dt, n).',
)
def _build_add_months(dt, n, alias=None):
    from .functions import Function

    return Function('addMonths', dt, n, alias=alias)


@register_function(
    name='add_quarters',
    clickhouse_name='addQuarters',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['addQuarters'],
    doc='Add quarters to datetime. Maps to addQuarters(dt, n).',
)
def _build_add_quarters(dt, n, alias=None):
    from .functions import Function

    return Function('addQuarters', dt, n, alias=alias)


@register_function(
    name='add_years',
    clickhouse_name='addYears',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['addYears'],
    doc='Add years to datetime. Maps to addYears(dt, n).',
)
def _build_add_years(dt, n, alias=None):
    from .functions import Function

    return Function('addYears', dt, n, alias=alias)


@register_function(
    name='subtract_seconds',
    clickhouse_name='subtractSeconds',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['subtractSeconds'],
    doc='Subtract seconds. Maps to subtractSeconds(dt, n).',
)
def _build_subtract_seconds(dt, n, alias=None):
    from .functions import Function

    return Function('subtractSeconds', dt, n, alias=alias)


@register_function(
    name='subtract_minutes',
    clickhouse_name='subtractMinutes',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['subtractMinutes'],
    doc='Subtract minutes. Maps to subtractMinutes(dt, n).',
)
def _build_subtract_minutes(dt, n, alias=None):
    from .functions import Function

    return Function('subtractMinutes', dt, n, alias=alias)


@register_function(
    name='subtract_hours',
    clickhouse_name='subtractHours',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['subtractHours'],
    doc='Subtract hours. Maps to subtractHours(dt, n).',
)
def _build_subtract_hours(dt, n, alias=None):
    from .functions import Function

    return Function('subtractHours', dt, n, alias=alias)


@register_function(
    name='subtract_days',
    clickhouse_name='subtractDays',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['subtractDays'],
    doc='Subtract days. Maps to subtractDays(dt, n).',
)
def _build_subtract_days(dt, n, alias=None):
    from .functions import Function

    return Function('subtractDays', dt, n, alias=alias)


@register_function(
    name='subtract_weeks',
    clickhouse_name='subtractWeeks',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['subtractWeeks'],
    doc='Subtract weeks. Maps to subtractWeeks(dt, n).',
)
def _build_subtract_weeks(dt, n, alias=None):
    from .functions import Function

    return Function('subtractWeeks', dt, n, alias=alias)


@register_function(
    name='subtract_months',
    clickhouse_name='subtractMonths',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['subtractMonths'],
    doc='Subtract months. Maps to subtractMonths(dt, n).',
)
def _build_subtract_months(dt, n, alias=None):
    from .functions import Function

    return Function('subtractMonths', dt, n, alias=alias)


@register_function(
    name='subtract_years',
    clickhouse_name='subtractYears',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['subtractYears'],
    doc='Subtract years. Maps to subtractYears(dt, n).',
)
def _build_subtract_years(dt, n, alias=None):
    from .functions import Function

    return Function('subtractYears', dt, n, alias=alias)


@register_function(
    name='date_diff',
    clickhouse_name='dateDiff',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['dateDiff', 'date_diff'],
    doc='Difference between dates. Maps to dateDiff(unit, start, end).',
)
def _build_date_diff(unit: str, start, end, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('dateDiff', Literal(unit), start, end, alias=alias)


@register_function(
    name='date_trunc',
    clickhouse_name='dateTrunc',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['dateTrunc', 'date_trunc'],
    doc='Truncate date to unit. Maps to dateTrunc(unit, dt).',
)
def _build_date_trunc(unit: str, dt, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('dateTrunc', Literal(unit), dt, alias=alias)


@register_function(
    name='date_name',
    clickhouse_name='dateName',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['dateName'],
    doc='Get date part name. Maps to dateName(part, dt).',
)
def _build_date_name(part: str, dt, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('dateName', Literal(part), dt, alias=alias)


@register_function(
    name='month_name',
    clickhouse_name='monthName',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['monthName'],
    doc='Get month name. Maps to monthName(dt).',
)
def _build_month_name(dt, alias=None):
    from .functions import Function

    return Function('monthName', dt, alias=alias)


@register_function(
    name='age_func',
    clickhouse_name='age',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['age'],
    doc='Age between dates. Maps to age(unit, start, end).',
)
def _build_age_func(unit: str, start, end, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('age', Literal(unit), start, end, alias=alias)


@register_function(
    name='make_date',
    clickhouse_name='makeDate',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['makeDate'],
    doc='Create date from parts. Maps to makeDate(year, month, day).',
)
def _build_make_date(year, month, day, alias=None):
    from .functions import Function

    return Function('makeDate', year, month, day, alias=alias)


@register_function(
    name='make_datetime',
    clickhouse_name='makeDateTime',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['makeDateTime'],
    doc='Create datetime from parts. Maps to makeDateTime(year, month, day, hour, minute, second).',
)
def _build_make_datetime(year, month, day, hour=0, minute=0, second=0, alias=None):
    from .functions import Function

    return Function('makeDateTime', year, month, day, hour, minute, second, alias=alias)


@register_function(
    name='from_unix_timestamp',
    clickhouse_name='fromUnixTimestamp',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['fromUnixTimestamp'],
    doc='Convert unix timestamp to datetime. Maps to fromUnixTimestamp(ts).',
)
def _build_from_unix_timestamp(ts, alias=None):
    from .functions import Function

    return Function('fromUnixTimestamp', ts, alias=alias)


@register_function(
    name='to_unix_timestamp',
    clickhouse_name='toUnixTimestamp',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toUnixTimestamp'],
    doc='Convert datetime to unix timestamp. Maps to toUnixTimestamp(dt).',
)
def _build_to_unix_timestamp(dt, alias=None):
    from .functions import Function

    return Function('toUnixTimestamp', dt, alias=alias)


@register_function(
    name='parse_datetime_best_effort',
    clickhouse_name='parseDateTimeBestEffort',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['parseDateTimeBestEffort'],
    doc='Parse datetime with best effort. Maps to parseDateTimeBestEffort(s).',
)
def _build_parse_datetime_best_effort(s, alias=None):
    from .functions import Function

    return Function('parseDateTimeBestEffort', s, alias=alias)


@register_function(
    name='parse_datetime_best_effort_or_null',
    clickhouse_name='parseDateTimeBestEffortOrNull',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['parseDateTimeBestEffortOrNull'],
    doc='Parse datetime or NULL. Maps to parseDateTimeBestEffortOrNull(s).',
)
def _build_parse_datetime_best_effort_or_null(s, alias=None):
    from .functions import Function

    return Function('parseDateTimeBestEffortOrNull', s, alias=alias)


@register_function(
    name='to_last_day_of_month',
    clickhouse_name='toLastDayOfMonth',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toLastDayOfMonth'],
    doc='Get last day of month. Maps to toLastDayOfMonth(dt).',
)
def _build_to_last_day_of_month(dt, alias=None):
    from .functions import Function

    return Function('toLastDayOfMonth', dt, alias=alias)


@register_function(
    name='to_monday',
    clickhouse_name='toMonday',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toMonday'],
    doc='Get Monday of week. Maps to toMonday(dt).',
)
def _build_to_monday(dt, alias=None):
    from .functions import Function

    return Function('toMonday', dt, alias=alias)


# =============================================================================
# STRING DISTANCE FUNCTIONS
# =============================================================================


@register_function(
    name='levenshtein_distance',
    clickhouse_name='editDistance',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['editDistance', 'levenshteinDistance'],
    doc='Levenshtein distance. Maps to editDistance(s1, s2).',
)
def _build_levenshtein_distance(s1, s2, alias=None):
    from .functions import Function

    return Function('editDistance', s1, s2, alias=alias)


@register_function(
    name='damerau_levenshtein_distance',
    clickhouse_name='damerauLevenshteinDistance',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['damerauLevenshteinDistance'],
    doc='Damerau-Levenshtein distance. Maps to damerauLevenshteinDistance(s1, s2).',
)
def _build_damerau_levenshtein_distance(s1, s2, alias=None):
    from .functions import Function

    return Function('damerauLevenshteinDistance', s1, s2, alias=alias)


@register_function(
    name='jaro_similarity',
    clickhouse_name='jaroSimilarity',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['jaroSimilarity'],
    doc='Jaro similarity. Maps to jaroSimilarity(s1, s2).',
)
def _build_jaro_similarity(s1, s2, alias=None):
    from .functions import Function

    return Function('jaroSimilarity', s1, s2, alias=alias)


@register_function(
    name='jaro_winkler_similarity',
    clickhouse_name='jaroWinklerSimilarity',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['jaroWinklerSimilarity'],
    doc='Jaro-Winkler similarity. Maps to jaroWinklerSimilarity(s1, s2).',
)
def _build_jaro_winkler_similarity(s1, s2, alias=None):
    from .functions import Function

    return Function('jaroWinklerSimilarity', s1, s2, alias=alias)


@register_function(
    name='ngram_distance',
    clickhouse_name='ngramDistance',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['ngramDistance'],
    doc='N-gram distance. Maps to ngramDistance(s1, s2).',
)
def _build_ngram_distance(s1, s2, alias=None):
    from .functions import Function

    return Function('ngramDistance', s1, s2, alias=alias)


@register_function(
    name='ngram_search',
    clickhouse_name='ngramSearch',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['ngramSearch'],
    doc='N-gram search score. Maps to ngramSearch(haystack, needle).',
)
def _build_ngram_search(haystack, needle, alias=None):
    from .functions import Function

    return Function('ngramSearch', haystack, needle, alias=alias)


@register_function(
    name='cosine_distance',
    clickhouse_name='cosineDistance',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['cosineDistance'],
    doc='Cosine distance. Maps to cosineDistance(v1, v2).',
)
def _build_cosine_distance(v1, v2, alias=None):
    from .functions import Function

    return Function('cosineDistance', v1, v2, alias=alias)


# =============================================================================
# MORE STRING FUNCTIONS
# =============================================================================


@register_function(
    name='char_func',
    clickhouse_name='char',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['char'],
    doc='Character from code. Maps to char(code).',
)
def _build_char_func(code, alias=None):
    from .functions import Function

    return Function('char', code, alias=alias)


@register_function(
    name='ascii_func',
    clickhouse_name='ASCII',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['ASCII', 'ascii'],
    doc='ASCII code of first character. Maps to ASCII(s).',
)
def _build_ascii_func(s, alias=None):
    from .functions import Function

    return Function('ASCII', s, alias=alias)


@register_function(
    name='left_pad',
    clickhouse_name='leftPad',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['leftPad', 'lpad'],
    doc='Left pad string. Maps to leftPad(s, len, pad).',
)
def _build_left_pad(s, length: int, pad: str = ' ', alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('leftPad', s, Literal(length), Literal(pad), alias=alias)


@register_function(
    name='right_pad',
    clickhouse_name='rightPad',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['rightPad', 'rpad'],
    doc='Right pad string. Maps to rightPad(s, len, pad).',
)
def _build_right_pad(s, length: int, pad: str = ' ', alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('rightPad', s, Literal(length), Literal(pad), alias=alias)


@register_function(
    name='format_func',
    clickhouse_name='format',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['format'],
    doc='Format string with arguments. Maps to format(template, arg1, ...).',
)
def _build_format_func(template, *args, alias=None):
    from .functions import Function

    return Function('format', template, *args, alias=alias)


@register_function(
    name='concat_func',
    clickhouse_name='concat',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['concat'],
    doc='Concatenate strings. Maps to concat(s1, s2, ...).',
)
def _build_concat_func(*args, alias=None):
    from .functions import Function

    return Function('concat', *args, alias=alias)


@register_function(
    name='concat_with_separator',
    clickhouse_name='concatWithSeparator',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['concatWithSeparator'],
    doc='Concatenate with separator. Maps to concatWithSeparator(sep, s1, s2, ...).',
)
def _build_concat_with_separator(sep, *args, alias=None):
    from .functions import Function

    return Function('concatWithSeparator', sep, *args, alias=alias)


@register_function(
    name='tokens',
    clickhouse_name='tokens',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['tokens'],
    doc='Split into tokens. Maps to tokens(s).',
)
def _build_tokens(s, alias=None):
    from .functions import Function

    return Function('tokens', s, alias=alias)


@register_function(
    name='ngrams',
    clickhouse_name='ngrams',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['ngrams'],
    doc='Generate n-grams. Maps to ngrams(s, n).',
)
def _build_ngrams(s, n: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('ngrams', s, Literal(n), alias=alias)


@register_function(
    name='extract_text_from_html',
    clickhouse_name='extractTextFromHTML',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['extractTextFromHTML'],
    doc='Extract text from HTML. Maps to extractTextFromHTML(html).',
)
def _build_extract_text_from_html(html, alias=None):
    from .functions import Function

    return Function('extractTextFromHTML', html, alias=alias)


@register_function(
    name='regexp_extract',
    clickhouse_name='regexpExtract',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['regexpExtract', 'regexp_extract'],
    doc='Extract using regex. Maps to regexpExtract(s, pattern, index).',
)
def _build_regexp_extract(s, pattern: str, index: int = 0, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('regexpExtract', s, Literal(pattern), Literal(index), alias=alias)


@register_function(
    name='regexp_replace',
    clickhouse_name='replaceRegexpAll',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['replaceRegexpAll', 'regexp_replace'],
    doc='Replace using regex. Maps to replaceRegexpAll(s, pattern, replacement).',
)
def _build_regexp_replace(s, pattern: str, replacement: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('replaceRegexpAll', s, Literal(pattern), Literal(replacement), alias=alias)


@register_function(
    name='multi_match_any',
    clickhouse_name='multiMatchAny',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['multiMatchAny'],
    doc='Check if any pattern matches. Maps to multiMatchAny(s, [patterns]).',
)
def _build_multi_match_any(s, patterns, alias=None):
    from .functions import Function

    return Function('multiMatchAny', s, patterns, alias=alias)


@register_function(
    name='multi_match_any_index',
    clickhouse_name='multiMatchAnyIndex',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['multiMatchAnyIndex'],
    doc='Get index of first matching pattern. Maps to multiMatchAnyIndex(s, [patterns]).',
)
def _build_multi_match_any_index(s, patterns, alias=None):
    from .functions import Function

    return Function('multiMatchAnyIndex', s, patterns, alias=alias)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


@register_function(
    name='to_type_name',
    clickhouse_name='toTypeName',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toTypeName'],
    doc='Get type name of value. Maps to toTypeName(x).',
)
def _build_to_type_name(expr, alias=None):
    from .functions import Function

    return Function('toTypeName', expr, alias=alias)


@register_function(
    name='materialize',
    clickhouse_name='materialize',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['materialize'],
    doc='Force materialization. Maps to materialize(x).',
)
def _build_execute(expr, alias=None):
    from .functions import Function

    return Function('materialize', expr, alias=alias)


@register_function(
    name='ignore_func',
    clickhouse_name='ignore',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['ignore'],
    doc='Ignore values. Maps to ignore(...).',
)
def _build_ignore_func(*args, alias=None):
    from .functions import Function

    return Function('ignore', *args, alias=alias)


@register_function(
    name='sleep_func',
    clickhouse_name='sleep',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['sleep'],
    doc='Sleep for seconds. Maps to sleep(seconds).',
)
def _build_sleep_func(seconds, alias=None):
    from .functions import Function

    return Function('sleep', seconds, alias=alias)


@register_function(
    name='throw_if',
    clickhouse_name='throwIf',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['throwIf'],
    doc='Throw exception if condition. Maps to throwIf(cond, message).',
)
def _build_throw_if(cond, message: str = None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if message:
        return Function('throwIf', cond, Literal(message), alias=alias)
    return Function('throwIf', cond, alias=alias)


@register_function(
    name='format_readable_size',
    clickhouse_name='formatReadableSize',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['formatReadableSize'],
    doc='Format bytes as readable. Maps to formatReadableSize(bytes).',
)
def _build_format_readable_size(bytes_val, alias=None):
    from .functions import Function

    return Function('formatReadableSize', bytes_val, alias=alias)


@register_function(
    name='format_readable_quantity',
    clickhouse_name='formatReadableQuantity',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['formatReadableQuantity'],
    doc='Format number with suffix. Maps to formatReadableQuantity(x).',
)
def _build_format_readable_quantity(x, alias=None):
    from .functions import Function

    return Function('formatReadableQuantity', x, alias=alias)


@register_function(
    name='format_readable_time_delta',
    clickhouse_name='formatReadableTimeDelta',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['formatReadableTimeDelta'],
    doc='Format seconds as readable time. Maps to formatReadableTimeDelta(seconds).',
)
def _build_format_readable_time_delta(seconds, alias=None):
    from .functions import Function

    return Function('formatReadableTimeDelta', seconds, alias=alias)


@register_function(
    name='generate_uuid_v4',
    clickhouse_name='generateUUIDv4',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.UUID,
    aliases=['generateUUIDv4'],
    doc='Generate UUIDv4. Maps to generateUUIDv4().',
)
def _build_generate_uuid_v4(alias=None):
    from .functions import Function

    return Function('generateUUIDv4', alias=alias)


@register_function(
    name='generate_uuid_v7',
    clickhouse_name='generateUUIDv7',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.UUID,
    aliases=['generateUUIDv7'],
    doc='Generate UUIDv7. Maps to generateUUIDv7().',
)
def _build_generate_uuid_v7(alias=None):
    from .functions import Function

    return Function('generateUUIDv7', alias=alias)


@register_function(
    name='snowflake_to_datetime',
    clickhouse_name='snowflakeToDateTime',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['snowflakeToDateTime'],
    doc='Convert Snowflake ID to datetime. Maps to snowflakeToDateTime(id).',
)
def _build_snowflake_to_datetime(id_val, alias=None):
    from .functions import Function

    return Function('snowflakeToDateTime', id_val, alias=alias)


@register_function(
    name='datetime_to_snowflake',
    clickhouse_name='dateTimeToSnowflake',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['dateTimeToSnowflake'],
    doc='Convert datetime to Snowflake ID. Maps to dateTimeToSnowflake(dt).',
)
def _build_datetime_to_snowflake(dt, alias=None):
    from .functions import Function

    return Function('dateTimeToSnowflake', dt, alias=alias)


# =============================================================================
# VECTOR DISTANCE FUNCTIONS
# =============================================================================


@register_function(
    name='l1_norm',
    clickhouse_name='L1Norm',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['L1Norm'],
    doc='L1 norm of vector. Maps to L1Norm(vec).',
)
def _build_l1_norm(vec, alias=None):
    from .functions import Function

    return Function('L1Norm', vec, alias=alias)


@register_function(
    name='l2_norm',
    clickhouse_name='L2Norm',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['L2Norm'],
    doc='L2 norm of vector. Maps to L2Norm(vec).',
)
def _build_l2_norm(vec, alias=None):
    from .functions import Function

    return Function('L2Norm', vec, alias=alias)


@register_function(
    name='linf_norm',
    clickhouse_name='LinfNorm',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['LinfNorm'],
    doc='Linf norm of vector. Maps to LinfNorm(vec).',
)
def _build_linf_norm(vec, alias=None):
    from .functions import Function

    return Function('LinfNorm', vec, alias=alias)


@register_function(
    name='linf_distance',
    clickhouse_name='LinfDistance',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['LinfDistance'],
    doc='Linf distance. Maps to LinfDistance(v1, v2).',
)
def _build_linf_distance(v1, v2, alias=None):
    from .functions import Function

    return Function('LinfDistance', v1, v2, alias=alias)


@register_function(
    name='lp_norm',
    clickhouse_name='LpNorm',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['LpNorm'],
    doc='Lp norm of vector. Maps to LpNorm(vec, p).',
)
def _build_lp_norm(vec, p, alias=None):
    from .functions import Function

    return Function('LpNorm', vec, p, alias=alias)


@register_function(
    name='lp_distance',
    clickhouse_name='LpDistance',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['LpDistance'],
    doc='Lp distance. Maps to LpDistance(v1, v2, p).',
)
def _build_lp_distance(v1, v2, p, alias=None):
    from .functions import Function

    return Function('LpDistance', v1, v2, p, alias=alias)


@register_function(
    name='l1_normalize',
    clickhouse_name='L1Normalize',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['L1Normalize'],
    doc='L1 normalize vector. Maps to L1Normalize(vec).',
)
def _build_l1_normalize(vec, alias=None):
    from .functions import Function

    return Function('L1Normalize', vec, alias=alias)


@register_function(
    name='l2_normalize',
    clickhouse_name='L2Normalize',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.GEO,
    aliases=['L2Normalize'],
    doc='L2 normalize vector. Maps to L2Normalize(vec).',
)
def _build_l2_normalize(vec, alias=None):
    from .functions import Function

    return Function('L2Normalize', vec, alias=alias)


# =============================================================================
# MORE AGGREGATE FUNCTIONS (Extended)
# =============================================================================


@register_function(
    name='avg_weighted',
    clickhouse_name='avgWeighted',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['avgWeighted'],
    doc='Weighted average. Maps to avgWeighted(x, weight).',
)
def _build_avg_weighted(x, weight, alias=None):
    from .functions import Function

    return Function('avgWeighted', x, weight, alias=alias)


@register_function(
    name='sum_count',
    clickhouse_name='sumCount',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['sumCount'],
    doc='Sum and count. Maps to sumCount(x).',
)
def _build_sum_count(x, alias=None):
    from .functions import Function

    return Function('sumCount', x, alias=alias)


@register_function(
    name='sum_kahan',
    clickhouse_name='sumKahan',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['sumKahan'],
    doc='Kahan summation. Maps to sumKahan(x).',
)
def _build_sum_kahan(x, alias=None):
    from .functions import Function

    return Function('sumKahan', x, alias=alias)


@register_function(
    name='count_equal',
    clickhouse_name='countEqual',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['countEqual'],
    doc='Count equal elements in array. Maps to countEqual(arr, x).',
)
def _build_count_equal(arr, x, alias=None):
    from .functions import Function

    return Function('countEqual', arr, x, alias=alias)


@register_function(
    name='group_array_moving_avg',
    clickhouse_name='groupArrayMovingAvg',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['groupArrayMovingAvg'],
    doc='Moving average in array. Maps to groupArrayMovingAvg(window)(x).',
)
def _build_group_array_moving_avg(x, window: int = None, alias=None):
    from .functions import Function

    if window:
        return Function(f'groupArrayMovingAvg({window})', x, alias=alias)
    return Function('groupArrayMovingAvg', x, alias=alias)


@register_function(
    name='group_array_moving_sum',
    clickhouse_name='groupArrayMovingSum',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['groupArrayMovingSum'],
    doc='Moving sum in array. Maps to groupArrayMovingSum(window)(x).',
)
def _build_group_array_moving_sum(x, window: int = None, alias=None):
    from .functions import Function

    if window:
        return Function(f'groupArrayMovingSum({window})', x, alias=alias)
    return Function('groupArrayMovingSum', x, alias=alias)


@register_function(
    name='group_array_sample',
    clickhouse_name='groupArraySample',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['groupArraySample'],
    doc='Random sample from group. Maps to groupArraySample(n)(x).',
)
def _build_group_array_sample(x, n: int, alias=None):
    from .functions import Function

    return Function(f'groupArraySample({n})', x, alias=alias)


@register_function(
    name='group_array_sorted',
    clickhouse_name='groupArraySorted',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['groupArraySorted'],
    doc='Sorted array from group. Maps to groupArraySorted(n)(x).',
)
def _build_group_array_sorted(x, n: int = None, alias=None):
    from .functions import Function

    if n:
        return Function(f'groupArraySorted({n})', x, alias=alias)
    return Function('groupArraySorted', x, alias=alias)


@register_function(
    name='group_bit_or',
    clickhouse_name='groupBitOr',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['groupBitOr'],
    doc='Bitwise OR of group. Maps to groupBitOr(x).',
)
def _build_group_bit_or(x, alias=None):
    from .functions import Function

    return Function('groupBitOr', x, alias=alias)


@register_function(
    name='group_bit_and',
    clickhouse_name='groupBitAnd',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['groupBitAnd'],
    doc='Bitwise AND of group. Maps to groupBitAnd(x).',
)
def _build_group_bit_and(x, alias=None):
    from .functions import Function

    return Function('groupBitAnd', x, alias=alias)


@register_function(
    name='group_bit_xor',
    clickhouse_name='groupBitXor',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['groupBitXor'],
    doc='Bitwise XOR of group. Maps to groupBitXor(x).',
)
def _build_group_bit_xor(x, alias=None):
    from .functions import Function

    return Function('groupBitXor', x, alias=alias)


@register_function(
    name='retention',
    clickhouse_name='retention',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['retention'],
    doc='Retention analysis. Maps to retention(cond1, cond2, ...).',
)
def _build_retention(*conds, alias=None):
    from .functions import Function

    return Function('retention', *conds, alias=alias)


@register_function(
    name='uniq_up_to',
    clickhouse_name='uniqUpTo',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['uniqUpTo'],
    doc='Count uniques up to limit. Maps to uniqUpTo(n)(x).',
)
def _build_uniq_up_to(x, n: int = 5, alias=None):
    from .functions import Function

    return Function(f'uniqUpTo({n})', x, alias=alias)


@register_function(
    name='sequence_match',
    clickhouse_name='sequenceMatch',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['sequenceMatch'],
    doc='Sequence pattern matching. Maps to sequenceMatch(pattern)(ts, cond1, ...).',
)
def _build_sequence_match(pattern: str, ts, *conds, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function(f"sequenceMatch('{pattern}')", ts, *conds, alias=alias)


@register_function(
    name='sequence_count',
    clickhouse_name='sequenceCount',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['sequenceCount'],
    doc='Count sequence matches. Maps to sequenceCount(pattern)(ts, cond1, ...).',
)
def _build_sequence_count(pattern: str, ts, *conds, alias=None):
    from .functions import Function

    return Function(f"sequenceCount('{pattern}')", ts, *conds, alias=alias)


@register_function(
    name='window_funnel',
    clickhouse_name='windowFunnel',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['windowFunnel'],
    doc='Funnel analysis. Maps to windowFunnel(window)(ts, cond1, cond2, ...).',
)
def _build_window_funnel(window: int, ts, *conds, alias=None):
    from .functions import Function

    return Function(f'windowFunnel({window})', ts, *conds, alias=alias)


@register_function(
    name='exponential_moving_average',
    clickhouse_name='exponentialMovingAverage',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['exponentialMovingAverage', 'ema'],
    doc='Exponential moving average. Maps to exponentialMovingAverage(x, ts, alpha).',
)
def _build_exponential_moving_average(x, ts, alpha, alias=None):
    from .functions import Function

    return Function('exponentialMovingAverage', x, ts, alpha, alias=alias)


@register_function(
    name='rank_corr',
    clickhouse_name='rankCorr',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['rankCorr', 'spearmanRankCorr'],
    doc='Spearman rank correlation. Maps to rankCorr(x, y).',
)
def _build_rank_corr(x, y, alias=None):
    from .functions import Function

    return Function('rankCorr', x, y, alias=alias)


@register_function(
    name='contingency',
    clickhouse_name='contingency',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['contingency'],
    doc='Contingency coefficient. Maps to contingency(x, y).',
)
def _build_contingency(x, y, alias=None):
    from .functions import Function

    return Function('contingency', x, y, alias=alias)


@register_function(
    name='cramers_v',
    clickhouse_name='cramersV',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['cramersV'],
    doc="Cramer's V. Maps to cramersV(x, y).",
)
def _build_cramers_v(x, y, alias=None):
    from .functions import Function

    return Function('cramersV', x, y, alias=alias)


@register_function(
    name='theils_u',
    clickhouse_name='theilsU',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['theilsU'],
    doc="Theil's U. Maps to theilsU(x, y).",
)
def _build_theils_u(x, y, alias=None):
    from .functions import Function

    return Function('theilsU', x, y, alias=alias)


@register_function(
    name='mann_whitney_u_test',
    clickhouse_name='mannWhitneyUTest',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['mannWhitneyUTest'],
    doc='Mann-Whitney U test. Maps to mannWhitneyUTest(x, y).',
)
def _build_mann_whitney_u_test(x, y, alias=None):
    from .functions import Function

    return Function('mannWhitneyUTest', x, y, alias=alias)


@register_function(
    name='student_t_test',
    clickhouse_name='studentTTest',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['studentTTest'],
    doc="Student's t-test. Maps to studentTTest(x, y).",
)
def _build_student_t_test(x, y, alias=None):
    from .functions import Function

    return Function('studentTTest', x, y, alias=alias)


@register_function(
    name='welch_t_test',
    clickhouse_name='welchTTest',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['welchTTest'],
    doc="Welch's t-test. Maps to welchTTest(x, y).",
)
def _build_welch_t_test(x, y, alias=None):
    from .functions import Function

    return Function('welchTTest', x, y, alias=alias)


@register_function(
    name='kolmogorov_smirnov_test',
    clickhouse_name='kolmogorovSmirnovTest',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['kolmogorovSmirnovTest', 'ksTest'],
    doc='Kolmogorov-Smirnov test. Maps to kolmogorovSmirnovTest(x, y).',
)
def _build_kolmogorov_smirnov_test(x, y, alias=None):
    from .functions import Function

    return Function('kolmogorovSmirnovTest', x, y, alias=alias)


@register_function(
    name='delta_sum',
    clickhouse_name='deltaSum',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['deltaSum'],
    doc='Sum of differences. Maps to deltaSum(x).',
)
def _build_delta_sum(x, alias=None):
    from .functions import Function

    return Function('deltaSum', x, alias=alias)


@register_function(
    name='delta_sum_timestamp',
    clickhouse_name='deltaSumTimestamp',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['deltaSumTimestamp'],
    doc='Sum of differences with timestamp. Maps to deltaSumTimestamp(x, ts).',
)
def _build_delta_sum_timestamp(x, ts, alias=None):
    from .functions import Function

    return Function('deltaSumTimestamp', x, ts, alias=alias)


@register_function(
    name='bounding_ratio',
    clickhouse_name='boundingRatio',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['boundingRatio'],
    doc='Slope of bounding line. Maps to boundingRatio(x, y).',
)
def _build_bounding_ratio(x, y, alias=None):
    from .functions import Function

    return Function('boundingRatio', x, y, alias=alias)


@register_function(
    name='spark_bar',
    clickhouse_name='sparkBar',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['sparkBar', 'sparkbar'],
    doc='Sparkline bar chart. Maps to sparkBar(buckets)(x, y).',
)
def _build_spark_bar(x, y=None, buckets: int = 10, alias=None):
    from .functions import Function

    if y is not None:
        return Function(f'sparkBar({buckets})', x, y, alias=alias)
    return Function(f'sparkBar({buckets})', x, alias=alias)


# =============================================================================
# MORE WINDOW FUNCTIONS
# =============================================================================


@register_function(
    name='row_number_func',
    clickhouse_name='row_number',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    aliases=['row_number', 'rowNumber'],
    doc='Row number in window. Maps to row_number() OVER (...).',
)
def _build_row_number_func(alias=None):
    from .functions import Function

    return Function('row_number', alias=alias)


@register_function(
    name='rank_func',
    clickhouse_name='rank',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    aliases=['rank'],
    doc='Rank with gaps. Maps to rank() OVER (...).',
)
def _build_rank_func(alias=None):
    from .functions import Function

    return Function('rank', alias=alias)


@register_function(
    name='dense_rank_func',
    clickhouse_name='dense_rank',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    aliases=['dense_rank', 'denseRank'],
    doc='Dense rank without gaps. Maps to dense_rank() OVER (...).',
)
def _build_dense_rank_func(alias=None):
    from .functions import Function

    return Function('dense_rank', alias=alias)


@register_function(
    name='ntile_func',
    clickhouse_name='ntile',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    aliases=['ntile'],
    doc='N-tile bucket number. Maps to ntile(n) OVER (...).',
)
def _build_ntile_func(n: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('ntile', Literal(n), alias=alias)


@register_function(
    name='lead_func',
    clickhouse_name='lead',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    aliases=['lead'],
    doc='Value from next row. Maps to lead(x, offset, default) OVER (...).',
)
def _build_lead_func(x, offset: int = 1, default=None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if default is not None:
        return Function('lead', x, Literal(offset), default, alias=alias)
    return Function('lead', x, Literal(offset), alias=alias)


@register_function(
    name='lag_func',
    clickhouse_name='lag',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    aliases=['lag'],
    doc='Value from previous row. Maps to lag(x, offset, default) OVER (...).',
)
def _build_lag_func(x, offset: int = 1, default=None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if default is not None:
        return Function('lag', x, Literal(offset), default, alias=alias)
    return Function('lag', x, Literal(offset), alias=alias)


@register_function(
    name='nth_value_func',
    clickhouse_name='nth_value',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    aliases=['nth_value', 'nthValue'],
    doc='N-th value in window. Maps to nth_value(x, n) OVER (...).',
)
def _build_nth_value_func(x, n: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('nth_value', x, Literal(n), alias=alias)


@register_function(
    name='lag_in_frame',
    clickhouse_name='lagInFrame',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    aliases=['lagInFrame'],
    doc='Lag within frame. Maps to lagInFrame(x, offset, default).',
)
def _build_lag_in_frame(x, offset: int = 1, default=None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if default is not None:
        return Function('lagInFrame', x, Literal(offset), default, alias=alias)
    return Function('lagInFrame', x, Literal(offset), alias=alias)


@register_function(
    name='lead_in_frame',
    clickhouse_name='leadInFrame',
    func_type=FunctionType.WINDOW,
    category=FunctionCategory.WINDOW,
    aliases=['leadInFrame'],
    doc='Lead within frame. Maps to leadInFrame(x, offset, default).',
)
def _build_lead_in_frame(x, offset: int = 1, default=None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if default is not None:
        return Function('leadInFrame', x, Literal(offset), default, alias=alias)
    return Function('leadInFrame', x, Literal(offset), alias=alias)


# =============================================================================
# MORE DATETIME FUNCTIONS (Extended)
# =============================================================================


@register_function(
    name='to_yyyymm',
    clickhouse_name='toYYYYMM',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toYYYYMM'],
    doc='Convert to YYYYMM format. Maps to toYYYYMM(dt).',
)
def _build_to_yyyymm(dt, alias=None):
    from .functions import Function

    return Function('toYYYYMM', dt, alias=alias)


@register_function(
    name='to_yyyymmdd',
    clickhouse_name='toYYYYMMDD',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toYYYYMMDD'],
    doc='Convert to YYYYMMDD format. Maps to toYYYYMMDD(dt).',
)
def _build_to_yyyymmdd(dt, alias=None):
    from .functions import Function

    return Function('toYYYYMMDD', dt, alias=alias)


@register_function(
    name='to_yyyymmddhhmmss',
    clickhouse_name='toYYYYMMDDhhmmss',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toYYYYMMDDhhmmss'],
    doc='Convert to YYYYMMDDhhmmss format. Maps to toYYYYMMDDhhmmss(dt).',
)
def _build_to_yyyymmddhhmmss(dt, alias=None):
    from .functions import Function

    return Function('toYYYYMMDDhhmmss', dt, alias=alias)


@register_function(
    name='format_datetime',
    clickhouse_name='formatDateTime',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['formatDateTime'],
    doc='Format datetime. Maps to formatDateTime(dt, format).',
)
def _build_format_datetime(dt, format_str: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('formatDateTime', dt, Literal(format_str), alias=alias)


@register_function(
    name='parse_datetime',
    clickhouse_name='parseDateTime',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['parseDateTime'],
    doc='Parse datetime string. Maps to parseDateTime(s, format).',
)
def _build_parse_datetime(s, format_str: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('parseDateTime', s, Literal(format_str), alias=alias)


@register_function(
    name='to_timezone',
    clickhouse_name='toTimezone',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toTimezone', 'toTimeZone'],
    doc='Convert to timezone. Maps to toTimezone(dt, tz).',
)
def _build_to_timezone(dt, tz: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('toTimezone', dt, Literal(tz), alias=alias)


@register_function(
    name='timezone_func',
    clickhouse_name='timezone',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['timezone', 'serverTimezone'],
    doc='Get server timezone. Maps to timezone().',
)
def _build_timezone_func(alias=None):
    from .functions import Function

    return Function('timezone', alias=alias)


@register_function(
    name='timezone_of',
    clickhouse_name='timezoneOf',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['timezoneOf', 'timeZoneOf'],
    doc='Get timezone of datetime. Maps to timezoneOf(dt).',
)
def _build_timezone_of(dt, alias=None):
    from .functions import Function

    return Function('timezoneOf', dt, alias=alias)


@register_function(
    name='to_start_of_second',
    clickhouse_name='toStartOfSecond',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toStartOfSecond'],
    doc='Truncate to second. Maps to toStartOfSecond(dt).',
)
def _build_to_start_of_second(dt, alias=None):
    from .functions import Function

    return Function('toStartOfSecond', dt, alias=alias)


@register_function(
    name='to_start_of_minute',
    clickhouse_name='toStartOfMinute',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toStartOfMinute'],
    doc='Truncate to minute. Maps to toStartOfMinute(dt).',
)
def _build_to_start_of_minute(dt, alias=None):
    from .functions import Function

    return Function('toStartOfMinute', dt, alias=alias)


@register_function(
    name='to_start_of_five_minutes',
    clickhouse_name='toStartOfFiveMinutes',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toStartOfFiveMinutes'],
    doc='Truncate to 5 minutes. Maps to toStartOfFiveMinutes(dt).',
)
def _build_to_start_of_five_minutes(dt, alias=None):
    from .functions import Function

    return Function('toStartOfFiveMinutes', dt, alias=alias)


@register_function(
    name='to_start_of_ten_minutes',
    clickhouse_name='toStartOfTenMinutes',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toStartOfTenMinutes'],
    doc='Truncate to 10 minutes. Maps to toStartOfTenMinutes(dt).',
)
def _build_to_start_of_ten_minutes(dt, alias=None):
    from .functions import Function

    return Function('toStartOfTenMinutes', dt, alias=alias)


@register_function(
    name='to_start_of_fifteen_minutes',
    clickhouse_name='toStartOfFifteenMinutes',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toStartOfFifteenMinutes'],
    doc='Truncate to 15 minutes. Maps to toStartOfFifteenMinutes(dt).',
)
def _build_to_start_of_fifteen_minutes(dt, alias=None):
    from .functions import Function

    return Function('toStartOfFifteenMinutes', dt, alias=alias)


@register_function(
    name='to_start_of_interval',
    clickhouse_name='toStartOfInterval',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toStartOfInterval'],
    doc='Truncate to interval. Maps to toStartOfInterval(dt, INTERVAL n unit).',
)
def _build_to_start_of_interval(dt, interval, alias=None):
    from .functions import Function

    return Function('toStartOfInterval', dt, interval, alias=alias)


@register_function(
    name='to_time',
    clickhouse_name='toTime',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toTime'],
    doc='Extract time part. Maps to toTime(dt).',
)
def _build_to_time(dt, alias=None):
    from .functions import Function

    return Function('toTime', dt, alias=alias)


@register_function(
    name='to_year_week',
    clickhouse_name='toYearWeek',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toYearWeek'],
    doc='Get year and week number. Maps to toYearWeek(dt, mode).',
)
def _build_to_year_week(dt, mode: int = 0, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('toYearWeek', dt, Literal(mode), alias=alias)


@register_function(
    name='to_iso_year',
    clickhouse_name='toISOYear',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toISOYear'],
    doc='Get ISO year. Maps to toISOYear(dt).',
)
def _build_to_iso_year(dt, alias=None):
    from .functions import Function

    return Function('toISOYear', dt, alias=alias)


@register_function(
    name='to_days_since_year_zero',
    clickhouse_name='toDaysSinceYearZero',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toDaysSinceYearZero'],
    doc='Days since year 0. Maps to toDaysSinceYearZero(dt).',
)
def _build_to_days_since_year_zero(dt, alias=None):
    from .functions import Function

    return Function('toDaysSinceYearZero', dt, alias=alias)


@register_function(
    name='from_days_since_year_zero',
    clickhouse_name='fromDaysSinceYearZero',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['fromDaysSinceYearZero'],
    doc='Date from days since year 0. Maps to fromDaysSinceYearZero(days).',
)
def _build_from_days_since_year_zero(days, alias=None):
    from .functions import Function

    return Function('fromDaysSinceYearZero', days, alias=alias)


@register_function(
    name='utc_timestamp',
    clickhouse_name='UTCTimestamp',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['UTCTimestamp'],
    doc='Current UTC timestamp. Maps to UTCTimestamp().',
)
def _build_utc_timestamp(alias=None):
    from .functions import Function

    return Function('UTCTimestamp', alias=alias)


# =============================================================================
# MORE ENCODING FUNCTIONS
# =============================================================================


@register_function(
    name='hex_func',
    clickhouse_name='hex',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ENCODING,
    aliases=['hex'],
    doc='Convert to hex string. Maps to hex(x).',
)
def _build_hex_func(x, alias=None):
    from .functions import Function

    return Function('hex', x, alias=alias)


@register_function(
    name='unhex_func',
    clickhouse_name='unhex',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ENCODING,
    aliases=['unhex'],
    doc='Convert from hex string. Maps to unhex(s).',
)
def _build_unhex_func(s, alias=None):
    from .functions import Function

    return Function('unhex', s, alias=alias)


@register_function(
    name='bin_func',
    clickhouse_name='bin',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ENCODING,
    aliases=['bin'],
    doc='Convert to binary string. Maps to bin(x).',
)
def _build_bin_func(x, alias=None):
    from .functions import Function

    return Function('bin', x, alias=alias)


@register_function(
    name='unbin_func',
    clickhouse_name='unbin',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ENCODING,
    aliases=['unbin'],
    doc='Convert from binary string. Maps to unbin(s).',
)
def _build_unbin_func(s, alias=None):
    from .functions import Function

    return Function('unbin', s, alias=alias)


@register_function(
    name='bitmask_to_array',
    clickhouse_name='bitmaskToArray',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ENCODING,
    aliases=['bitmaskToArray'],
    doc='Convert bitmask to array of bit positions. Maps to bitmaskToArray(x).',
)
def _build_bitmask_to_array(x, alias=None):
    from .functions import Function

    return Function('bitmaskToArray', x, alias=alias)


@register_function(
    name='bitmask_to_list',
    clickhouse_name='bitmaskToList',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ENCODING,
    aliases=['bitmaskToList'],
    doc='Convert bitmask to list string. Maps to bitmaskToList(x).',
)
def _build_bitmask_to_list(x, alias=None):
    from .functions import Function

    return Function('bitmaskToList', x, alias=alias)


# =============================================================================
# UTILITY FUNCTIONS (Extended)
# =============================================================================


@register_function(
    name='coalesce_func',
    clickhouse_name='coalesce',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['coalesce'],
    doc='Return first non-NULL. Maps to coalesce(x1, x2, ...).',
)
def _build_coalesce_func(*args, alias=None):
    from .functions import Function

    return Function('coalesce', *args, alias=alias)


@register_function(
    name='if_not_finite',
    clickhouse_name='ifNotFinite',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['ifNotFinite'],
    doc='Replace non-finite with value. Maps to ifNotFinite(x, alt).',
)
def _build_if_not_finite(x, alt, alias=None):
    from .functions import Function

    return Function('ifNotFinite', x, alt, alias=alias)


@register_function(
    name='is_finite',
    clickhouse_name='isFinite',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['isFinite'],
    doc='Check if finite. Maps to isFinite(x).',
)
def _build_is_finite(x, alias=None):
    from .functions import Function

    return Function('isFinite', x, alias=alias)


@register_function(
    name='is_infinite',
    clickhouse_name='isInfinite',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['isInfinite'],
    doc='Check if infinite. Maps to isInfinite(x).',
)
def _build_is_infinite(x, alias=None):
    from .functions import Function

    return Function('isInfinite', x, alias=alias)


@register_function(
    name='is_nan',
    clickhouse_name='isNaN',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['isNaN'],
    doc='Check if NaN. Maps to isNaN(x).',
)
def _build_is_nan(x, alias=None):
    from .functions import Function

    return Function('isNaN', x, alias=alias)


@register_function(
    name='transform_func',
    clickhouse_name='transform',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['transform'],
    doc='Transform values. Maps to transform(x, from, to, default).',
)
def _build_transform_func(x, from_arr, to_arr, default=None, alias=None):
    from .functions import Function

    if default is not None:
        return Function('transform', x, from_arr, to_arr, default, alias=alias)
    return Function('transform', x, from_arr, to_arr, alias=alias)


@register_function(
    name='bar_func',
    clickhouse_name='bar',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['bar'],
    doc='Draw bar chart. Maps to bar(x, min, max, width).',
)
def _build_bar_func(x, min_val, max_val, width: int = 80, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('bar', x, min_val, max_val, Literal(width), alias=alias)


@register_function(
    name='array_join_func',
    clickhouse_name='arrayJoin',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['arrayJoin'],
    doc='Expand array to rows. Maps to arrayJoin(arr).',
)
def _build_array_join_func(arr, alias=None):
    from .functions import Function

    return Function('arrayJoin', arr, alias=alias)


@register_function(
    name='running_difference',
    clickhouse_name='runningDifference',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['runningDifference'],
    doc='Running difference. Maps to runningDifference(x).',
)
def _build_running_difference(x, alias=None):
    from .functions import Function

    return Function('runningDifference', x, alias=alias)


@register_function(
    name='running_difference_starting_with_first_value',
    clickhouse_name='runningDifferenceStartingWithFirstValue',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['runningDifferenceStartingWithFirstValue'],
    doc='Running difference starting with first. Maps to runningDifferenceStartingWithFirstValue(x).',
)
def _build_running_difference_starting_with_first_value(x, alias=None):
    from .functions import Function

    return Function('runningDifferenceStartingWithFirstValue', x, alias=alias)


@register_function(
    name='neighbor_func',
    clickhouse_name='neighbor',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['neighbor'],
    doc='Get neighbor value. Maps to neighbor(x, offset, default).',
)
def _build_neighbor_func(x, offset: int, default=None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if default is not None:
        return Function('neighbor', x, Literal(offset), default, alias=alias)
    return Function('neighbor', x, Literal(offset), alias=alias)


@register_function(
    name='current_database',
    clickhouse_name='currentDatabase',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['currentDatabase', 'database'],
    doc='Get current database. Maps to currentDatabase().',
)
def _build_current_database(alias=None):
    from .functions import Function

    return Function('currentDatabase', alias=alias)


@register_function(
    name='current_user',
    clickhouse_name='currentUser',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['currentUser', 'user'],
    doc='Get current user. Maps to currentUser().',
)
def _build_current_user(alias=None):
    from .functions import Function

    return Function('currentUser', alias=alias)


@register_function(
    name='host_name',
    clickhouse_name='hostName',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['hostName', 'hostname'],
    doc='Get hostname. Maps to hostName().',
)
def _build_host_name(alias=None):
    from .functions import Function

    return Function('hostName', alias=alias)


@register_function(
    name='version_func',
    clickhouse_name='version',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['version'],
    doc='Get ClickHouse version. Maps to version().',
)
def _build_version_func(alias=None):
    from .functions import Function

    return Function('version', alias=alias)


@register_function(
    name='uptime_func',
    clickhouse_name='uptime',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['uptime'],
    doc='Get server uptime in seconds. Maps to uptime().',
)
def _build_uptime_func(alias=None):
    from .functions import Function

    return Function('uptime', alias=alias)


@register_function(
    name='block_size',
    clickhouse_name='blockSize',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['blockSize'],
    doc='Get block size. Maps to blockSize().',
)
def _build_block_size(alias=None):
    from .functions import Function

    return Function('blockSize', alias=alias)


@register_function(
    name='block_number',
    clickhouse_name='blockNumber',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['blockNumber'],
    doc='Get block number. Maps to blockNumber().',
)
def _build_block_number(alias=None):
    from .functions import Function

    return Function('blockNumber', alias=alias)


@register_function(
    name='row_number_in_block',
    clickhouse_name='rowNumberInBlock',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['rowNumberInBlock'],
    doc='Row number within block. Maps to rowNumberInBlock().',
)
def _build_row_number_in_block(alias=None):
    from .functions import Function

    return Function('rowNumberInBlock', alias=alias)


@register_function(
    name='row_number_in_all_blocks',
    clickhouse_name='rowNumberInAllBlocks',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['rowNumberInAllBlocks'],
    doc='Row number across all blocks. Maps to rowNumberInAllBlocks().',
)
def _build_row_number_in_all_blocks(alias=None):
    from .functions import Function

    return Function('rowNumberInAllBlocks', alias=alias)


@register_function(
    name='identity_func',
    clickhouse_name='identity',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['identity'],
    doc='Identity function. Maps to identity(x).',
)
def _build_identity_func(x, alias=None):
    from .functions import Function

    return Function('identity', x, alias=alias)


@register_function(
    name='is_constant',
    clickhouse_name='isConstant',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['isConstant'],
    doc='Check if constant. Maps to isConstant(x).',
)
def _build_is_constant(x, alias=None):
    from .functions import Function

    return Function('isConstant', x, alias=alias)


@register_function(
    name='byte_size',
    clickhouse_name='byteSize',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['byteSize'],
    doc='Get byte size. Maps to byteSize(x).',
)
def _build_byte_size(x, alias=None):
    from .functions import Function

    return Function('byteSize', x, alias=alias)


@register_function(
    name='filesystem_available',
    clickhouse_name='filesystemAvailable',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['filesystemAvailable'],
    doc='Get available filesystem space. Maps to filesystemAvailable().',
)
def _build_filesystem_available(alias=None):
    from .functions import Function

    return Function('filesystemAvailable', alias=alias)


@register_function(
    name='filesystem_capacity',
    clickhouse_name='filesystemCapacity',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['filesystemCapacity'],
    doc='Get filesystem capacity. Maps to filesystemCapacity().',
)
def _build_filesystem_capacity(alias=None):
    from .functions import Function

    return Function('filesystemCapacity', alias=alias)


@register_function(
    name='to_interval_second',
    clickhouse_name='toIntervalSecond',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toIntervalSecond'],
    doc='Create interval in seconds. Maps to toIntervalSecond(n).',
)
def _build_to_interval_second(n, alias=None):
    from .functions import Function

    return Function('toIntervalSecond', n, alias=alias)


@register_function(
    name='to_interval_minute',
    clickhouse_name='toIntervalMinute',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toIntervalMinute'],
    doc='Create interval in minutes. Maps to toIntervalMinute(n).',
)
def _build_to_interval_minute(n, alias=None):
    from .functions import Function

    return Function('toIntervalMinute', n, alias=alias)


@register_function(
    name='to_interval_hour',
    clickhouse_name='toIntervalHour',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toIntervalHour'],
    doc='Create interval in hours. Maps to toIntervalHour(n).',
)
def _build_to_interval_hour(n, alias=None):
    from .functions import Function

    return Function('toIntervalHour', n, alias=alias)


@register_function(
    name='to_interval_day',
    clickhouse_name='toIntervalDay',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toIntervalDay'],
    doc='Create interval in days. Maps to toIntervalDay(n).',
)
def _build_to_interval_day(n, alias=None):
    from .functions import Function

    return Function('toIntervalDay', n, alias=alias)


@register_function(
    name='to_interval_week',
    clickhouse_name='toIntervalWeek',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toIntervalWeek'],
    doc='Create interval in weeks. Maps to toIntervalWeek(n).',
)
def _build_to_interval_week(n, alias=None):
    from .functions import Function

    return Function('toIntervalWeek', n, alias=alias)


@register_function(
    name='to_interval_month',
    clickhouse_name='toIntervalMonth',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toIntervalMonth'],
    doc='Create interval in months. Maps to toIntervalMonth(n).',
)
def _build_to_interval_month(n, alias=None):
    from .functions import Function

    return Function('toIntervalMonth', n, alias=alias)


@register_function(
    name='to_interval_quarter',
    clickhouse_name='toIntervalQuarter',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toIntervalQuarter'],
    doc='Create interval in quarters. Maps to toIntervalQuarter(n).',
)
def _build_to_interval_quarter(n, alias=None):
    from .functions import Function

    return Function('toIntervalQuarter', n, alias=alias)


@register_function(
    name='to_interval_year',
    clickhouse_name='toIntervalYear',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toIntervalYear'],
    doc='Create interval in years. Maps to toIntervalYear(n).',
)
def _build_to_interval_year(n, alias=None):
    from .functions import Function

    return Function('toIntervalYear', n, alias=alias)


@register_function(
    name='to_decimal_string',
    clickhouse_name='toDecimalString',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toDecimalString'],
    doc='Convert to decimal string. Maps to toDecimalString(x, scale).',
)
def _build_to_decimal_string(x, scale: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('toDecimalString', x, Literal(scale), alias=alias)


@register_function(
    name='accurate_cast',
    clickhouse_name='accurateCast',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['accurateCast'],
    doc='Accurate type cast. Maps to accurateCast(x, T).',
)
def _build_accurate_cast(x, type_name: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('accurateCast', x, Literal(type_name), alias=alias)


# =============================================================================
# TYPE CONVERSION FUNCTIONS (Complete from ClickHouse docs)
# https://clickhouse.com/docs/sql-reference/functions/type-conversion-functions
# =============================================================================


# --- Integer conversions ---
@register_function(
    name='to_int8',
    clickhouse_name='toInt8',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toInt8'],
    doc='Convert to Int8. Maps to toInt8(x).',
)
def _build_to_int8(x, alias=None):
    from .functions import Function

    return Function('toInt8', x, alias=alias)


@register_function(
    name='to_int16',
    clickhouse_name='toInt16',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toInt16'],
    doc='Convert to Int16. Maps to toInt16(x).',
)
def _build_to_int16(x, alias=None):
    from .functions import Function

    return Function('toInt16', x, alias=alias)


@register_function(
    name='to_int32',
    clickhouse_name='toInt32',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toInt32'],
    doc='Convert to Int32. Maps to toInt32(x).',
)
def _build_to_int32(x, alias=None):
    from .functions import Function

    return Function('toInt32', x, alias=alias)


@register_function(
    name='to_int64',
    clickhouse_name='toInt64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toInt64'],
    doc='Convert to Int64. Maps to toInt64(x).',
)
def _build_to_int64(x, alias=None):
    from .functions import Function

    return Function('toInt64', x, alias=alias)


@register_function(
    name='to_int128',
    clickhouse_name='toInt128',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toInt128'],
    doc='Convert to Int128. Maps to toInt128(x).',
)
def _build_to_int128(x, alias=None):
    from .functions import Function

    return Function('toInt128', x, alias=alias)


@register_function(
    name='to_int256',
    clickhouse_name='toInt256',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toInt256'],
    doc='Convert to Int256. Maps to toInt256(x).',
)
def _build_to_int256(x, alias=None):
    from .functions import Function

    return Function('toInt256', x, alias=alias)


# --- Unsigned integer conversions ---
@register_function(
    name='to_uint8',
    clickhouse_name='toUInt8',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toUInt8'],
    doc='Convert to UInt8. Maps to toUInt8(x).',
)
def _build_to_uint8(x, alias=None):
    from .functions import Function

    return Function('toUInt8', x, alias=alias)


@register_function(
    name='to_uint16',
    clickhouse_name='toUInt16',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toUInt16'],
    doc='Convert to UInt16. Maps to toUInt16(x).',
)
def _build_to_uint16(x, alias=None):
    from .functions import Function

    return Function('toUInt16', x, alias=alias)


@register_function(
    name='to_uint32',
    clickhouse_name='toUInt32',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toUInt32'],
    doc='Convert to UInt32. Maps to toUInt32(x).',
)
def _build_to_uint32(x, alias=None):
    from .functions import Function

    return Function('toUInt32', x, alias=alias)


@register_function(
    name='to_uint64',
    clickhouse_name='toUInt64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toUInt64'],
    doc='Convert to UInt64. Maps to toUInt64(x).',
)
def _build_to_uint64(x, alias=None):
    from .functions import Function

    return Function('toUInt64', x, alias=alias)


@register_function(
    name='to_uint128',
    clickhouse_name='toUInt128',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toUInt128'],
    doc='Convert to UInt128. Maps to toUInt128(x).',
)
def _build_to_uint128(x, alias=None):
    from .functions import Function

    return Function('toUInt128', x, alias=alias)


@register_function(
    name='to_uint256',
    clickhouse_name='toUInt256',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toUInt256'],
    doc='Convert to UInt256. Maps to toUInt256(x).',
)
def _build_to_uint256(x, alias=None):
    from .functions import Function

    return Function('toUInt256', x, alias=alias)


# --- Float conversions ---
@register_function(
    name='to_float32',
    clickhouse_name='toFloat32',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toFloat32'],
    doc='Convert to Float32. Maps to toFloat32(x).',
)
def _build_to_float32(x, alias=None):
    from .functions import Function

    return Function('toFloat32', x, alias=alias)


@register_function(
    name='to_float64',
    clickhouse_name='toFloat64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toFloat64'],
    doc='Convert to Float64. Maps to toFloat64(x).',
)
def _build_to_float64(x, alias=None):
    from .functions import Function

    return Function('toFloat64', x, alias=alias)


@register_function(
    name='to_bfloat16',
    clickhouse_name='toBFloat16',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toBFloat16'],
    doc='Convert to BFloat16. Maps to toBFloat16(x).',
)
def _build_to_bfloat16(x, alias=None):
    from .functions import Function

    return Function('toBFloat16', x, alias=alias)


# --- Date/DateTime conversions ---
@register_function(
    name='to_date',
    clickhouse_name='toDate',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toDate'],
    doc='Convert to Date. Maps to toDate(x).',
)
def _build_to_date(x, alias=None):
    from .functions import Function

    return Function('toDate', x, alias=alias)


@register_function(
    name='to_date32',
    clickhouse_name='toDate32',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toDate32'],
    doc='Convert to Date32. Maps to toDate32(x).',
)
def _build_to_date32(x, alias=None):
    from .functions import Function

    return Function('toDate32', x, alias=alias)


@register_function(
    name='to_datetime',
    clickhouse_name='toDateTime',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toDateTime'],
    doc='Convert to DateTime. Maps to toDateTime(x).',
)
def _build_to_datetime(x, alias=None):
    from .functions import Function

    return Function('toDateTime', x, alias=alias)


@register_function(
    name='to_datetime32',
    clickhouse_name='toDateTime32',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toDateTime32'],
    doc='Convert to DateTime32. Maps to toDateTime32(x).',
)
def _build_to_datetime32(x, alias=None):
    from .functions import Function

    return Function('toDateTime32', x, alias=alias)


@register_function(
    name='to_datetime64',
    clickhouse_name='toDateTime64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toDateTime64'],
    doc='Convert to DateTime64. Maps to toDateTime64(x, scale).',
)
def _build_to_datetime64(x, scale: int = 3, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('toDateTime64', x, Literal(scale), alias=alias)


# --- Decimal conversions ---
@register_function(
    name='to_decimal32',
    clickhouse_name='toDecimal32',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toDecimal32'],
    doc='Convert to Decimal32. Maps to toDecimal32(x, scale).',
)
def _build_to_decimal32(x, scale: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('toDecimal32', x, Literal(scale), alias=alias)


@register_function(
    name='to_decimal64',
    clickhouse_name='toDecimal64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toDecimal64'],
    doc='Convert to Decimal64. Maps to toDecimal64(x, scale).',
)
def _build_to_decimal64(x, scale: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('toDecimal64', x, Literal(scale), alias=alias)


@register_function(
    name='to_decimal128',
    clickhouse_name='toDecimal128',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toDecimal128'],
    doc='Convert to Decimal128. Maps to toDecimal128(x, scale).',
)
def _build_to_decimal128(x, scale: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('toDecimal128', x, Literal(scale), alias=alias)


@register_function(
    name='to_decimal256',
    clickhouse_name='toDecimal256',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toDecimal256'],
    doc='Convert to Decimal256. Maps to toDecimal256(x, scale).',
)
def _build_to_decimal256(x, scale: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('toDecimal256', x, Literal(scale), alias=alias)


# --- String conversions ---
@register_function(
    name='to_string',
    clickhouse_name='toString',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toString'],
    doc='Convert to String. Maps to toString(x).',
)
def _build_to_string(x, alias=None):
    from .functions import Function

    return Function('toString', x, alias=alias)


@register_function(
    name='to_fixed_string',
    clickhouse_name='toFixedString',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toFixedString'],
    doc='Convert to FixedString. Maps to toFixedString(s, n).',
)
def _build_to_fixed_string(s, n: int, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('toFixedString', s, Literal(n), alias=alias)


@register_function(
    name='to_string_cut_to_zero',
    clickhouse_name='toStringCutToZero',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toStringCutToZero'],
    doc='Convert to string, cut at zero byte. Maps to toStringCutToZero(s).',
)
def _build_to_string_cut_to_zero(s, alias=None):
    from .functions import Function

    return Function('toStringCutToZero', s, alias=alias)


# --- Reinterpret functions ---
@register_function(
    name='reinterpret_as_uint8',
    clickhouse_name='reinterpretAsUInt8',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpretAsUInt8'],
    doc='Reinterpret as UInt8. Maps to reinterpretAsUInt8(x).',
)
def _build_reinterpret_as_uint8(x, alias=None):
    from .functions import Function

    return Function('reinterpretAsUInt8', x, alias=alias)


@register_function(
    name='reinterpret_as_uint16',
    clickhouse_name='reinterpretAsUInt16',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpretAsUInt16'],
    doc='Reinterpret as UInt16. Maps to reinterpretAsUInt16(x).',
)
def _build_reinterpret_as_uint16(x, alias=None):
    from .functions import Function

    return Function('reinterpretAsUInt16', x, alias=alias)


@register_function(
    name='reinterpret_as_uint32',
    clickhouse_name='reinterpretAsUInt32',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpretAsUInt32'],
    doc='Reinterpret as UInt32. Maps to reinterpretAsUInt32(x).',
)
def _build_reinterpret_as_uint32(x, alias=None):
    from .functions import Function

    return Function('reinterpretAsUInt32', x, alias=alias)


@register_function(
    name='reinterpret_as_uint64',
    clickhouse_name='reinterpretAsUInt64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpretAsUInt64'],
    doc='Reinterpret as UInt64. Maps to reinterpretAsUInt64(x).',
)
def _build_reinterpret_as_uint64(x, alias=None):
    from .functions import Function

    return Function('reinterpretAsUInt64', x, alias=alias)


@register_function(
    name='reinterpret_as_uint128',
    clickhouse_name='reinterpretAsUInt128',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpretAsUInt128'],
    doc='Reinterpret as UInt128. Maps to reinterpretAsUInt128(x).',
)
def _build_reinterpret_as_uint128(x, alias=None):
    from .functions import Function

    return Function('reinterpretAsUInt128', x, alias=alias)


@register_function(
    name='reinterpret_as_uint256',
    clickhouse_name='reinterpretAsUInt256',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpretAsUInt256'],
    doc='Reinterpret as UInt256. Maps to reinterpretAsUInt256(x).',
)
def _build_reinterpret_as_uint256(x, alias=None):
    from .functions import Function

    return Function('reinterpretAsUInt256', x, alias=alias)


@register_function(
    name='reinterpret_as_int8',
    clickhouse_name='reinterpretAsInt8',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpretAsInt8'],
    doc='Reinterpret as Int8. Maps to reinterpretAsInt8(x).',
)
def _build_reinterpret_as_int8(x, alias=None):
    from .functions import Function

    return Function('reinterpretAsInt8', x, alias=alias)


@register_function(
    name='reinterpret_as_int16',
    clickhouse_name='reinterpretAsInt16',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpretAsInt16'],
    doc='Reinterpret as Int16. Maps to reinterpretAsInt16(x).',
)
def _build_reinterpret_as_int16(x, alias=None):
    from .functions import Function

    return Function('reinterpretAsInt16', x, alias=alias)


@register_function(
    name='reinterpret_as_int32',
    clickhouse_name='reinterpretAsInt32',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpretAsInt32'],
    doc='Reinterpret as Int32. Maps to reinterpretAsInt32(x).',
)
def _build_reinterpret_as_int32(x, alias=None):
    from .functions import Function

    return Function('reinterpretAsInt32', x, alias=alias)


@register_function(
    name='reinterpret_as_int64',
    clickhouse_name='reinterpretAsInt64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpretAsInt64'],
    doc='Reinterpret as Int64. Maps to reinterpretAsInt64(x).',
)
def _build_reinterpret_as_int64(x, alias=None):
    from .functions import Function

    return Function('reinterpretAsInt64', x, alias=alias)


@register_function(
    name='reinterpret_as_int128',
    clickhouse_name='reinterpretAsInt128',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpretAsInt128'],
    doc='Reinterpret as Int128. Maps to reinterpretAsInt128(x).',
)
def _build_reinterpret_as_int128(x, alias=None):
    from .functions import Function

    return Function('reinterpretAsInt128', x, alias=alias)


@register_function(
    name='reinterpret_as_int256',
    clickhouse_name='reinterpretAsInt256',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpretAsInt256'],
    doc='Reinterpret as Int256. Maps to reinterpretAsInt256(x).',
)
def _build_reinterpret_as_int256(x, alias=None):
    from .functions import Function

    return Function('reinterpretAsInt256', x, alias=alias)


@register_function(
    name='reinterpret_as_float32',
    clickhouse_name='reinterpretAsFloat32',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpretAsFloat32'],
    doc='Reinterpret as Float32. Maps to reinterpretAsFloat32(x).',
)
def _build_reinterpret_as_float32(x, alias=None):
    from .functions import Function

    return Function('reinterpretAsFloat32', x, alias=alias)


@register_function(
    name='reinterpret_as_float64',
    clickhouse_name='reinterpretAsFloat64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpretAsFloat64'],
    doc='Reinterpret as Float64. Maps to reinterpretAsFloat64(x).',
)
def _build_reinterpret_as_float64(x, alias=None):
    from .functions import Function

    return Function('reinterpretAsFloat64', x, alias=alias)


@register_function(
    name='reinterpret_as_date',
    clickhouse_name='reinterpretAsDate',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpretAsDate'],
    doc='Reinterpret as Date. Maps to reinterpretAsDate(x).',
)
def _build_reinterpret_as_date(x, alias=None):
    from .functions import Function

    return Function('reinterpretAsDate', x, alias=alias)


@register_function(
    name='reinterpret_as_datetime',
    clickhouse_name='reinterpretAsDateTime',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpretAsDateTime'],
    doc='Reinterpret as DateTime. Maps to reinterpretAsDateTime(x).',
)
def _build_reinterpret_as_datetime(x, alias=None):
    from .functions import Function

    return Function('reinterpretAsDateTime', x, alias=alias)


@register_function(
    name='reinterpret_as_string',
    clickhouse_name='reinterpretAsString',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpretAsString'],
    doc='Reinterpret as String. Maps to reinterpretAsString(x).',
)
def _build_reinterpret_as_string(x, alias=None):
    from .functions import Function

    return Function('reinterpretAsString', x, alias=alias)


@register_function(
    name='reinterpret_as_fixed_string',
    clickhouse_name='reinterpretAsFixedString',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpretAsFixedString'],
    doc='Reinterpret as FixedString. Maps to reinterpretAsFixedString(x).',
)
def _build_reinterpret_as_fixed_string(x, alias=None):
    from .functions import Function

    return Function('reinterpretAsFixedString', x, alias=alias)


@register_function(
    name='reinterpret_as_uuid',
    clickhouse_name='reinterpretAsUUID',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpretAsUUID'],
    doc='Reinterpret as UUID. Maps to reinterpretAsUUID(x).',
)
def _build_reinterpret_as_uuid(x, alias=None):
    from .functions import Function

    return Function('reinterpretAsUUID', x, alias=alias)


@register_function(
    name='reinterpret',
    clickhouse_name='reinterpret',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['reinterpret'],
    doc='Generic reinterpret. Maps to reinterpret(x, T).',
)
def _build_reinterpret(x, type_name: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('reinterpret', x, Literal(type_name), alias=alias)


# --- CAST ---
@register_function(
    name='cast_func',
    clickhouse_name='CAST',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['CAST', 'cast'],
    doc='Type cast. Maps to CAST(x AS T) or CAST(x, T).',
)
def _build_cast_func(x, type_name: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('CAST', x, Literal(type_name), alias=alias)


@register_function(
    name='accurate_cast_or_null',
    clickhouse_name='accurateCastOrNull',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['accurateCastOrNull'],
    doc='Accurate cast or NULL. Maps to accurateCastOrNull(x, T).',
)
def _build_accurate_cast_or_null(x, type_name: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('accurateCastOrNull', x, Literal(type_name), alias=alias)


@register_function(
    name='accurate_cast_or_default',
    clickhouse_name='accurateCastOrDefault',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['accurateCastOrDefault'],
    doc='Accurate cast or default. Maps to accurateCastOrDefault(x, T, default).',
)
def _build_accurate_cast_or_default(x, type_name: str, default=None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if default is not None:
        return Function('accurateCastOrDefault', x, Literal(type_name), default, alias=alias)
    return Function('accurateCastOrDefault', x, Literal(type_name), alias=alias)


# --- parseDateTime variants ---
@register_function(
    name='parse_datetime_or_null',
    clickhouse_name='parseDateTimeOrNull',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['parseDateTimeOrNull'],
    doc='Parse datetime or NULL. Maps to parseDateTimeOrNull(s, format).',
)
def _build_parse_datetime_or_null(s, format_str: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('parseDateTimeOrNull', s, Literal(format_str), alias=alias)


@register_function(
    name='parse_datetime_or_zero',
    clickhouse_name='parseDateTimeOrZero',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['parseDateTimeOrZero'],
    doc='Parse datetime or zero. Maps to parseDateTimeOrZero(s, format).',
)
def _build_parse_datetime_or_zero(s, format_str: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('parseDateTimeOrZero', s, Literal(format_str), alias=alias)


@register_function(
    name='parse_datetime_in_joda_syntax',
    clickhouse_name='parseDateTimeInJodaSyntax',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['parseDateTimeInJodaSyntax'],
    doc='Parse datetime with Joda syntax. Maps to parseDateTimeInJodaSyntax(s, format).',
)
def _build_parse_datetime_in_joda_syntax(s, format_str: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('parseDateTimeInJodaSyntax', s, Literal(format_str), alias=alias)


@register_function(
    name='parse_datetime64',
    clickhouse_name='parseDateTime64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['parseDateTime64'],
    doc='Parse DateTime64. Maps to parseDateTime64(s, scale, format).',
)
def _build_parse_datetime64(s, scale: int, format_str: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('parseDateTime64', s, Literal(scale), Literal(format_str), alias=alias)


@register_function(
    name='parse_datetime64_best_effort',
    clickhouse_name='parseDateTime64BestEffort',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['parseDateTime64BestEffort'],
    doc='Parse DateTime64 with best effort. Maps to parseDateTime64BestEffort(s).',
)
def _build_parse_datetime64_best_effort(s, alias=None):
    from .functions import Function

    return Function('parseDateTime64BestEffort', s, alias=alias)


@register_function(
    name='parse_datetime64_best_effort_or_null',
    clickhouse_name='parseDateTime64BestEffortOrNull',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['parseDateTime64BestEffortOrNull'],
    doc='Parse DateTime64 best effort or NULL. Maps to parseDateTime64BestEffortOrNull(s).',
)
def _build_parse_datetime64_best_effort_or_null(s, alias=None):
    from .functions import Function

    return Function('parseDateTime64BestEffortOrNull', s, alias=alias)


@register_function(
    name='parse_datetime64_best_effort_or_zero',
    clickhouse_name='parseDateTime64BestEffortOrZero',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['parseDateTime64BestEffortOrZero'],
    doc='Parse DateTime64 best effort or zero. Maps to parseDateTime64BestEffortOrZero(s).',
)
def _build_parse_datetime64_best_effort_or_zero(s, alias=None):
    from .functions import Function

    return Function('parseDateTime64BestEffortOrZero', s, alias=alias)


@register_function(
    name='parse_datetime_best_effort_us',
    clickhouse_name='parseDateTimeBestEffortUS',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['parseDateTimeBestEffortUS'],
    doc='Parse datetime best effort US format. Maps to parseDateTimeBestEffortUS(s).',
)
def _build_parse_datetime_best_effort_us(s, alias=None):
    from .functions import Function

    return Function('parseDateTimeBestEffortUS', s, alias=alias)


# --- toLowCardinality ---
@register_function(
    name='to_low_cardinality',
    clickhouse_name='toLowCardinality',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toLowCardinality'],
    doc='Convert to LowCardinality. Maps to toLowCardinality(x).',
)
def _build_to_low_cardinality(x, alias=None):
    from .functions import Function

    return Function('toLowCardinality', x, alias=alias)


# --- toNullable ---
@register_function(
    name='to_nullable',
    clickhouse_name='toNullable',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toNullable'],
    doc='Convert to Nullable. Maps to toNullable(x).',
)
def _build_to_nullable(x, alias=None):
    from .functions import Function

    return Function('toNullable', x, alias=alias)


@register_function(
    name='assume_not_null',
    clickhouse_name='assumeNotNull',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['assumeNotNull'],
    doc='Assume not NULL. Maps to assumeNotNull(x).',
)
def _build_assume_not_null(x, alias=None):
    from .functions import Function

    return Function('assumeNotNull', x, alias=alias)


@register_function(
    name='null_if',
    clickhouse_name='nullIf',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['nullIf'],
    doc='Return NULL if equal. Maps to nullIf(x, y).',
)
def _build_null_if(x, y, alias=None):
    from .functions import Function

    return Function('nullIf', x, y, alias=alias)


# --- Unix timestamp 64-bit variants ---
@register_function(
    name='to_unix_timestamp64_second',
    clickhouse_name='toUnixTimestamp64Second',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toUnixTimestamp64Second'],
    doc='Convert to Unix timestamp 64 (seconds). Maps to toUnixTimestamp64Second(dt).',
)
def _build_to_unix_timestamp64_second(dt, alias=None):
    from .functions import Function

    return Function('toUnixTimestamp64Second', dt, alias=alias)


@register_function(
    name='to_unix_timestamp64_milli',
    clickhouse_name='toUnixTimestamp64Milli',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toUnixTimestamp64Milli'],
    doc='Convert to Unix timestamp 64 (milliseconds). Maps to toUnixTimestamp64Milli(dt).',
)
def _build_to_unix_timestamp64_milli(dt, alias=None):
    from .functions import Function

    return Function('toUnixTimestamp64Milli', dt, alias=alias)


@register_function(
    name='to_unix_timestamp64_micro',
    clickhouse_name='toUnixTimestamp64Micro',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toUnixTimestamp64Micro'],
    doc='Convert to Unix timestamp 64 (microseconds). Maps to toUnixTimestamp64Micro(dt).',
)
def _build_to_unix_timestamp64_micro(dt, alias=None):
    from .functions import Function

    return Function('toUnixTimestamp64Micro', dt, alias=alias)


@register_function(
    name='to_unix_timestamp64_nano',
    clickhouse_name='toUnixTimestamp64Nano',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toUnixTimestamp64Nano'],
    doc='Convert to Unix timestamp 64 (nanoseconds). Maps to toUnixTimestamp64Nano(dt).',
)
def _build_to_unix_timestamp64_nano(dt, alias=None):
    from .functions import Function

    return Function('toUnixTimestamp64Nano', dt, alias=alias)


@register_function(
    name='from_unix_timestamp64_second',
    clickhouse_name='fromUnixTimestamp64Second',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['fromUnixTimestamp64Second'],
    doc='Convert from Unix timestamp 64 (seconds). Maps to fromUnixTimestamp64Second(ts).',
)
def _build_from_unix_timestamp64_second(ts, alias=None):
    from .functions import Function

    return Function('fromUnixTimestamp64Second', ts, alias=alias)


@register_function(
    name='from_unix_timestamp64_milli',
    clickhouse_name='fromUnixTimestamp64Milli',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['fromUnixTimestamp64Milli'],
    doc='Convert from Unix timestamp 64 (milliseconds). Maps to fromUnixTimestamp64Milli(ts).',
)
def _build_from_unix_timestamp64_milli(ts, alias=None):
    from .functions import Function

    return Function('fromUnixTimestamp64Milli', ts, alias=alias)


@register_function(
    name='from_unix_timestamp64_micro',
    clickhouse_name='fromUnixTimestamp64Micro',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['fromUnixTimestamp64Micro'],
    doc='Convert from Unix timestamp 64 (microseconds). Maps to fromUnixTimestamp64Micro(ts).',
)
def _build_from_unix_timestamp64_micro(ts, alias=None):
    from .functions import Function

    return Function('fromUnixTimestamp64Micro', ts, alias=alias)


@register_function(
    name='from_unix_timestamp64_nano',
    clickhouse_name='fromUnixTimestamp64Nano',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['fromUnixTimestamp64Nano'],
    doc='Convert from Unix timestamp 64 (nanoseconds). Maps to fromUnixTimestamp64Nano(ts).',
)
def _build_from_unix_timestamp64_nano(ts, alias=None):
    from .functions import Function

    return Function('fromUnixTimestamp64Nano', ts, alias=alias)


# --- UUID conversions ---
@register_function(
    name='to_uuid',
    clickhouse_name='toUUID',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.UUID,
    aliases=['toUUID'],
    doc='Convert to UUID. Maps to toUUID(s).',
)
def _build_to_uuid(s, alias=None):
    from .functions import Function

    return Function('toUUID', s, alias=alias)


@register_function(
    name='to_uuid_or_zero',
    clickhouse_name='toUUIDOrZero',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.UUID,
    aliases=['toUUIDOrZero'],
    doc='Convert to UUID or zero. Maps to toUUIDOrZero(s).',
)
def _build_to_uuid_or_zero(s, alias=None):
    from .functions import Function

    return Function('toUUIDOrZero', s, alias=alias)


# --- toBool ---
@register_function(
    name='to_bool',
    clickhouse_name='toBool',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['toBool'],
    doc='Convert to Bool. Maps to toBool(x).',
)
def _build_to_bool(x, alias=None):
    from .functions import Function

    return Function('toBool', x, alias=alias)


# --- formatRow ---
@register_function(
    name='format_row',
    clickhouse_name='formatRow',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['formatRow'],
    doc='Format row in given format. Maps to formatRow(format, x, ...).',
)
def _build_format_row(format_name: str, *args, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('formatRow', Literal(format_name), *args, alias=alias)


@register_function(
    name='format_row_no_newline',
    clickhouse_name='formatRowNoNewline',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.TYPE_CONVERSION,
    aliases=['formatRowNoNewline'],
    doc='Format row without newline. Maps to formatRowNoNewline(format, x, ...).',
)
def _build_format_row_no_newline(format_name: str, *args, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('formatRowNoNewline', Literal(format_name), *args, alias=alias)


# --- toTime ---
@register_function(
    name='to_time64',
    clickhouse_name='toTime64',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toTime64'],
    doc='Convert to Time64. Maps to toTime64(dt).',
)
def _build_to_time64(dt, alias=None):
    from .functions import Function

    return Function('toTime64', dt, alias=alias)


# =============================================================================
# INTERVAL FUNCTIONS (Additional)
# =============================================================================


@register_function(
    name='to_interval_millisecond',
    clickhouse_name='toIntervalMillisecond',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toIntervalMillisecond'],
    doc='Create interval in milliseconds. Maps to toIntervalMillisecond(n).',
)
def _build_to_interval_millisecond(n, alias=None):
    from .functions import Function

    return Function('toIntervalMillisecond', n, alias=alias)


@register_function(
    name='to_interval_microsecond',
    clickhouse_name='toIntervalMicrosecond',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toIntervalMicrosecond'],
    doc='Create interval in microseconds. Maps to toIntervalMicrosecond(n).',
)
def _build_to_interval_microsecond(n, alias=None):
    from .functions import Function

    return Function('toIntervalMicrosecond', n, alias=alias)


@register_function(
    name='to_interval_nanosecond',
    clickhouse_name='toIntervalNanosecond',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.DATETIME,
    aliases=['toIntervalNanosecond'],
    doc='Create interval in nanoseconds. Maps to toIntervalNanosecond(n).',
)
def _build_to_interval_nanosecond(n, alias=None):
    from .functions import Function

    return Function('toIntervalNanosecond', n, alias=alias)


# =============================================================================
# ROUNDING FUNCTIONS (Data Science Core)
# =============================================================================


@register_function(
    name='floor_func',
    clickhouse_name='floor',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['floor'],
    doc='Round down to nearest integer. Maps to floor(x).',
)
def _build_floor_func(x, n: int = None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if n is not None:
        return Function('floor', x, Literal(n), alias=alias)
    return Function('floor', x, alias=alias)


@register_function(
    name='ceil_func',
    clickhouse_name='ceil',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['ceil', 'ceiling'],
    doc='Round up to nearest integer. Maps to ceil(x).',
)
def _build_ceil_func(x, n: int = None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if n is not None:
        return Function('ceil', x, Literal(n), alias=alias)
    return Function('ceil', x, alias=alias)


@register_function(
    name='round_func',
    clickhouse_name='round',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['round'],
    doc='Round to nearest integer. Maps to round(x, n).',
)
def _build_round_func(x, n: int = 0, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('round', x, Literal(n), alias=alias)


@register_function(
    name='trunc_func',
    clickhouse_name='trunc',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['trunc', 'truncate'],
    doc='Truncate to integer. Maps to trunc(x).',
)
def _build_trunc_func(x, n: int = None, alias=None):
    from .functions import Function
    from .expressions import Literal

    if n is not None:
        return Function('trunc', x, Literal(n), alias=alias)
    return Function('trunc', x, alias=alias)


@register_function(
    name='round_bankers',
    clickhouse_name='roundBankers',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['roundBankers'],
    doc="Banker's rounding. Maps to roundBankers(x, n).",
)
def _build_round_bankers(x, n: int = 0, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('roundBankers', x, Literal(n), alias=alias)


@register_function(
    name='round_down',
    clickhouse_name='roundDown',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['roundDown'],
    doc='Round down to array element. Maps to roundDown(x, arr).',
)
def _build_round_down(x, arr, alias=None):
    from .functions import Function

    return Function('roundDown', x, arr, alias=alias)


@register_function(
    name='round_to_exp2',
    clickhouse_name='roundToExp2',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['roundToExp2'],
    doc='Round down to power of 2. Maps to roundToExp2(x).',
)
def _build_round_to_exp2(x, alias=None):
    from .functions import Function

    return Function('roundToExp2', x, alias=alias)


@register_function(
    name='round_age',
    clickhouse_name='roundAge',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['roundAge'],
    doc='Round age to standard age groups. Maps to roundAge(x).',
)
def _build_round_age(x, alias=None):
    from .functions import Function

    return Function('roundAge', x, alias=alias)


@register_function(
    name='round_duration',
    clickhouse_name='roundDuration',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.MATH,
    aliases=['roundDuration'],
    doc='Round duration to standard intervals. Maps to roundDuration(x).',
)
def _build_round_duration(x, alias=None):
    from .functions import Function

    return Function('roundDuration', x, alias=alias)


# =============================================================================
# STATISTICS FUNCTIONS (Statistical Analysis Core)
# =============================================================================


@register_function(
    name='var_pop',
    clickhouse_name='varPop',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['varPop', 'var_pop'],
    doc='Population variance. Maps to varPop(x).',
)
def _build_var_pop(x, alias=None):
    from .functions import Function

    return Function('varPop', x, alias=alias)


@register_function(
    name='var_samp',
    clickhouse_name='varSamp',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['varSamp', 'var_samp', 'variance'],
    doc='Sample variance. Maps to varSamp(x).',
)
def _build_var_samp(x, alias=None):
    from .functions import Function

    return Function('varSamp', x, alias=alias)


@register_function(
    name='stddev_pop',
    clickhouse_name='stddevPop',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['stddevPop', 'stddev_pop'],
    doc='Population standard deviation. Maps to stddevPop(x).',
)
def _build_stddev_pop(x, alias=None):
    from .functions import Function

    return Function('stddevPop', x, alias=alias)


@register_function(
    name='stddev_samp',
    clickhouse_name='stddevSamp',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['stddevSamp', 'stddev_samp', 'stddev'],
    doc='Sample standard deviation. Maps to stddevSamp(x).',
)
def _build_stddev_samp(x, alias=None):
    from .functions import Function

    return Function('stddevSamp', x, alias=alias)


@register_function(
    name='covar_pop',
    clickhouse_name='covarPop',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['covarPop', 'covar_pop'],
    doc='Population covariance. Maps to covarPop(x, y).',
)
def _build_covar_pop(x, y, alias=None):
    from .functions import Function

    return Function('covarPop', x, y, alias=alias)


@register_function(
    name='covar_samp',
    clickhouse_name='covarSamp',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['covarSamp', 'covar_samp', 'covariance'],
    doc='Sample covariance. Maps to covarSamp(x, y).',
)
def _build_covar_samp(x, y, alias=None):
    from .functions import Function

    return Function('covarSamp', x, y, alias=alias)


@register_function(
    name='corr_func',
    clickhouse_name='corr',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['corr', 'correlation'],
    doc='Pearson correlation. Maps to corr(x, y).',
)
def _build_corr_func(x, y, alias=None):
    from .functions import Function

    return Function('corr', x, y, alias=alias)


@register_function(
    name='skew_pop',
    clickhouse_name='skewPop',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['skewPop', 'skew_pop'],
    doc='Population skewness. Maps to skewPop(x).',
)
def _build_skew_pop(x, alias=None):
    from .functions import Function

    return Function('skewPop', x, alias=alias)


@register_function(
    name='skew_samp',
    clickhouse_name='skewSamp',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['skewSamp', 'skew_samp', 'skewness'],
    doc='Sample skewness. Maps to skewSamp(x).',
)
def _build_skew_samp(x, alias=None):
    from .functions import Function

    return Function('skewSamp', x, alias=alias)


@register_function(
    name='kurt_pop',
    clickhouse_name='kurtPop',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['kurtPop', 'kurt_pop'],
    doc='Population kurtosis. Maps to kurtPop(x).',
)
def _build_kurt_pop(x, alias=None):
    from .functions import Function

    return Function('kurtPop', x, alias=alias)


@register_function(
    name='kurt_samp',
    clickhouse_name='kurtSamp',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['kurtSamp', 'kurt_samp', 'kurtosis'],
    doc='Sample kurtosis. Maps to kurtSamp(x).',
)
def _build_kurt_samp(x, alias=None):
    from .functions import Function

    return Function('kurtSamp', x, alias=alias)


@register_function(
    name='entropy_func',
    clickhouse_name='entropy',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['entropy'],
    doc='Shannon entropy. Maps to entropy(x).',
)
def _build_entropy_func(x, alias=None):
    from .functions import Function

    return Function('entropy', x, alias=alias)


@register_function(
    name='simple_linear_regression',
    clickhouse_name='simpleLinearRegression',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['simpleLinearRegression'],
    doc='Simple linear regression. Maps to simpleLinearRegression(x, y).',
)
def _build_simple_linear_regression(x, y, alias=None):
    from .functions import Function

    return Function('simpleLinearRegression', x, y, alias=alias)


# =============================================================================
# QUANTILE FUNCTIONS (Quantile Statistics)
# =============================================================================


@register_function(
    name='quantile_func',
    clickhouse_name='quantile',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['quantile'],
    doc='Quantile. Maps to quantile(level)(x).',
)
def _build_quantile_func(x, level: float = 0.5, alias=None):
    from .functions import Function

    return Function(f'quantile({level})', x, alias=alias)


@register_function(
    name='quantile_exact',
    clickhouse_name='quantileExact',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['quantileExact'],
    doc='Exact quantile. Maps to quantileExact(level)(x).',
)
def _build_quantile_exact(x, level: float = 0.5, alias=None):
    from .functions import Function

    return Function(f'quantileExact({level})', x, alias=alias)


@register_function(
    name='quantile_timing',
    clickhouse_name='quantileTiming',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['quantileTiming'],
    doc='Quantile for timing data. Maps to quantileTiming(level)(x).',
)
def _build_quantile_timing(x, level: float = 0.5, alias=None):
    from .functions import Function

    return Function(f'quantileTiming({level})', x, alias=alias)


@register_function(
    name='quantile_deterministic',
    clickhouse_name='quantileDeterministic',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['quantileDeterministic'],
    doc='Deterministic quantile. Maps to quantileDeterministic(level)(x, determinator).',
)
def _build_quantile_deterministic(x, determinator, level: float = 0.5, alias=None):
    from .functions import Function

    return Function(f'quantileDeterministic({level})', x, determinator, alias=alias)


@register_function(
    name='quantiles_func',
    clickhouse_name='quantiles',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['quantiles'],
    doc='Multiple quantiles. Maps to quantiles(l1, l2, ...)(x).',
)
def _build_quantiles_func(x, *levels, alias=None):
    from .functions import Function

    levels_str = ', '.join(str(l) for l in levels) if levels else '0.25, 0.5, 0.75'
    return Function(f'quantiles({levels_str})', x, alias=alias)


@register_function(
    name='median_func',
    clickhouse_name='median',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['median'],
    doc='Median (quantile 0.5). Maps to median(x).',
)
def _build_median_func(x, alias=None):
    from .functions import Function

    return Function('median', x, alias=alias)


@register_function(
    name='median_exact',
    clickhouse_name='medianExact',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['medianExact'],
    doc='Exact median. Maps to medianExact(x).',
)
def _build_median_exact(x, alias=None):
    from .functions import Function

    return Function('medianExact', x, alias=alias)


# =============================================================================
# STRING SPLITTING FUNCTIONS (String Splitting - Common in ETL)
# =============================================================================


@register_function(
    name='split_by_char',
    clickhouse_name='splitByChar',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['splitByChar'],
    doc='Split string by character. Maps to splitByChar(sep, s).',
)
def _build_split_by_char(sep: str, s, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('splitByChar', Literal(sep), s, alias=alias)


@register_function(
    name='split_by_string',
    clickhouse_name='splitByString',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['splitByString'],
    doc='Split string by substring. Maps to splitByString(sep, s).',
)
def _build_split_by_string(sep: str, s, alias=None):
    from .functions import Function
    from .expressions import Literal

    # Wrap with ifNull to handle Nullable columns - ClickHouse doesn't support Nullable(Array)
    safe_s = Function('ifNull', s, Literal(''))
    return Function('splitByString', Literal(sep), safe_s, alias=alias)


@register_function(
    name='split_by_regexp',
    clickhouse_name='splitByRegexp',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['splitByRegexp'],
    doc='Split string by regexp. Maps to splitByRegexp(pattern, s).',
)
def _build_split_by_regexp(pattern: str, s, alias=None):
    from .functions import Function
    from .expressions import Literal

    # Wrap with ifNull to handle Nullable columns - ClickHouse doesn't support Nullable(Array)
    safe_s = Function('ifNull', s, Literal(''))
    return Function('splitByRegexp', Literal(pattern), safe_s, alias=alias)


@register_function(
    name='split_by_whitespace',
    clickhouse_name='splitByWhitespace',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['splitByWhitespace'],
    doc='Split string by whitespace. Maps to splitByWhitespace(s).',
)
def _build_split_by_whitespace(s, alias=None):
    from .functions import Function
    from .expressions import Literal

    # Wrap with ifNull to handle Nullable columns - ClickHouse doesn't support Nullable(Array)
    safe_s = Function('ifNull', s, Literal(''))
    return Function('splitByWhitespace', safe_s, alias=alias)


@register_function(
    name='split_by_non_alpha',
    clickhouse_name='splitByNonAlpha',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['splitByNonAlpha'],
    doc='Split string by non-alpha characters. Maps to splitByNonAlpha(s).',
)
def _build_split_by_non_alpha(s, alias=None):
    from .functions import Function
    from .expressions import Literal

    # Wrap with ifNull to handle Nullable columns - ClickHouse doesn't support Nullable(Array)
    safe_s = Function('ifNull', s, Literal(''))
    return Function('splitByNonAlpha', safe_s, alias=alias)


@register_function(
    name='alpha_tokens',
    clickhouse_name='alphaTokens',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['alphaTokens'],
    doc='Extract alphabetic tokens. Maps to alphaTokens(s).',
)
def _build_alpha_tokens(s, alias=None):
    from .functions import Function

    return Function('alphaTokens', s, alias=alias)


@register_function(
    name='array_string_concat',
    clickhouse_name='arrayStringConcat',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['arrayStringConcat'],
    doc='Concatenate array elements. Maps to arrayStringConcat(arr, sep).',
)
def _build_array_string_concat(arr, sep: str = '', alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('arrayStringConcat', arr, Literal(sep), alias=alias)


# =============================================================================
# STRING SEARCH/MATCH FUNCTIONS (Pattern Matching)
# =============================================================================


@register_function(
    name='like_func',
    clickhouse_name='like',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['like'],
    doc='SQL LIKE pattern match. Maps to like(s, pattern).',
)
def _build_like_func(s, pattern: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('like', s, Literal(pattern), alias=alias)


@register_function(
    name='not_like',
    clickhouse_name='notLike',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['notLike'],
    doc='SQL NOT LIKE pattern match. Maps to notLike(s, pattern).',
)
def _build_not_like(s, pattern: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('notLike', s, Literal(pattern), alias=alias)


@register_function(
    name='ilike_func',
    clickhouse_name='ilike',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['ilike'],
    doc='Case-insensitive LIKE. Maps to ilike(s, pattern).',
)
def _build_ilike_func(s, pattern: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('ilike', s, Literal(pattern), alias=alias)


@register_function(
    name='not_ilike',
    clickhouse_name='notILike',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['notILike'],
    doc='Case-insensitive NOT LIKE. Maps to notILike(s, pattern).',
)
def _build_not_ilike(s, pattern: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('notILike', s, Literal(pattern), alias=alias)


@register_function(
    name='match_func',
    clickhouse_name='match',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['match'],
    doc='Regexp match. Maps to match(s, pattern).',
)
def _build_match_func(s, pattern: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('match', s, Literal(pattern), alias=alias)


@register_function(
    name='extract_func',
    clickhouse_name='extract',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['extract'],
    doc='Extract using regexp. Maps to extract(s, pattern).',
)
def _build_extract_func(s, pattern: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('extract', s, Literal(pattern), alias=alias)


@register_function(
    name='extract_all',
    clickhouse_name='extractAll',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['extractAll'],
    doc='Extract all matches. Maps to extractAll(s, pattern).',
)
def _build_extract_all(s, pattern: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('extractAll', s, Literal(pattern), alias=alias)


@register_function(
    name='extract_all_groups',
    clickhouse_name='extractAllGroups',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['extractAllGroups'],
    doc='Extract all regexp groups. Maps to extractAllGroups(s, pattern).',
)
def _build_extract_all_groups(s, pattern: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('extractAllGroups', s, Literal(pattern), alias=alias)


@register_function(
    name='position_utf8',
    clickhouse_name='positionUTF8',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['positionUTF8'],
    doc='Position in UTF-8 string. Maps to positionUTF8(haystack, needle).',
)
def _build_position_utf8(haystack, needle, alias=None):
    from .functions import Function

    return Function('positionUTF8', haystack, needle, alias=alias)


@register_function(
    name='position_case_insensitive',
    clickhouse_name='positionCaseInsensitive',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['positionCaseInsensitive'],
    doc='Case-insensitive position. Maps to positionCaseInsensitive(haystack, needle).',
)
def _build_position_case_insensitive(haystack, needle, alias=None):
    from .functions import Function

    return Function('positionCaseInsensitive', haystack, needle, alias=alias)


@register_function(
    name='multi_search_any',
    clickhouse_name='multiSearchAny',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['multiSearchAny'],
    doc='Check if any needle found. Maps to multiSearchAny(haystack, [needles]).',
)
def _build_multi_search_any(haystack, needles, alias=None):
    from .functions import Function

    return Function('multiSearchAny', haystack, needles, alias=alias)


@register_function(
    name='multi_search_first_index',
    clickhouse_name='multiSearchFirstIndex',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['multiSearchFirstIndex'],
    doc='Index of first found needle. Maps to multiSearchFirstIndex(haystack, [needles]).',
)
def _build_multi_search_first_index(haystack, needles, alias=None):
    from .functions import Function

    return Function('multiSearchFirstIndex', haystack, needles, alias=alias)


@register_function(
    name='multi_search_first_position',
    clickhouse_name='multiSearchFirstPosition',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['multiSearchFirstPosition'],
    doc='Position of first found needle. Maps to multiSearchFirstPosition(haystack, [needles]).',
)
def _build_multi_search_first_position(haystack, needles, alias=None):
    from .functions import Function

    return Function('multiSearchFirstPosition', haystack, needles, alias=alias)


@register_function(
    name='multi_search_all_positions',
    clickhouse_name='multiSearchAllPositions',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['multiSearchAllPositions'],
    doc='All positions of all needles. Maps to multiSearchAllPositions(haystack, [needles]).',
)
def _build_multi_search_all_positions(haystack, needles, alias=None):
    from .functions import Function

    return Function('multiSearchAllPositions', haystack, needles, alias=alias)


# =============================================================================
# TOP-K & ARG MIN/MAX (Ranking Analysis)
# =============================================================================


@register_function(
    name='any_heavy',
    clickhouse_name='anyHeavy',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['anyHeavy'],
    doc='Heavy hitter (frequent value). Maps to anyHeavy(x).',
)
def _build_any_heavy(x, alias=None):
    from .functions import Function

    return Function('anyHeavy', x, alias=alias)


@register_function(
    name='any_last',
    clickhouse_name='anyLast',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['anyLast'],
    doc='Last encountered value. Maps to anyLast(x).',
)
def _build_any_last(x, alias=None):
    from .functions import Function

    return Function('anyLast', x, alias=alias)


@register_function(
    name='arg_min',
    clickhouse_name='argMin',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['argMin'],
    doc='Value at minimum of arg. Maps to argMin(val, arg).',
)
def _build_arg_min(val, arg, alias=None):
    from .functions import Function

    return Function('argMin', val, arg, alias=alias)


@register_function(
    name='arg_max',
    clickhouse_name='argMax',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['argMax'],
    doc='Value at maximum of arg. Maps to argMax(val, arg).',
)
def _build_arg_max(val, arg, alias=None):
    from .functions import Function

    return Function('argMax', val, arg, alias=alias)


@register_function(
    name='top_k',
    clickhouse_name='topK',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['topK'],
    doc='Top K frequent values. Maps to topK(k)(x).',
)
def _build_top_k(x, k: int = 10, alias=None):
    from .functions import Function

    return Function(f'topK({k})', x, alias=alias)


@register_function(
    name='top_k_weighted',
    clickhouse_name='topKWeighted',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['topKWeighted'],
    doc='Top K weighted values. Maps to topKWeighted(k)(x, weight).',
)
def _build_top_k_weighted(x, weight, k: int = 10, alias=None):
    from .functions import Function

    return Function(f'topKWeighted({k})', x, weight, alias=alias)


# =============================================================================
# BITMAP FUNCTIONS (Efficient Set Operations)
# =============================================================================


@register_function(
    name='bitmap_build',
    clickhouse_name='bitmapBuild',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['bitmapBuild'],
    doc='Build bitmap from array. Maps to bitmapBuild(arr).',
)
def _build_bitmap_build(arr, alias=None):
    from .functions import Function

    return Function('bitmapBuild', arr, alias=alias)


@register_function(
    name='bitmap_cardinality',
    clickhouse_name='bitmapCardinality',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['bitmapCardinality'],
    doc='Bitmap cardinality. Maps to bitmapCardinality(bitmap).',
)
def _build_bitmap_cardinality(bitmap, alias=None):
    from .functions import Function

    return Function('bitmapCardinality', bitmap, alias=alias)


@register_function(
    name='bitmap_to_array',
    clickhouse_name='bitmapToArray',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['bitmapToArray'],
    doc='Convert bitmap to array. Maps to bitmapToArray(bitmap).',
)
def _build_bitmap_to_array(bitmap, alias=None):
    from .functions import Function

    return Function('bitmapToArray', bitmap, alias=alias)


@register_function(
    name='bitmap_contains',
    clickhouse_name='bitmapContains',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['bitmapContains'],
    doc='Check if bitmap contains value. Maps to bitmapContains(bitmap, x).',
)
def _build_bitmap_contains(bitmap, x, alias=None):
    from .functions import Function

    return Function('bitmapContains', bitmap, x, alias=alias)


@register_function(
    name='bitmap_and',
    clickhouse_name='bitmapAnd',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['bitmapAnd'],
    doc='Bitmap AND. Maps to bitmapAnd(b1, b2).',
)
def _build_bitmap_and(b1, b2, alias=None):
    from .functions import Function

    return Function('bitmapAnd', b1, b2, alias=alias)


@register_function(
    name='bitmap_or',
    clickhouse_name='bitmapOr',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['bitmapOr'],
    doc='Bitmap OR. Maps to bitmapOr(b1, b2).',
)
def _build_bitmap_or(b1, b2, alias=None):
    from .functions import Function

    return Function('bitmapOr', b1, b2, alias=alias)


@register_function(
    name='bitmap_xor',
    clickhouse_name='bitmapXor',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['bitmapXor'],
    doc='Bitmap XOR. Maps to bitmapXor(b1, b2).',
)
def _build_bitmap_xor(b1, b2, alias=None):
    from .functions import Function

    return Function('bitmapXor', b1, b2, alias=alias)


@register_function(
    name='bitmap_andnot',
    clickhouse_name='bitmapAndnot',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['bitmapAndnot'],
    doc='Bitmap AND NOT. Maps to bitmapAndnot(b1, b2).',
)
def _build_bitmap_andnot(b1, b2, alias=None):
    from .functions import Function

    return Function('bitmapAndnot', b1, b2, alias=alias)


@register_function(
    name='bitmap_has_any',
    clickhouse_name='bitmapHasAny',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['bitmapHasAny'],
    doc='Check if bitmaps have any common element. Maps to bitmapHasAny(b1, b2).',
)
def _build_bitmap_has_any(b1, b2, alias=None):
    from .functions import Function

    return Function('bitmapHasAny', b1, b2, alias=alias)


@register_function(
    name='bitmap_has_all',
    clickhouse_name='bitmapHasAll',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.ARRAY,
    aliases=['bitmapHasAll'],
    doc='Check if b1 contains all elements of b2. Maps to bitmapHasAll(b1, b2).',
)
def _build_bitmap_has_all(b1, b2, alias=None):
    from .functions import Function

    return Function('bitmapHasAll', b1, b2, alias=alias)


# =============================================================================
# UTILITY VALIDATION FUNCTIONS
# =============================================================================


@register_function(
    name='is_valid_json',
    clickhouse_name='isValidJSON',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['isValidJSON'],
    doc='Check if string is valid JSON. Maps to isValidJSON(s).',
)
def _build_is_valid_json(s, alias=None):
    from .functions import Function

    return Function('isValidJSON', s, alias=alias)


@register_function(
    name='is_valid_utf8',
    clickhouse_name='isValidUTF8',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['isValidUTF8'],
    doc='Check if string is valid UTF-8. Maps to isValidUTF8(s).',
)
def _build_is_valid_utf8(s, alias=None):
    from .functions import Function

    return Function('isValidUTF8', s, alias=alias)


@register_function(
    name='to_valid_utf8',
    clickhouse_name='toValidUTF8',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['toValidUTF8'],
    doc='Convert to valid UTF-8. Maps to toValidUTF8(s).',
)
def _build_to_valid_utf8(s, alias=None):
    from .functions import Function

    return Function('toValidUTF8', s, alias=alias)


@register_function(
    name='soundex',
    clickhouse_name='soundex',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['soundex'],
    doc='Soundex code. Maps to soundex(s).',
)
def _build_soundex(s, alias=None):
    from .functions import Function

    return Function('soundex', s, alias=alias)


@register_function(
    name='normalize_query',
    clickhouse_name='normalizeQuery',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['normalizeQuery'],
    doc='Normalize SQL query. Maps to normalizeQuery(s).',
)
def _build_normalize_query(s, alias=None):
    from .functions import Function

    return Function('normalizeQuery', s, alias=alias)


@register_function(
    name='normalized_query_hash',
    clickhouse_name='normalizedQueryHash',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.HASH,
    aliases=['normalizedQueryHash'],
    doc='Hash of normalized query. Maps to normalizedQueryHash(s).',
)
def _build_normalized_query_hash(s, alias=None):
    from .functions import Function

    return Function('normalizedQueryHash', s, alias=alias)


# =============================================================================
# NLP/TEXT FUNCTIONS
# =============================================================================


@register_function(
    name='ngram_min_hash',
    clickhouse_name='ngramMinHash',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['ngramMinHash'],
    doc='N-gram MinHash. Maps to ngramMinHash(s).',
)
def _build_ngram_min_hash(s, alias=None):
    from .functions import Function

    return Function('ngramMinHash', s, alias=alias)


@register_function(
    name='word_shingle_min_hash',
    clickhouse_name='wordShingleMinHash',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['wordShingleMinHash'],
    doc='Word shingle MinHash. Maps to wordShingleMinHash(s).',
)
def _build_word_shingle_min_hash(s, alias=None):
    from .functions import Function

    return Function('wordShingleMinHash', s, alias=alias)


@register_function(
    name='word_shingle_sim_hash',
    clickhouse_name='wordShingleSimHash',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['wordShingleSimHash'],
    doc='Word shingle SimHash. Maps to wordShingleSimHash(s).',
)
def _build_word_shingle_sim_hash(s, alias=None):
    from .functions import Function

    return Function('wordShingleSimHash', s, alias=alias)


# =============================================================================
# MACHINE LEARNING FUNCTIONS
# =============================================================================


@register_function(
    name='stochastic_linear_regression',
    clickhouse_name='stochasticLinearRegression',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['stochasticLinearRegression'],
    doc='Stochastic gradient descent linear regression. Maps to stochasticLinearRegression(...).',
)
def _build_stochastic_linear_regression(*args, alias=None):
    from .functions import Function

    return Function('stochasticLinearRegression', *args, alias=alias)


@register_function(
    name='stochastic_logistic_regression',
    clickhouse_name='stochasticLogisticRegression',
    func_type=FunctionType.AGGREGATE,
    category=FunctionCategory.AGGREGATE,
    aliases=['stochasticLogisticRegression'],
    doc='Stochastic gradient descent logistic regression. Maps to stochasticLogisticRegression(...).',
)
def _build_stochastic_logistic_regression(*args, alias=None):
    from .functions import Function

    return Function('stochasticLogisticRegression', *args, alias=alias)


@register_function(
    name='min_sample_size_continuous',
    clickhouse_name='minSampleSizeContinuous',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.AGGREGATE,
    aliases=['minSampleSizeContinuous'],
    doc='Minimum sample size for continuous metric. Maps to minSampleSizeContinuous(baseline, mde, power, alpha).',
)
def _build_min_sample_size_continuous(baseline, mde, power, alpha, alias=None):
    from .functions import Function

    return Function('minSampleSizeContinuous', baseline, mde, power, alpha, alias=alias)


@register_function(
    name='min_sample_size_conversion',
    clickhouse_name='minSampleSizeConversion',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.AGGREGATE,
    aliases=['minSampleSizeConversion'],
    doc='Minimum sample size for conversion. Maps to minSampleSizeConversion(baseline, mde, power, alpha).',
)
def _build_min_sample_size_conversion(baseline, mde, power, alpha, alias=None):
    from .functions import Function

    return Function('minSampleSizeConversion', baseline, mde, power, alpha, alias=alias)


# Ensure functions are registered when this module is imported
ensure_functions_registered()
