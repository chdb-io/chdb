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


@register_function(
    name='replace',
    clickhouse_name='replace',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['replaceAll'],
    doc='Replace occurrences. Maps to replace(s, from, to).',
)
def _build_replace(expr, pattern: str, replacement: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('replace', expr, Literal(pattern), Literal(replacement), alias=alias)


@register_function(
    name='trim',
    clickhouse_name='trim',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['strip'],
    doc='Trim whitespace from both sides. Maps to trim(x).',
)
def _build_trim(expr, alias=None):
    from .functions import Function

    return Function('trim', expr, alias=alias)


@register_function(
    name='ltrim',
    clickhouse_name='trimLeft',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['trimLeft', 'lstrip'],
    doc='Trim whitespace from left. Maps to trimLeft(x).',
)
def _build_ltrim(expr, alias=None):
    from .functions import Function

    return Function('trimLeft', expr, alias=alias)


@register_function(
    name='rtrim',
    clickhouse_name='trimRight',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['trimRight', 'rstrip'],
    doc='Trim whitespace from right. Maps to trimRight(x).',
)
def _build_rtrim(expr, alias=None):
    from .functions import Function

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
def _build_starts_with(expr, prefix: str, alias=None):
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
def _build_ends_with(expr, suffix: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('endsWith', expr, Literal(suffix), alias=alias)


@register_function(
    name='contains',
    clickhouse_name='position',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['has_substring'],
    doc='Check if string contains substring. Returns position (>0 if found). Maps to position(s, needle).',
)
def _build_contains(expr, needle: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('position', expr, Literal(needle), alias=alias)


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
def _build_pad(expr, length: int, fill: str = ' ', alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('leftPad', expr, Literal(length), Literal(fill), alias=alias)


@register_function(
    name='rpad',
    clickhouse_name='rightPad',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    aliases=['rightPad'],
    doc='Pad string on right to length with fill char. Maps to rightPad(s, length, fill).',
)
def _build_rpad(expr, length: int, fill: str = ' ', alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('rightPad', expr, Literal(length), Literal(fill), alias=alias)


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
    doc='Split string by separator. Maps to splitByString(sep, s).',
)
def _build_split(expr, sep: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('splitByString', Literal(sep), expr, alias=alias)


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
    aliases=['stddevPop', 'std', 'stddev_pop'],
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
    aliases=['stddevSamp'],
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
    doc='Check if NULL. Maps to isNull(x).',
)
def _build_isna(expr, alias=None):
    from .functions import Function

    return Function('isNull', expr, alias=alias)


@register_function(
    name='notna',
    clickhouse_name='isNotNull',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.CONDITIONAL,
    aliases=['notnull'],
    doc='Check if not NULL. Maps to isNotNull(x).',
)
def _build_notna(expr, alias=None):
    from .functions import Function

    return Function('isNotNull', expr, alias=alias)


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


@register_function(
    name='json_extract_string',
    clickhouse_name='JSONExtractString',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractString'],
    doc='Extract string from JSON. Maps to JSONExtractString(json, path).',
)
def _build_json_extract_string(json, path: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('JSONExtractString', json, Literal(path), alias=alias)


@register_function(
    name='json_extract_int',
    clickhouse_name='JSONExtractInt',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractInt'],
    doc='Extract integer from JSON. Maps to JSONExtractInt(json, path).',
)
def _build_json_extract_int(json, path: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('JSONExtractInt', json, Literal(path), alias=alias)


@register_function(
    name='json_extract_float',
    clickhouse_name='JSONExtractFloat',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractFloat'],
    doc='Extract float from JSON. Maps to JSONExtractFloat(json, path).',
)
def _build_json_extract_float(json, path: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('JSONExtractFloat', json, Literal(path), alias=alias)


@register_function(
    name='json_extract_bool',
    clickhouse_name='JSONExtractBool',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.JSON,
    aliases=['JSONExtractBool'],
    doc='Extract boolean from JSON. Maps to JSONExtractBool(json, path).',
)
def _build_json_extract_bool(json, path: str, alias=None):
    from .functions import Function
    from .expressions import Literal

    return Function('JSONExtractBool', json, Literal(path), alias=alias)


# Ensure functions are registered when this module is imported
ensure_functions_registered()
