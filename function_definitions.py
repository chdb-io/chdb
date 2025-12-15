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
    aliases=['arrayElement', 'get'],
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
    aliases=['get'],
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


# Ensure functions are registered when this module is imported
ensure_functions_registered()
