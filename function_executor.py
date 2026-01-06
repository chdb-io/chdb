"""
Function Executor Configuration

This module provides a unified configuration system for controlling how
functions are executed - whether via chDB SQL engine or Pandas.

The default is to prefer chDB for better performance, but users can
configure specific functions to use Pandas when needed.
"""

from typing import Dict, Callable, Optional, Any, Set
from enum import Enum
import pandas as pd


class ExecutionEngine(Enum):
    """Execution engine for function calls."""

    CHDB = 'chdb'  # Execute via chDB SQL engine (default, preferred)
    PANDAS = 'pandas'  # Execute via Pandas operations
    AUTO = 'auto'  # Automatically choose based on context


class FunctionExecutorConfig:
    """
    Configuration for function execution strategy.

    Controls whether overlapping functions (those available in both chDB and Pandas)
    should be executed via chDB SQL engine or Pandas.

    Default behavior: prefer chDB for better performance.

    Example:
        >>> from datastore import function_config
        >>>
        >>> # Use Pandas for specific functions
        >>> function_config.use_pandas('upper', 'lower')
        >>>
        >>> # Use chDB for specific functions (default)
        >>> function_config.use_chdb('length', 'substring')
        >>>
        >>> # Check current setting
        >>> function_config.get_engine('upper')  # Returns ExecutionEngine.PANDAS
        >>>
        >>> # Reset to defaults
        >>> function_config.reset()
    """

    # Functions that have both chDB and Pandas implementations
    OVERLAPPING_FUNCTIONS: Set[str] = {
        # String functions
        'upper',
        'lower',
        'length',
        'len',
        'substring',
        'substr',
        'replace',
        'trim',
        'ltrim',
        'rtrim',
        'strip',
        'lstrip',
        'rstrip',
        'concat',
        'startswith',
        'endswith',
        # 'contains' moved to PANDAS_ONLY_FUNCTIONS due to chDB NaN handling issues
        'split',
        'initcap',  # ClickHouse has initcap, not capitalize/title
        # Math functions
        'abs',
        'round',
        'floor',
        'ceil',
        'ceiling',
        'sqrt',
        'exp',
        'log',
        'log10',
        'log2',
        'log1p',
        'sin',
        'cos',
        'tan',
        'asin',
        'acos',
        'atan',
        'sinh',
        'cosh',
        'tanh',
        'asinh',
        'acosh',
        'atanh',
        'pow',
        'power',
        'mod',
        'sign',
        'trunc',
        'clip',
        'degrees',
        'radians',
        # Aggregate functions
        'sum',
        'avg',
        'mean',
        'min',
        'max',
        'count',
        'std',
        'stddev',
        'var',
        'variance',
        'median',
        'quantile',
        'first',
        'last',
        'nunique',
        'prod',
        'product',
        'skew',
        'kurt',
        'kurtosis',
        'sem',
        # Date/time functions
        'year',
        'month',
        'day',
        'hour',
        'minute',
        'second',
        'dayofweek',
        'weekday',
        'dayofyear',
        'weekofyear',
        'week',
        'quarter',
        'date',
        'time',
        'strftime',
        # Type conversion
        'tostring',
        'toint',
        'tofloat',
        'todate',
        'todatetime',
        'astype',
        # Conditional value replacement (CASE WHEN in SQL)
        'where',
        'mask',
        # CASE WHEN expression (ds.when().otherwise())
        'when',
        'case_when',
        # Null detection (uses toBool(isNull/isNotNull) in SQL for bool dtype)
        'isna',
        'isnull',
        'notna',
        'notnull',
    }

    # Pandas-only functions (no ClickHouse equivalent or require pandas-specific features)
    # NOTE: As of chDB 4.0.0b4, NaN/NULL handling is fixed at the SQL level (see GitHub issue #447).
    # Functions like isNull/isNotNull/ifNull now work correctly with NaN in SQL.
    # HOWEVER, the DataStore lazy execution pipeline still has issues extracting
    # function results correctly (returns original column instead of function result).
    # Until the execution pipeline is fixed, keep these functions using pandas.
    PANDAS_ONLY_FUNCTIONS: Set[str] = {
        # DateTime strftime - ClickHouse formatDateTime uses different format codes
        # Python %M = minute, ClickHouse %M = month name
        # Python %H = hour 24h, ClickHouse %H = hour 24h (same)
        # Using pandas ensures Python strftime format compatibility
        'strftime',
        # isocalendar - chDB timezone handling causes issues at day boundaries
        # e.g., 2024-12-31 23:59:59 may be interpreted differently
        'isocalendar',
        # String functions that need pandas for na parameter support
        'contains',  # pandas str.contains has na parameter (na=True/False), SQL propagates NULL
        # Cumulative functions (no simple SQL equivalent)
        'cumsum',
        'cummax',
        'cummin',
        'cumprod',
        # Shift and diff (no simple SQL equivalent)
        'shift',
        'diff',
        'pct_change',
        # Ranking (pandas-specific behavior)
        'rank',
        'nlargest',
        'nsmallest',
        # Value operations (pandas-specific)
        'unique',
        'value_counts',
        'duplicated',
        'drop_duplicates',
        # Missing value handling
        # NOTE: chDB 4.0.0b3 fixed NaN handling in SQL (see GitHub issue #447).
        # isNull/isNotNull/ifNull now work correctly in SQL.
        # isna/notna now use SQL via toBool(isNull/isNotNull) for bool dtype.
        'fillna',  # Keep pandas for method='ffill'/'bfill' support
        'dropna',
        'ffill',  # Forward fill - needs window functions
        'bfill',  # Backward fill - needs window functions
        'interpolate',  # Complex pandas-only
        # Type checking
        'isin',
        'between',
        # Apply and map
        'apply',
        'map',
        # Comparison
        'eq',
        'ne',
        'lt',
        'le',
        'gt',
        'ge',
        # Clipping
        'clip_lower',
        'clip_upper',
        # Binning
        'cut',
        'qcut',
        # String-only (no CH equivalent)
        'title',
        'capitalize',
        'swapcase',
        'casefold',  # Case conversion
        'isalpha',
        'isdigit',
        'isalnum',
        'isspace',
        'isupper',
        'islower',
        'istitle',
        'isnumeric',
        'isdecimal',  # String info
        'encode',
        'decode',
        'normalize',
        'translate',
        'wrap',
        'cat',
        'join',
        'extractall',
        'findall',
        'get',
        'rfind',
        'count_matches',
        'slice',
        'slice_replace',
        'center',
        'ljust',
        'rjust',
        'partition',
        'rpartition',
        'rsplit',
        'removeprefix',  # Python 3.9+ string method, no SQL equivalent
        'removesuffix',  # Python 3.9+ string method, no SQL equivalent
        # Pandas arithmetic methods (use operators in SQL: +, -, *, /)
        # These methods have different semantics (fill_value, axis, level params)
        'add',
        'sub',
        'mul',
        'div',
        'truediv',
        'floordiv',
        'mod',  # pandas mod method (not SQL % operator)
        'pow',  # pandas pow method (not SQL power function)
        'radd',
        'rsub',
        'rmul',
        'rdiv',
        'rtruediv',
        'rfloordiv',
        'rmod',
        'rpow',
        # Other pandas-specific methods
        'combine',
        'combine_first',
        'dot',
        # DateTime-only (no CH equivalent)
        'microsecond',
        'nanosecond',
        'days_in_month',
        'daysinmonth',
        'is_month_start',
        'is_month_end',
        'is_quarter_start',
        'is_quarter_end',
        'is_year_start',
        'is_year_end',
        'is_leap_year',
        'tz_localize',
        'tz_convert',
        'to_period',
        'to_pydatetime',
        'to_pytimedelta',
    }

    # Accessor-level pandas fallback conditions
    # Maps accessor.method -> condition dict
    # Condition format:
    #   {'param_name': specific_value} - fallback when param equals value
    #   {'param_name': '*'} - always fallback regardless of param value
    #   '*' - always uses pandas (no SQL implementation)
    #
    # This allows fine-grained control over when accessor methods use pandas
    # based on specific parameter combinations.
    #
    # Use function_config.needs_accessor_fallback('str.split', expand=True) to check
    ACCESSOR_PARAM_PANDAS_FALLBACK: Dict[str, Any] = {
        # String accessor methods
        'str.split': {'expand': True},  # expand=True returns DataFrame, needs pandas
        'str.rsplit': {'expand': True},  # expand=True returns DataFrame, needs pandas
        'str.extract': '*',  # regex extraction always needs pandas
        'str.extractall': '*',  # multi-match extraction always needs pandas
        'str.findall': '*',  # find all matches needs pandas
        'str.wrap': '*',  # text wrapping needs pandas
        'str.get': '*',  # character extraction needs pandas
        'str.join': '*',  # array join needs pandas (chDB array type issues)
        'str.cat': '*',  # string concatenation with others needs pandas
        'str.partition': '*',  # split around separator needs pandas
        'str.rpartition': '*',  # reverse split around separator needs pandas
        'str.normalize': '*',  # unicode normalization needs pandas
        'str.encode': '*',  # encoding needs pandas
        'str.decode': '*',  # decoding needs pandas
        'str.translate': '*',  # character translation needs pandas
        # DateTime accessor methods
        'dt.strftime': '*',  # format codes differ: Python %M=minute, chDB %M=month name
        # dt.floor, dt.ceil, dt.round now support common frequencies (h, min, s, D) via SQL
        # Unsupported frequencies fall back to pandas
        'dt.tz_localize': '*',  # timezone localization needs pandas
        'dt.tz_convert': '*',  # timezone conversion needs pandas
        'dt.to_period': '*',  # period conversion needs pandas
        'dt.to_pydatetime': '*',  # datetime conversion needs pandas
        # dt.normalize now uses toStartOfDay SQL function
        # DateTime properties that always need pandas
        'dt.microsecond': '*',  # no direct chDB equivalent
        'dt.nanosecond': '*',  # no direct chDB equivalent
        'dt.days_in_month': '*',  # complex calculation
        'dt.is_month_start': '*',  # boolean date property
        'dt.is_month_end': '*',  # boolean date property
        'dt.is_quarter_start': '*',  # boolean date property
        'dt.is_quarter_end': '*',  # boolean date property
        'dt.is_year_start': '*',  # boolean date property
        'dt.is_year_end': '*',  # boolean date property
        'dt.is_leap_year': '*',  # boolean date property
    }

    # Alias mappings: maps user-facing names to canonical SQL function names
    # This allows users to configure using either name (e.g., 'mean' or 'avg')
    FUNCTION_ALIASES: Dict[str, str] = {
        # Aggregate function aliases (pandas name -> SQL name)
        'mean': 'avg',
        'std': 'stddev',
        'var': 'variance',
        'prod': 'product',
        'kurt': 'kurtosis',
        # String function aliases
        'len': 'length',
        'strip': 'trim',
        'lstrip': 'ltrim',
        'rstrip': 'rtrim',
        # Math function aliases
        'ceil': 'ceiling',
        'pow': 'power',
    }

    # Reverse alias mappings (SQL name -> pandas name)
    REVERSE_ALIASES: Dict[str, str] = {v: k for k, v in FUNCTION_ALIASES.items()}

    def __init__(self):
        """Initialize with default configuration (prefer chDB)."""
        self._default_engine = ExecutionEngine.CHDB
        self._function_engines: Dict[str, ExecutionEngine] = {}

        # Pandas implementations for overlapping functions
        # Maps function_name -> (pandas_callable, accepts_args)
        self._pandas_implementations: Dict[str, Callable] = {}

        # Register default pandas implementations
        self._register_default_pandas_implementations()

    def _register_default_pandas_implementations(self):
        """Register default Pandas implementations for common functions."""
        self._register_string_functions()
        self._register_math_functions()
        self._register_aggregate_functions()
        self._register_datetime_functions()
        self._register_pandas_only_functions()

    def _register_string_functions(self):
        """Register string function implementations."""
        # Basic string functions (overlapping with ClickHouse)
        self._pandas_implementations['upper'] = lambda s: s.str.upper()
        self._pandas_implementations['lower'] = lambda s: s.str.lower()
        self._pandas_implementations['length'] = lambda s: s.str.len()
        self._pandas_implementations['len'] = lambda s: s.str.len()
        self._pandas_implementations['trim'] = lambda s: s.str.strip()
        self._pandas_implementations['strip'] = lambda s: s.str.strip()
        self._pandas_implementations['ltrim'] = lambda s: s.str.lstrip()
        self._pandas_implementations['lstrip'] = lambda s: s.str.lstrip()
        self._pandas_implementations['rtrim'] = lambda s: s.str.rstrip()
        self._pandas_implementations['rstrip'] = lambda s: s.str.rstrip()

        # Additional string functions
        self._pandas_implementations['title'] = lambda s: s.str.title()
        self._pandas_implementations['capitalize'] = lambda s: s.str.capitalize()
        self._pandas_implementations['swapcase'] = lambda s: s.str.swapcase()
        self._pandas_implementations['casefold'] = lambda s: s.str.casefold()

        # String search and match
        self._pandas_implementations['contains'] = (
            lambda s, pat, case=True, flags=0, na=None, regex=True: s.str.contains(
                pat, case=case, flags=flags, na=na, regex=regex
            )
        )
        self._pandas_implementations['startswith'] = lambda s, pat: s.str.startswith(pat)
        self._pandas_implementations['endswith'] = lambda s, pat: s.str.endswith(pat)
        self._pandas_implementations['match'] = lambda s, pat: s.str.match(pat)
        self._pandas_implementations['find'] = lambda s, sub, start=0, end=None: s.str.find(sub, start, end)
        self._pandas_implementations['rfind'] = lambda s, sub, start=0, end=None: s.str.rfind(sub, start, end)
        self._pandas_implementations['count_matches'] = lambda s, pat: s.str.count(pat)

        # String manipulation
        self._pandas_implementations['replace'] = lambda s, pat, repl, regex=True: s.str.replace(pat, repl, regex=regex)
        self._pandas_implementations['slice'] = lambda s, start=None, stop=None, step=None: s.str.slice(
            start, stop, step
        )
        self._pandas_implementations['slice_replace'] = lambda s, start=None, stop=None, repl=None: s.str.slice_replace(
            start, stop, repl
        )
        self._pandas_implementations['pad'] = lambda s, width, side='left', fillchar=' ': s.str.pad(
            width, side, fillchar
        )
        self._pandas_implementations['center'] = lambda s, width, fillchar=' ': s.str.center(width, fillchar)
        self._pandas_implementations['ljust'] = lambda s, width, fillchar=' ': s.str.ljust(width, fillchar)
        self._pandas_implementations['rjust'] = lambda s, width, fillchar=' ': s.str.rjust(width, fillchar)
        self._pandas_implementations['zfill'] = lambda s, width: s.str.zfill(width)
        self._pandas_implementations['wrap'] = lambda s, width: s.str.wrap(width)

        # String split and join
        self._pandas_implementations['split'] = lambda s, pat=None, n=-1, expand=False: s.str.split(
            pat, n=n, expand=expand
        )
        self._pandas_implementations['rsplit'] = lambda s, pat=None, n=-1, expand=False: s.str.rsplit(
            pat, n=n, expand=expand
        )
        self._pandas_implementations['partition'] = lambda s, sep=' ': s.str.partition(sep)
        self._pandas_implementations['rpartition'] = lambda s, sep=' ': s.str.rpartition(sep)
        self._pandas_implementations['cat'] = lambda s, others=None, sep=None: s.str.cat(others, sep=sep)
        self._pandas_implementations['join'] = lambda s, sep: s.str.join(sep)

        # String extraction
        self._pandas_implementations['extract'] = lambda s, pat, expand=True: s.str.extract(pat, expand=expand)
        self._pandas_implementations['extractall'] = lambda s, pat: s.str.extractall(pat)
        self._pandas_implementations['findall'] = lambda s, pat: s.str.findall(pat)
        self._pandas_implementations['get'] = lambda s, i: s.str.get(i)

        # String info
        self._pandas_implementations['isalpha'] = lambda s: s.str.isalpha()
        self._pandas_implementations['isdigit'] = lambda s: s.str.isdigit()
        self._pandas_implementations['isalnum'] = lambda s: s.str.isalnum()
        self._pandas_implementations['isspace'] = lambda s: s.str.isspace()
        self._pandas_implementations['isupper'] = lambda s: s.str.isupper()
        self._pandas_implementations['islower'] = lambda s: s.str.islower()
        self._pandas_implementations['istitle'] = lambda s: s.str.istitle()
        self._pandas_implementations['isnumeric'] = lambda s: s.str.isnumeric()
        self._pandas_implementations['isdecimal'] = lambda s: s.str.isdecimal()

        # Encoding
        self._pandas_implementations['encode'] = lambda s, encoding='utf-8': s.str.encode(encoding)
        self._pandas_implementations['decode'] = lambda s, encoding='utf-8': s.str.decode(encoding)

        # Normalization
        self._pandas_implementations['normalize'] = lambda s, form='NFC': s.str.normalize(form)
        self._pandas_implementations['translate'] = lambda s, table: s.str.translate(table)

    def _register_math_functions(self):
        """Register math function implementations."""
        import numpy as np

        # Basic math (overlapping with ClickHouse)
        self._pandas_implementations['abs'] = lambda s: s.abs()
        self._pandas_implementations['round'] = lambda s, decimals=0: s.round(decimals)
        self._pandas_implementations['floor'] = lambda s: np.floor(s)
        self._pandas_implementations['ceil'] = lambda s: np.ceil(s)
        self._pandas_implementations['ceiling'] = lambda s: np.ceil(s)
        self._pandas_implementations['sqrt'] = lambda s: np.sqrt(s)

        # Logarithms and exponentials
        self._pandas_implementations['log'] = lambda s: np.log(s)
        self._pandas_implementations['log10'] = lambda s: np.log10(s)
        self._pandas_implementations['log2'] = lambda s: np.log2(s)
        self._pandas_implementations['log1p'] = lambda s: np.log1p(s)
        self._pandas_implementations['exp'] = lambda s: np.exp(s)
        self._pandas_implementations['expm1'] = lambda s: np.expm1(s)
        self._pandas_implementations['pow'] = lambda s, exp: np.power(s, exp)
        self._pandas_implementations['power'] = lambda s, exp: np.power(s, exp)

        # Trigonometric
        self._pandas_implementations['sin'] = lambda s: np.sin(s)
        self._pandas_implementations['cos'] = lambda s: np.cos(s)
        self._pandas_implementations['tan'] = lambda s: np.tan(s)
        self._pandas_implementations['asin'] = lambda s: np.arcsin(s)
        self._pandas_implementations['acos'] = lambda s: np.arccos(s)
        self._pandas_implementations['atan'] = lambda s: np.arctan(s)
        self._pandas_implementations['atan2'] = lambda s, other: np.arctan2(s, other)
        self._pandas_implementations['sinh'] = lambda s: np.sinh(s)
        self._pandas_implementations['cosh'] = lambda s: np.cosh(s)
        self._pandas_implementations['tanh'] = lambda s: np.tanh(s)
        self._pandas_implementations['asinh'] = lambda s: np.arcsinh(s)
        self._pandas_implementations['acosh'] = lambda s: np.arccosh(s)
        self._pandas_implementations['atanh'] = lambda s: np.arctanh(s)

        # Rounding and sign
        self._pandas_implementations['sign'] = lambda s: np.sign(s)
        self._pandas_implementations['trunc'] = lambda s: np.trunc(s)
        self._pandas_implementations['mod'] = lambda s, other: np.mod(s, other)
        self._pandas_implementations['fmod'] = lambda s, other: np.fmod(s, other)
        self._pandas_implementations['clip'] = lambda s, lower=None, upper=None: s.clip(lower, upper)

        # Other math
        self._pandas_implementations['degrees'] = lambda s: np.degrees(s)
        self._pandas_implementations['radians'] = lambda s: np.radians(s)
        self._pandas_implementations['hypot'] = lambda s, other: np.hypot(s, other)

    def _register_aggregate_functions(self):
        """Register aggregate function implementations."""
        self._pandas_implementations['sum'] = lambda s: s.sum()
        self._pandas_implementations['avg'] = lambda s: s.mean()
        self._pandas_implementations['mean'] = lambda s: s.mean()
        self._pandas_implementations['min'] = lambda s: s.min()
        self._pandas_implementations['max'] = lambda s: s.max()
        self._pandas_implementations['count'] = lambda s: s.count()
        self._pandas_implementations['std'] = lambda s: s.std()
        self._pandas_implementations['stddev'] = lambda s: s.std()
        self._pandas_implementations['var'] = lambda s: s.var()
        self._pandas_implementations['variance'] = lambda s: s.var()
        self._pandas_implementations['median'] = lambda s: s.median()
        self._pandas_implementations['mode'] = lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else None
        self._pandas_implementations['first'] = lambda s: s.iloc[0] if len(s) > 0 else None
        self._pandas_implementations['last'] = lambda s: s.iloc[-1] if len(s) > 0 else None
        self._pandas_implementations['nunique'] = lambda s: s.nunique()
        self._pandas_implementations['prod'] = lambda s: s.prod()
        self._pandas_implementations['product'] = lambda s: s.prod()
        self._pandas_implementations['sem'] = lambda s: s.sem()
        self._pandas_implementations['skew'] = lambda s: s.skew()
        self._pandas_implementations['kurt'] = lambda s: s.kurt()
        self._pandas_implementations['kurtosis'] = lambda s: s.kurtosis()

        # Quantiles
        self._pandas_implementations['quantile'] = lambda s, q=0.5: s.quantile(q)
        self._pandas_implementations['percentile'] = lambda s, q: s.quantile(q / 100)

    def _register_datetime_functions(self):
        """Register datetime function implementations."""
        # Date part extraction
        self._pandas_implementations['year'] = lambda s: s.dt.year
        self._pandas_implementations['month'] = lambda s: s.dt.month
        self._pandas_implementations['day'] = lambda s: s.dt.day
        self._pandas_implementations['hour'] = lambda s: s.dt.hour
        self._pandas_implementations['minute'] = lambda s: s.dt.minute
        self._pandas_implementations['second'] = lambda s: s.dt.second
        self._pandas_implementations['microsecond'] = lambda s: s.dt.microsecond
        self._pandas_implementations['nanosecond'] = lambda s: s.dt.nanosecond
        self._pandas_implementations['dayofweek'] = lambda s: s.dt.dayofweek
        self._pandas_implementations['weekday'] = lambda s: s.dt.weekday
        self._pandas_implementations['dayofyear'] = lambda s: s.dt.dayofyear
        self._pandas_implementations['weekofyear'] = lambda s: s.dt.isocalendar().week
        self._pandas_implementations['week'] = lambda s: s.dt.isocalendar().week
        self._pandas_implementations['quarter'] = lambda s: s.dt.quarter

        # Date properties
        self._pandas_implementations['date'] = lambda s: s.dt.date
        self._pandas_implementations['time'] = lambda s: s.dt.time
        self._pandas_implementations['days_in_month'] = lambda s: s.dt.days_in_month
        self._pandas_implementations['daysinmonth'] = lambda s: s.dt.daysinmonth
        self._pandas_implementations['is_month_start'] = lambda s: s.dt.is_month_start
        self._pandas_implementations['is_month_end'] = lambda s: s.dt.is_month_end
        self._pandas_implementations['is_quarter_start'] = lambda s: s.dt.is_quarter_start
        self._pandas_implementations['is_quarter_end'] = lambda s: s.dt.is_quarter_end
        self._pandas_implementations['is_year_start'] = lambda s: s.dt.is_year_start
        self._pandas_implementations['is_year_end'] = lambda s: s.dt.is_year_end
        self._pandas_implementations['is_leap_year'] = lambda s: s.dt.is_leap_year

        # Timezone
        self._pandas_implementations['tz_localize'] = lambda s, tz: s.dt.tz_localize(tz)
        self._pandas_implementations['tz_convert'] = lambda s, tz: s.dt.tz_convert(tz)

        # Date formatting
        self._pandas_implementations['strftime'] = lambda s, fmt: s.dt.strftime(fmt)

        # Date normalization
        self._pandas_implementations['normalize'] = lambda s: s.dt.normalize()
        self._pandas_implementations['floor_dt'] = lambda s, freq: s.dt.floor(freq)
        self._pandas_implementations['ceil_dt'] = lambda s, freq: s.dt.ceil(freq)
        self._pandas_implementations['round_dt'] = lambda s, freq: s.dt.round(freq)

        # Date conversion
        self._pandas_implementations['to_period'] = lambda s, freq=None: s.dt.to_period(freq)
        self._pandas_implementations['to_pydatetime'] = lambda s: s.dt.to_pydatetime()
        self._pandas_implementations['to_pytimedelta'] = lambda s: s.dt.to_pytimedelta()

    def _register_pandas_only_functions(self):
        """Register Pandas-specific functions (no ClickHouse equivalent)."""
        # Cumulative functions
        self._pandas_implementations['cumsum'] = lambda s: s.cumsum()
        self._pandas_implementations['cummax'] = lambda s: s.cummax()
        self._pandas_implementations['cummin'] = lambda s: s.cummin()
        self._pandas_implementations['cumprod'] = lambda s: s.cumprod()

        # Shift and diff
        self._pandas_implementations['shift'] = lambda s, periods=1, fill_value=None: s.shift(
            periods, fill_value=fill_value
        )
        self._pandas_implementations['diff'] = lambda s, periods=1: s.diff(periods)
        self._pandas_implementations['pct_change'] = lambda s, periods=1: s.pct_change(periods)

        # Ranking
        self._pandas_implementations['rank'] = lambda s, method='average', ascending=True: s.rank(
            method=method, ascending=ascending
        )
        self._pandas_implementations['nlargest'] = lambda s, n=5: s.nlargest(n)
        self._pandas_implementations['nsmallest'] = lambda s, n=5: s.nsmallest(n)

        # Value operations
        self._pandas_implementations['unique'] = lambda s: pd.Series(s.unique())
        self._pandas_implementations['value_counts'] = lambda s, normalize=False: s.value_counts(normalize=normalize)
        self._pandas_implementations['duplicated'] = lambda s, keep='first': s.duplicated(keep=keep)
        self._pandas_implementations['drop_duplicates'] = lambda s, keep='first': s.drop_duplicates(keep=keep)

        # Missing value handling (pandas method names)
        self._pandas_implementations['isna'] = lambda s: s.isna()
        self._pandas_implementations['isnull'] = lambda s: s.isnull()
        self._pandas_implementations['notna'] = lambda s: s.notna()
        self._pandas_implementations['notnull'] = lambda s: s.notnull()
        self._pandas_implementations['fillna'] = lambda s, value=None, method=None: s.fillna(value=value, method=method)
        self._pandas_implementations['dropna'] = lambda s: s.dropna()
        self._pandas_implementations['ffill'] = lambda s: s.ffill()
        self._pandas_implementations['bfill'] = lambda s: s.bfill()
        self._pandas_implementations['interpolate'] = lambda s, method='linear': s.interpolate(method=method)
        # Missing value handling (SQL function names - map to pandas for NaN support)
        # Note: keys must be lowercase as function names are lowercased before lookup
        self._pandas_implementations['isnull'] = lambda s: s.isna()
        self._pandas_implementations['isnotnull'] = lambda s: s.notna()

        # Type checking
        self._pandas_implementations['isin'] = lambda s, values: s.isin(values)
        self._pandas_implementations['between'] = lambda s, left, right, inclusive='both': s.between(
            left, right, inclusive=inclusive
        )

        # Apply and map
        self._pandas_implementations['apply'] = lambda s, func: s.apply(func)
        self._pandas_implementations['map'] = lambda s, arg: s.map(arg)

        # Comparison with shift
        self._pandas_implementations['eq'] = lambda s, other: s.eq(other)
        self._pandas_implementations['ne'] = lambda s, other: s.ne(other)
        self._pandas_implementations['lt'] = lambda s, other: s.lt(other)
        self._pandas_implementations['le'] = lambda s, other: s.le(other)
        self._pandas_implementations['gt'] = lambda s, other: s.gt(other)
        self._pandas_implementations['ge'] = lambda s, other: s.ge(other)

        # Clipping
        self._pandas_implementations['clip_lower'] = lambda s, threshold: s.clip(lower=threshold)
        self._pandas_implementations['clip_upper'] = lambda s, threshold: s.clip(upper=threshold)

        # Binning
        self._pandas_implementations['cut'] = lambda s, bins, labels=None: pd.cut(s, bins, labels=labels)
        self._pandas_implementations['qcut'] = lambda s, q, labels=None: pd.qcut(s, q, labels=labels)

        # Type conversion
        self._pandas_implementations['astype'] = lambda s, dtype: s.astype(dtype)

    @property
    def default_engine(self) -> ExecutionEngine:
        """Get the default execution engine."""
        return self._default_engine

    @default_engine.setter
    def default_engine(self, engine: ExecutionEngine):
        """Set the default execution engine."""
        self._default_engine = engine

    def _get_all_names(self, name: str) -> list:
        """
        Get all equivalent names for a function (original + aliases).

        Args:
            name: Function name (lowercase)

        Returns:
            List of all equivalent function names
        """
        names = [name]
        # Check if this name has an alias (pandas name -> SQL name)
        if name in self.FUNCTION_ALIASES:
            names.append(self.FUNCTION_ALIASES[name])
        # Check if this name has a reverse alias (SQL name -> pandas name)
        if name in self.REVERSE_ALIASES:
            names.append(self.REVERSE_ALIASES[name])
        return names

    def use_chdb(self, *function_names: str) -> 'FunctionExecutorConfig':
        """
        Configure functions to use chDB SQL engine.

        Args:
            *function_names: Function names to configure (supports aliases like 'mean'/'avg')

        Returns:
            self for chaining

        Example:
            >>> function_config.use_chdb('upper', 'lower', 'length')
            >>> function_config.use_chdb('mean')  # Also sets 'avg'
        """
        for name in function_names:
            name_lower = name.lower()
            # Set for all equivalent names (original + aliases)
            for equiv_name in self._get_all_names(name_lower):
                self._function_engines[equiv_name] = ExecutionEngine.CHDB
        return self

    def use_pandas(self, *function_names: str) -> 'FunctionExecutorConfig':
        """
        Configure functions to use Pandas execution.

        Args:
            *function_names: Function names to configure (supports aliases like 'mean'/'avg')

        Returns:
            self for chaining

        Example:
            >>> function_config.use_pandas('upper', 'lower')
            >>> function_config.use_pandas('mean')  # Also sets 'avg'
        """
        for name in function_names:
            name_lower = name.lower()
            # Set for all equivalent names (original + aliases)
            for equiv_name in self._get_all_names(name_lower):
                self._function_engines[equiv_name] = ExecutionEngine.PANDAS
        return self

    def get_engine(self, function_name: str) -> ExecutionEngine:
        """
        Get the execution engine for a specific function.

        Args:
            function_name: The function name (supports aliases like 'mean'/'avg')

        Returns:
            ExecutionEngine for the function
        """
        name_lower = function_name.lower()

        # Pandas-only functions must use Pandas
        if name_lower in self.PANDAS_ONLY_FUNCTIONS:
            return ExecutionEngine.PANDAS

        # Check user configuration for this name or any of its aliases
        for equiv_name in self._get_all_names(name_lower):
            if equiv_name in self._function_engines:
                return self._function_engines[equiv_name]

        return self._default_engine

    def should_use_chdb(self, function_name: str) -> bool:
        """Check if a function should use chDB."""
        name_lower = function_name.lower()
        # Pandas-only functions cannot use chDB
        if name_lower in self.PANDAS_ONLY_FUNCTIONS:
            return False
        engine = self.get_engine(function_name)
        return engine in (ExecutionEngine.CHDB, ExecutionEngine.AUTO)

    def should_use_pandas(self, function_name: str) -> bool:
        """Check if a function should use Pandas."""
        name_lower = function_name.lower()
        # Pandas-only functions must use Pandas
        if name_lower in self.PANDAS_ONLY_FUNCTIONS:
            return True
        return self.get_engine(function_name) == ExecutionEngine.PANDAS

    def is_pandas_only(self, function_name: str) -> bool:
        """Check if a function is Pandas-only (no ClickHouse equivalent)."""
        return function_name.lower() in self.PANDAS_ONLY_FUNCTIONS

    def needs_accessor_fallback(self, accessor_method: str, **kwargs) -> bool:
        """
        Check if an accessor method needs pandas fallback based on parameters.

        Args:
            accessor_method: The accessor method name (e.g., 'str.split', 'dt.strftime')
            **kwargs: The parameters passed to the method

        Returns:
            True if pandas fallback is needed, False otherwise

        Example:
            >>> function_config.needs_accessor_fallback('str.split', expand=True)
            True
            >>> function_config.needs_accessor_fallback('str.split', expand=False)
            False
            >>> function_config.needs_accessor_fallback('str.extract')
            True  # Always needs pandas
        """
        method_key = accessor_method.lower()

        # Check if this method is in the fallback registry
        if method_key not in self.ACCESSOR_PARAM_PANDAS_FALLBACK:
            return False

        condition = self.ACCESSOR_PARAM_PANDAS_FALLBACK[method_key]

        # '*' means always use pandas
        if condition == '*':
            return True

        # Dict means check specific parameter values
        if isinstance(condition, dict):
            for param_name, required_value in condition.items():
                if param_name in kwargs:
                    if required_value == '*':
                        # Any value of this param triggers fallback
                        return True
                    elif kwargs[param_name] == required_value:
                        # Specific value triggers fallback
                        return True

        return False

    def get_accessor_fallback_reason(self, accessor_method: str, **kwargs) -> Optional[str]:
        """
        Get the reason why an accessor method needs pandas fallback.

        Args:
            accessor_method: The accessor method name (e.g., 'str.split', 'dt.strftime')
            **kwargs: The parameters passed to the method

        Returns:
            Reason string if fallback is needed, None otherwise

        Example:
            >>> function_config.get_accessor_fallback_reason('str.split', expand=True)
            'str.split with expand=True requires pandas (returns DataFrame)'
        """
        method_key = accessor_method.lower()

        if method_key not in self.ACCESSOR_PARAM_PANDAS_FALLBACK:
            return None

        condition = self.ACCESSOR_PARAM_PANDAS_FALLBACK[method_key]

        if condition == '*':
            return f"'{accessor_method}' always requires pandas (no SQL implementation)"

        if isinstance(condition, dict):
            for param_name, required_value in condition.items():
                if param_name in kwargs:
                    if required_value == '*':
                        return f"'{accessor_method}' with {param_name}={kwargs[param_name]} requires pandas"
                    elif kwargs[param_name] == required_value:
                        return f"'{accessor_method}' with {param_name}={required_value} requires pandas"

        return None

    def list_accessor_fallbacks(self) -> Dict[str, Any]:
        """
        Get a dictionary of all accessor fallback conditions.

        Returns:
            Dict mapping accessor.method to condition

        Example:
            >>> fallbacks = function_config.list_accessor_fallbacks()
            >>> print(fallbacks['str.split'])
            {'expand': True}
        """
        return dict(self.ACCESSOR_PARAM_PANDAS_FALLBACK)

    def has_pandas_implementation(self, function_name: str) -> bool:
        """Check if a Pandas implementation is available."""
        return function_name.lower() in self._pandas_implementations

    def get_pandas_implementation(self, function_name: str) -> Optional[Callable]:
        """Get the Pandas implementation for a function."""
        return self._pandas_implementations.get(function_name.lower())

    def register_pandas_implementation(self, function_name: str, implementation: Callable) -> 'FunctionExecutorConfig':
        """
        Register a custom Pandas implementation for a function.

        Args:
            function_name: The function name
            implementation: Callable that takes (Series, *args) and returns result

        Returns:
            self for chaining

        Example:
            >>> function_config.register_pandas_implementation(
            ...     'custom_func',
            ...     lambda s, x: s * x + 1
            ... )
        """
        self._pandas_implementations[function_name.lower()] = implementation
        return self

    def reset(self) -> 'FunctionExecutorConfig':
        """
        Reset all function-specific configurations to defaults.

        Returns:
            self for chaining
        """
        self._function_engines.clear()
        self._default_engine = ExecutionEngine.CHDB
        return self

    def prefer_pandas(self) -> 'FunctionExecutorConfig':
        """
        Set default engine to Pandas for all overlapping functions.

        Returns:
            self for chaining
        """
        self._default_engine = ExecutionEngine.PANDAS
        return self

    def prefer_chdb(self) -> 'FunctionExecutorConfig':
        """
        Set default engine to chDB for all overlapping functions (default).

        Returns:
            self for chaining
        """
        self._default_engine = ExecutionEngine.CHDB
        return self

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration.

        Returns:
            Dict with configuration details
        """
        return {
            'default_engine': self._default_engine.value,
            'custom_mappings': {k: v.value for k, v in self._function_engines.items()},
            'overlapping_functions': len(self.OVERLAPPING_FUNCTIONS),
            'pandas_implementations': len(self._pandas_implementations),
            'pandas_only_functions': len(self.PANDAS_ONLY_FUNCTIONS),
            'accessor_fallbacks': len(self.ACCESSOR_PARAM_PANDAS_FALLBACK),
        }

    def __repr__(self) -> str:
        custom_count = len(self._function_engines)
        return f"FunctionExecutorConfig(" f"default={self._default_engine.value}, " f"custom_mappings={custom_count})"


# Global configuration instance
function_config = FunctionExecutorConfig()


# Convenience functions for global config
def use_chdb(*function_names: str) -> FunctionExecutorConfig:
    """Configure functions to use chDB SQL engine."""
    return function_config.use_chdb(*function_names)


def use_pandas(*function_names: str) -> FunctionExecutorConfig:
    """Configure functions to use Pandas execution."""
    return function_config.use_pandas(*function_names)


def prefer_pandas() -> FunctionExecutorConfig:
    """Set default engine to Pandas."""
    return function_config.prefer_pandas()


def prefer_chdb() -> FunctionExecutorConfig:
    """Set default engine to chDB (default)."""
    return function_config.prefer_chdb()


def reset_function_config() -> FunctionExecutorConfig:
    """Reset function configuration to defaults."""
    return function_config.reset()
