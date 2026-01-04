"""
Pandas-compatible module-level functions for DataStore.

This module provides pandas-like API functions that can be used as drop-in
replacements for pandas functions. These functions delegate to pandas internally
and wrap results in DataStore objects where appropriate.

Usage:
    >>> import datastore as ds
    >>> df = ds.read_csv("data.csv")
    >>> ds.isna(None)
    True
    >>> ds.to_datetime("2024-01-15")
    Timestamp('2024-01-15 00:00:00')
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# IMPORTANT: Save a reference to the real pandas module at import time.
# This allows DataStore to work correctly even when users do monkey-patching like:
#   sys.modules['pandas'] = datastore
# Without this, any `import pandas` inside functions would get datastore instead,
# causing infinite recursion.
import pandas as _pd

# For backward compatibility with code that uses `pd` directly
pd = _pd

if TYPE_CHECKING:
    from .core import DataStore as DataStoreType


def _get_datastore_class():
    """Lazy import to avoid circular dependency."""
    from .core import DataStore

    return DataStore


# ========== IO Functions ==========


def read_csv(filepath_or_buffer, sep=',', **kwargs) -> 'DataStoreType':
    """
    Read a comma-separated values (CSV) file into DataStore.

    This function automatically chooses the optimal execution engine:
    - Uses chDB SQL engine when possible (enables full SQL compilation for
      subsequent operations like filter, groupby, sort, etc.)
    - Falls back to pandas when advanced pandas-only features are used

    The function strives to match pandas.read_csv() behavior as closely as
    possible. For CSV files, the first row is treated as column names by default
    (header=0 or 'infer'), matching pandas' default behavior.

    Args:
        filepath_or_buffer: Path to the CSV file, URL, or file-like object
        sep: Delimiter to use (default ',')
        **kwargs: All pandas.read_csv() arguments are supported:
            - header: Row number to use as column names
                - 0 or 'infer' (default): first row is column names
                - None: no header row, auto-generate column names
                - int > 0: use row N as header (falls back to pandas)
            - names: List of column names to use (falls back to pandas)
            - index_col: Column(s) to use as row labels (falls back to pandas)
            - usecols: Return a subset of columns (falls back to pandas)
            - dtype: Data type for columns (falls back to pandas)
            - skiprows: Number of rows to skip at the beginning
            - nrows: Number of rows to read
            - na_values: Additional strings to recognize as NA/NaN
            - parse_dates: Columns to parse as dates (falls back to pandas)
            - true_values: Values to consider as True (falls back to pandas)
            - false_values: Values to consider as False (falls back to pandas)
            - encoding: Encoding to use for reading (default 'utf-8')
            - compression: Compression type ('infer', 'gzip', 'bz2', etc.)
            - quotechar: Character used to denote quoted strings
            - escapechar: Character used to escape other characters
            - comment: Character indicating comment lines
            - thousands: Thousands separator
            - decimal: Character for decimal point
            - skip_blank_lines: Skip over blank lines (default True)
            - on_bad_lines: How to handle bad lines ('error', 'warn', 'skip')

    Returns:
        DataStore: A DataStore object containing the CSV data

    Example:
        >>> from datastore import read_csv
        >>> df = read_csv("data.csv")  # Uses SQL engine automatically
        >>> df.head()

        >>> # These use SQL engine (chDB supports these options)
        >>> df = read_csv("data.csv", header=None)  # No header row
        >>> df = read_csv("data.csv", compression='gzip')
        >>> df = read_csv("data.csv", skiprows=1)
        >>> df = read_csv("data.csv", nrows=100)

        >>> # These automatically fall back to pandas
        >>> df = read_csv("data.csv", parse_dates=['date_col'])
        >>> df = read_csv("data.csv", usecols=['name', 'age'])
        >>> df = read_csv("data.csv", dtype={'age': int})
        >>> df = read_csv("data.csv", true_values=['yes'], false_values=['no'])

    Notes:
        - Boolean values: By default, ClickHouse recognizes 'true'/'false' (case-insensitive).
          For custom boolean strings like 'yes'/'no', use true_values/false_values (falls back
          to pandas because chDB's bool settings only control OUTPUT format, not INPUT parsing).
        - Date/DateTime: chDB auto-infers date/datetime columns when possible.
          Use parse_dates for explicit control (falls back to pandas).
        - NULL values: Empty strings and '\\N' are treated as NULL by default.
          Use na_values for custom null strings (falls back to pandas).
    """
    DataStore = _get_datastore_class()

    # Extract header parameter
    header = kwargs.get('header', 'infer')

    # Check for unsupported iterator parameters FIRST
    # These return TextFileReader instead of DataFrame and are not supported
    if 'chunksize' in kwargs or 'iterator' in kwargs:
        raise NotImplementedError(
            "DataStore does not support chunked reading (chunksize/iterator parameters). "
            "Use pandas.read_csv() directly for chunked reading, or remove these parameters."
        )

    # Parameters that MUST fall back to pandas (chDB cannot handle)
    pandas_required_params = {
        'names',  # Custom column names (chDB doesn't support renaming on read)
        'index_col',  # Set index column (pandas concept)
        'usecols',  # Select specific columns (could use SQL SELECT but behavior differs)
        'dtype',  # Force type conversion (chDB has different type system)
        'parse_dates',  # Date parsing with specific format
        'date_parser',  # Custom date parser function
        'date_format',  # Date format string
        'dayfirst',  # Date format preference
        'converters',  # Custom converter functions
        'true_values',  # Custom boolean true values (chDB bool_true_representation is for OUTPUT only)
        'false_values',  # Custom boolean false values (chDB bool_false_representation is for OUTPUT only)
    }

    # Check if we must use pandas
    needs_pandas = any(param in kwargs for param in pandas_required_params)

    # Header values > 0 require pandas (e.g., header=2 means row 2 is header)
    if isinstance(header, int) and header > 0:
        needs_pandas = True
    # List of header rows requires pandas (MultiIndex columns)
    if isinstance(header, list):
        needs_pandas = True

    # Check for file-like objects (chDB needs file path)
    if hasattr(filepath_or_buffer, 'read'):
        needs_pandas = True

    # Check for URL (use pandas for HTTP URLs for simplicity)
    if isinstance(filepath_or_buffer, str) and (
        filepath_or_buffer.startswith('http://') or filepath_or_buffer.startswith('https://')
    ):
        needs_pandas = True

    # Parameters that can be mapped to chDB settings OR fall back to pandas
    # We try to handle as many as possible with chDB
    true_values = kwargs.get('true_values')
    false_values = kwargs.get('false_values')
    na_values = kwargs.get('na_values')
    keep_default_na = kwargs.get('keep_default_na', True)
    na_filter = kwargs.get('na_filter', True)

    # These parameters require pandas if na_values is complex
    if na_values is not None and not isinstance(na_values, (str, list)):
        # Dict mapping columns to different na_values requires pandas
        needs_pandas = True

    if needs_pandas:
        # Use pandas for full compatibility
        pandas_df = _pd.read_csv(filepath_or_buffer, sep=sep, **kwargs)
        return DataStore.from_df(pandas_df)

    # ===== Use chDB SQL engine =====
    # Extract and pop parameters we handle specially
    compression = kwargs.pop('compression', None)
    skiprows = kwargs.pop('skiprows', None)
    nrows = kwargs.pop('nrows', None)
    skip_blank_lines = kwargs.pop('skip_blank_lines', True)
    on_bad_lines = kwargs.pop('on_bad_lines', 'error')
    quotechar = kwargs.pop('quotechar', '"')
    escapechar = kwargs.pop('escapechar', None)
    comment = kwargs.pop('comment', None)
    thousands = kwargs.pop('thousands', None)
    decimal = kwargs.pop('decimal', '.')
    encoding = kwargs.pop('encoding', None)
    skipinitialspace = kwargs.pop('skipinitialspace', False)
    skipfooter = kwargs.pop('skipfooter', 0)
    low_memory = kwargs.pop('low_memory', True)
    memory_map = kwargs.pop('memory_map', False)
    float_precision = kwargs.pop('float_precision', None)

    # Remove handled params from kwargs
    kwargs.pop('header', None)
    kwargs.pop('true_values', None)
    kwargs.pop('false_values', None)
    kwargs.pop('na_values', None)
    kwargs.pop('keep_default_na', None)
    kwargs.pop('na_filter', None)

    # Some params require pandas but we haven't checked yet
    if skipfooter > 0 or comment is not None or thousands is not None:
        # These require pandas
        pandas_df = _pd.read_csv(
            filepath_or_buffer,
            sep=sep,
            header=header,
            skiprows=skiprows,
            nrows=nrows,
            skip_blank_lines=skip_blank_lines,
            on_bad_lines=on_bad_lines,
            quotechar=quotechar,
            escapechar=escapechar,
            comment=comment,
            thousands=thousands,
            decimal=decimal,
            encoding=encoding,
            skipinitialspace=skipinitialspace,
            skipfooter=skipfooter,
            low_memory=low_memory,
            memory_map=memory_map,
            float_precision=float_precision,
            true_values=true_values,
            false_values=false_values,
            na_values=na_values,
            keep_default_na=keep_default_na,
            na_filter=na_filter,
            **kwargs,
        )
        return DataStore.from_df(pandas_df)

    # For custom delimiters other than tab, fall back to pandas
    # ClickHouse's CSV format with custom delimiter has schema inference issues
    if sep != ',' and sep != '\t':
        pandas_df = _pd.read_csv(
            filepath_or_buffer,
            sep=sep,
            header=header,
            skiprows=skiprows,
            nrows=nrows,
            skip_blank_lines=skip_blank_lines,
            on_bad_lines=on_bad_lines,
            quotechar=quotechar,
            escapechar=escapechar,
            decimal=decimal,
            encoding=encoding,
            skipinitialspace=skipinitialspace,
            low_memory=low_memory,
            memory_map=memory_map,
            float_precision=float_precision,
            true_values=true_values,
            false_values=false_values,
            na_values=na_values,
            keep_default_na=keep_default_na,
            na_filter=na_filter,
            **kwargs,
        )
        return DataStore.from_df(pandas_df)

    # Determine format based on header and delimiter
    # For tab delimiter, use TSV format which natively supports tabs
    if sep == '\t':
        csv_format = 'TSVWithNames' if header != None else 'TSV'
    else:
        # header=0 or 'infer': first row is column names (CSVWithNames)
        # header=None: no header row, generate column names (CSV)
        csv_format = 'CSVWithNames'  # Default: first row is header
        if header is None:
            csv_format = 'CSV'  # No header row

    ds = DataStore.from_file(
        filepath_or_buffer,
        format=csv_format,
        compression=compression,
    )

    # Build chDB format settings
    settings = {}

    # Delimiter
    if sep != ',':
        settings['format_csv_delimiter'] = sep

    # Skip rows (after header)
    if skiprows:
        if isinstance(skiprows, int):
            settings['input_format_csv_skip_first_lines'] = skiprows
        else:
            # Complex skiprows (callable, list) requires pandas
            pandas_df = _pd.read_csv(filepath_or_buffer, sep=sep, skiprows=skiprows, **kwargs)
            return DataStore.from_df(pandas_df)

    # Note: true_values/false_values now fall back to pandas
    # ClickHouse's bool_true_representation/bool_false_representation settings
    # are for OUTPUT format only, not for parsing input values.

    # NA/NULL handling
    if na_values is not None:
        if isinstance(na_values, str):
            settings['format_csv_null_representation'] = na_values
        elif isinstance(na_values, list) and len(na_values) > 0:
            # Use the first value (chDB supports only one null representation)
            settings['format_csv_null_representation'] = str(na_values[0])

    # Whitespace handling
    if skipinitialspace:
        settings['input_format_csv_trim_whitespaces'] = 1

    # Quote handling
    if quotechar == '"':
        settings['format_csv_allow_double_quotes'] = 1
    elif quotechar == "'":
        settings['format_csv_allow_single_quotes'] = 1

    # Skip blank lines (default True in pandas)
    if skip_blank_lines:
        settings['input_format_csv_skip_trailing_empty_lines'] = 1

    # Error handling
    if on_bad_lines == 'skip' or on_bad_lines == 'warn':
        settings['input_format_csv_use_default_on_bad_values'] = 1
        settings['input_format_allow_errors_num'] = 10000  # Allow many errors
        settings['input_format_allow_errors_ratio'] = 0.1  # Up to 10% errors

    # Schema inference settings for better type detection
    settings['input_format_csv_use_best_effort_in_schema_inference'] = 1
    settings['input_format_try_infer_integers'] = 1
    settings['input_format_try_infer_dates'] = 1
    settings['input_format_try_infer_datetimes'] = 1

    # Apply settings
    if settings:
        ds = ds.with_format_settings(**settings)

    # Apply LIMIT for nrows
    if nrows is not None:
        ds = ds.limit(nrows)

    return ds


def read_parquet(path, columns=None, **kwargs) -> 'DataStoreType':
    """
    Read a Parquet file into DataStore.

    This function automatically chooses the optimal execution engine:
    - Uses chDB SQL engine when possible (enables full SQL compilation)
    - Falls back to pandas when advanced pandas-only features are used

    Args:
        path: Path to the Parquet file, URL, or file-like object
        columns: List of column names to read (None reads all columns)
        **kwargs: Additional pandas.read_parquet() arguments:
            - engine: Parquet library to use ('auto', 'pyarrow', 'fastparquet')
            - use_nullable_dtypes: Use nullable dtypes for pandas 1.0+
            - filters: List of filters for row group filtering

    Returns:
        DataStore: A DataStore object containing the Parquet data

    Example:
        >>> from datastore import read_parquet
        >>> df = read_parquet("data.parquet")  # Uses SQL engine
        >>> df = read_parquet("data.parquet", columns=['name', 'age'])  # Uses pandas
    """
    DataStore = _get_datastore_class()

    # Parameters that require pandas
    pandas_only_params = {
        'engine',  # Specific parquet engine
        'use_nullable_dtypes',  # Nullable dtypes
        'dtype_backend',  # Dtype backend selection
        'filesystem',  # Custom filesystem
        'filters',  # Row group filters (pyarrow specific)
        'storage_options',  # Storage options for remote
    }

    needs_pandas = any(param in kwargs for param in pandas_only_params)

    # columns parameter requires pandas (chDB reads all columns)
    if columns is not None:
        needs_pandas = True

    # File-like objects need pandas
    if hasattr(path, 'read'):
        needs_pandas = True

    # HTTP URLs - use pandas for simplicity
    if isinstance(path, str) and (path.startswith('http://') or path.startswith('https://')):
        needs_pandas = True

    if needs_pandas:
        pandas_df = pd.read_parquet(path, columns=columns, **kwargs)
        return DataStore.from_df(pandas_df)
    else:
        # Use chDB SQL engine
        return DataStore.from_file(path, format='Parquet')


def read_json(path_or_buf, orient=None, lines=False, **kwargs) -> 'DataStoreType':
    """
    Read a JSON file into DataStore.

    This function automatically chooses the optimal execution engine:
    - Uses chDB SQL engine for JSON Lines format (lines=True) when possible
    - Falls back to pandas for complex JSON formats or pandas-only features

    Args:
        path_or_buf: Path to the JSON file, URL, or file-like object
        orient: Expected JSON string format. Compatible values are:
            - 'split': dict like {index -> [index], columns -> [columns], data -> [values]}
            - 'records': list like [{column -> value}, ... , {column -> value}]
            - 'index': dict like {index -> {column -> value}}
            - 'columns': dict like {column -> {index -> value}}
            - 'values': just the values array
        lines: Read the file as JSON Lines (one JSON object per line)
        **kwargs: Additional pandas.read_json() arguments:
            - typ: Type of object to recover ('frame' or 'series')
            - dtype: Data types for columns
            - convert_axes: Try to convert axes to proper dtypes
            - convert_dates: Parse date columns
            - precise_float: Use higher precision float parsing
            - encoding: Encoding for reading
            - compression: Compression type

    Returns:
        DataStore: A DataStore object containing the JSON data

    Example:
        >>> from datastore import read_json
        >>> df = read_json("data.json")
        >>> df = read_json("data.json", orient='records')

        >>> # JSON Lines format - uses SQL engine
        >>> df = read_json("data.jsonl", lines=True)
    """
    DataStore = _get_datastore_class()

    # Parameters that require pandas
    pandas_only_params = {
        'typ',  # Object type (frame vs series)
        'dtype',  # Type conversion
        'convert_axes',  # Axes conversion
        'convert_dates',  # Date parsing
        'precise_float',  # Float precision
        'encoding',  # Encoding
        'date_unit',  # Date unit
        'encoding_errors',  # Encoding error handling
        'chunksize',  # Chunked reading
        'nrows',  # Row limit (pandas specific handling)
    }

    needs_pandas = any(param in kwargs for param in pandas_only_params)

    # File-like objects need pandas
    if hasattr(path_or_buf, 'read'):
        needs_pandas = True

    # HTTP URLs - use pandas for simplicity
    if isinstance(path_or_buf, str) and (path_or_buf.startswith('http://') or path_or_buf.startswith('https://')):
        needs_pandas = True

    # chDB only supports JSONEachRow (lines=True) format well
    # Other orient formats need pandas
    if not lines or orient is not None:
        needs_pandas = True

    if needs_pandas:
        pandas_df = pd.read_json(path_or_buf, orient=orient, lines=lines, **kwargs)
        return DataStore.from_df(pandas_df)
    else:
        # Use chDB SQL engine for JSON Lines format
        compression = kwargs.pop('compression', None)
        return DataStore.from_file(path_or_buf, format='JSONEachRow', compression=compression)


def read_excel(io, sheet_name=0, **kwargs) -> 'DataStoreType':
    """
    Read an Excel file into DataStore.

    Note: This reads the Excel file via pandas and wraps it in DataStore.

    Args:
        io: Path to the Excel file
        sheet_name: Sheet name or index to read (default 0)
        **kwargs: Additional arguments passed to pandas.read_excel()

    Returns:
        DataStore: A DataStore object containing the Excel data

    Example:
        >>> from datastore import read_excel
        >>> df = read_excel("data.xlsx")
        >>> df = read_excel("data.xlsx", sheet_name="Sheet2")
    """
    DataStore = _get_datastore_class()
    pandas_df = pd.read_excel(io, sheet_name=sheet_name, **kwargs)
    return DataStore.from_df(pandas_df)


def read_sql(sql, con, **kwargs) -> 'DataStoreType':
    """
    Read SQL query into DataStore.

    Note: This executes the SQL via pandas and wraps it in DataStore.

    Args:
        sql: SQL query string or table name
        con: Database connection (SQLAlchemy engine, connection string, etc.)
        **kwargs: Additional arguments passed to pandas.read_sql()

    Returns:
        DataStore: A DataStore object containing the SQL result

    Example:
        >>> from datastore import read_sql
        >>> df = read_sql("SELECT * FROM users", engine)
    """
    DataStore = _get_datastore_class()
    pandas_df = pd.read_sql(sql, con, **kwargs)
    return DataStore.from_df(pandas_df)


def read_table(filepath_or_buffer, sep='\t', **kwargs) -> 'DataStoreType':
    """
    Read general delimited file into DataStore.

    This is similar to read_csv but with tab ('\\t') as the default delimiter.
    Automatically chooses the optimal execution engine (chDB SQL or pandas).

    Args:
        filepath_or_buffer: Path to the file or file-like object
        sep: Delimiter to use (default '\\t' for tab)
        **kwargs: Additional arguments passed to pandas.read_table()
            - header: Row number to use as column names
            - names: List of column names to use
            - usecols: Return a subset of columns
            - dtype: Data type for columns
            - skiprows: Number of rows to skip
            - nrows: Number of rows to read
            - na_values: Additional strings to recognize as NA/NaN
            - encoding: Encoding to use for reading

    Returns:
        DataStore: A DataStore object containing the data

    Example:
        >>> from datastore import read_table
        >>> df = read_table("data.tsv")  # Tab-separated, uses SQL engine
        >>> df = read_table("data.txt", sep="|")  # Pipe-separated, uses SQL engine
        >>> df = read_table("data.tsv", dtype={'col': int})  # Uses pandas
    """
    # Delegate to read_csv with tab as default separator
    # read_csv handles automatic engine selection
    return read_csv(filepath_or_buffer, sep=sep, **kwargs)


def read_feather(path, columns=None, **kwargs) -> 'DataStoreType':
    """
    Read a Feather file into DataStore.

    This function automatically chooses the optimal execution engine:
    - Uses chDB SQL engine when possible (enables full SQL compilation)
    - Falls back to pandas when advanced pandas-only features are used

    Args:
        path: Path to the Feather file
        columns: List of column names to read (None reads all columns)
        **kwargs: Additional arguments passed to pandas.read_feather()

    Returns:
        DataStore: A DataStore object containing the Feather data

    Example:
        >>> from datastore import read_feather
        >>> df = read_feather("data.feather")  # Uses SQL engine
        >>> df = read_feather("data.feather", columns=['a', 'b'])  # Uses pandas
    """
    DataStore = _get_datastore_class()

    # Parameters that require pandas
    pandas_only_params = {
        'use_threads',  # Threading control
        'storage_options',  # Storage options
        'dtype_backend',  # Dtype backend selection
    }

    needs_pandas = any(param in kwargs for param in pandas_only_params)

    # columns parameter requires pandas (chDB reads all columns)
    if columns is not None:
        needs_pandas = True

    # File-like objects need pandas
    if hasattr(path, 'read'):
        needs_pandas = True

    if needs_pandas:
        pandas_df = pd.read_feather(path, columns=columns, **kwargs)
        return DataStore.from_df(pandas_df)
    else:
        # Use chDB SQL engine (Arrow format)
        return DataStore.from_file(path, format='Arrow')


def read_orc(path, columns=None, **kwargs) -> 'DataStoreType':
    """
    Read an ORC file into DataStore.

    This function automatically chooses the optimal execution engine:
    - Uses chDB SQL engine when possible (enables full SQL compilation)
    - Falls back to pandas when advanced pandas-only features are used

    Args:
        path: Path to the ORC file
        columns: List of column names to read (None reads all columns)
        **kwargs: Additional pandas.read_orc() arguments

    Returns:
        DataStore: A DataStore object containing the ORC data

    Example:
        >>> from datastore import read_orc
        >>> df = read_orc("data.orc")  # Uses SQL engine
        >>> df = read_orc("data.orc", columns=['name', 'age'])  # Uses pandas
    """
    DataStore = _get_datastore_class()

    # Parameters that require pandas
    pandas_only_params = {
        'dtype_backend',  # Dtype backend selection
        'filesystem',  # Custom filesystem
    }

    needs_pandas = any(param in kwargs for param in pandas_only_params)

    # columns parameter requires pandas (chDB reads all columns)
    if columns is not None:
        needs_pandas = True

    # File-like objects need pandas
    if hasattr(path, 'read'):
        needs_pandas = True

    if needs_pandas:
        pandas_df = pd.read_orc(path, columns=columns, **kwargs)
        return DataStore.from_df(pandas_df)
    else:
        # Use chDB SQL engine
        return DataStore.from_file(path, format='ORC')


def read_pickle(filepath_or_buffer, **kwargs) -> 'DataStoreType':
    """
    Read a pickled pandas DataFrame into DataStore.

    Note: This reads via pandas and wraps in DataStore.

    Args:
        filepath_or_buffer: Path to the pickle file
        **kwargs: Additional arguments passed to pandas.read_pickle()

    Returns:
        DataStore: A DataStore object containing the pickled data

    Example:
        >>> from datastore import read_pickle
        >>> df = read_pickle("data.pkl")
    """
    DataStore = _get_datastore_class()
    pandas_df = pd.read_pickle(filepath_or_buffer, **kwargs)
    return DataStore.from_df(pandas_df)


def read_fwf(filepath_or_buffer, colspecs='infer', widths=None, **kwargs) -> 'DataStoreType':
    """
    Read a table of fixed-width formatted lines into DataStore.

    Args:
        filepath_or_buffer: Path to file or file-like object
        colspecs: List of column edge pairs, or 'infer'
        widths: List of field widths
        **kwargs: Additional pandas.read_fwf() arguments

    Returns:
        DataStore: Parsed data

    Example:
        >>> import datastore as ds
        >>> df = ds.read_fwf("data.txt", widths=[10, 10, 5])
    """
    DataStore = _get_datastore_class()
    pandas_df = pd.read_fwf(filepath_or_buffer, colspecs=colspecs, widths=widths, **kwargs)
    return DataStore.from_df(pandas_df)


def read_html(
    io, match='.+', flavor=None, header=None, index_col=None, skiprows=None, attrs=None, parse_dates=False, **kwargs
):
    """
    Read HTML tables into a list of DataStore objects.

    Args:
        io: URL, file path, or file-like containing HTML
        match: Regex to match table
        flavor: Parsing engine ('lxml', 'bs4', 'html5lib')
        header: Row(s) to use as column headers
        index_col: Column(s) to use as row labels
        skiprows: Rows to skip
        attrs: Dict of table attributes to match
        parse_dates: Parse dates
        **kwargs: Additional pandas.read_html() arguments

    Returns:
        list of DataStore: One DataStore per table found

    Example:
        >>> import datastore as ds
        >>> tables = ds.read_html("https://example.com/table.html")
    """
    DataStore = _get_datastore_class()
    dfs = pd.read_html(
        io,
        match=match,
        flavor=flavor,
        header=header,
        index_col=index_col,
        skiprows=skiprows,
        attrs=attrs,
        parse_dates=parse_dates,
        **kwargs,
    )
    return [DataStore.from_df(df) for df in dfs]


def read_xml(
    path_or_buffer, xpath='./*', namespaces=None, elems_only=False, attrs_only=False, **kwargs
) -> 'DataStoreType':
    """
    Read XML document into DataStore.

    Args:
        path_or_buffer: Path to XML file or file-like object
        xpath: XPath to parse
        namespaces: Dict of namespace prefixes to URIs
        elems_only: Parse only elements
        attrs_only: Parse only attributes
        **kwargs: Additional pandas.read_xml() arguments

    Returns:
        DataStore: Parsed XML data

    Example:
        >>> import datastore as ds
        >>> df = ds.read_xml("data.xml")
    """
    DataStore = _get_datastore_class()
    pandas_df = pd.read_xml(
        path_or_buffer, xpath=xpath, namespaces=namespaces, elems_only=elems_only, attrs_only=attrs_only, **kwargs
    )
    return DataStore.from_df(pandas_df)


def read_stata(filepath_or_buffer, **kwargs) -> 'DataStoreType':
    """
    Read Stata file into DataStore.

    Args:
        filepath_or_buffer: Path to .dta file
        **kwargs: Additional pandas.read_stata() arguments

    Returns:
        DataStore: Stata data

    Example:
        >>> import datastore as ds
        >>> df = ds.read_stata("data.dta")
    """
    DataStore = _get_datastore_class()
    pandas_df = pd.read_stata(filepath_or_buffer, **kwargs)
    return DataStore.from_df(pandas_df)


def read_sas(filepath_or_buffer, format=None, index=None, encoding=None, **kwargs) -> 'DataStoreType':
    """
    Read SAS file into DataStore.

    Args:
        filepath_or_buffer: Path to SAS file
        format: 'xport' or 'sas7bdat' (auto-detected if None)
        index: Column to use as index
        encoding: Encoding for text data
        **kwargs: Additional pandas.read_sas() arguments

    Returns:
        DataStore: SAS data

    Example:
        >>> import datastore as ds
        >>> df = ds.read_sas("data.sas7bdat")
    """
    DataStore = _get_datastore_class()
    pandas_df = pd.read_sas(filepath_or_buffer, format=format, index=index, encoding=encoding, **kwargs)
    return DataStore.from_df(pandas_df)


def read_spss(path, usecols=None, convert_categoricals=True, dtype_backend=None) -> 'DataStoreType':
    """
    Read SPSS file into DataStore.

    Args:
        path: Path to SPSS file
        usecols: Columns to read
        convert_categoricals: Convert categorical columns
        dtype_backend: Backend data type

    Returns:
        DataStore: SPSS data

    Example:
        >>> import datastore as ds
        >>> df = ds.read_spss("data.sav")
    """
    DataStore = _get_datastore_class()
    pandas_df = pd.read_spss(
        path, usecols=usecols, convert_categoricals=convert_categoricals, dtype_backend=dtype_backend
    )
    return DataStore.from_df(pandas_df)


def read_hdf(path_or_buf, key=None, mode='r', **kwargs) -> 'DataStoreType':
    """
    Read HDF5 file into DataStore.

    Args:
        path_or_buf: Path to HDF5 file
        key: Identifier for group in store
        mode: Mode to open file
        **kwargs: Additional pandas.read_hdf() arguments

    Returns:
        DataStore: HDF5 data

    Example:
        >>> import datastore as ds
        >>> df = ds.read_hdf("data.h5", key="df")
    """
    DataStore = _get_datastore_class()
    pandas_df = pd.read_hdf(path_or_buf, key=key, mode=mode, **kwargs)
    return DataStore.from_df(pandas_df)


def read_sql_query(
    sql, con, index_col=None, coerce_float=True, params=None, parse_dates=None, chunksize=None, dtype=None, **kwargs
) -> 'DataStoreType':
    """
    Read SQL query into DataStore.

    Args:
        sql: SQL query string
        con: Database connection
        index_col: Column to use as index
        coerce_float: Convert values to float where possible
        params: Parameters to bind to query
        parse_dates: Columns to parse as dates
        chunksize: Rows per chunk
        dtype: Data types for columns
        **kwargs: Additional arguments

    Returns:
        DataStore: Query result

    Example:
        >>> import datastore as ds
        >>> df = ds.read_sql_query("SELECT * FROM users WHERE age > 18", engine)
    """
    DataStore = _get_datastore_class()
    pandas_df = pd.read_sql_query(
        sql,
        con,
        index_col=index_col,
        coerce_float=coerce_float,
        params=params,
        parse_dates=parse_dates,
        chunksize=chunksize,
        dtype=dtype,
        **kwargs,
    )
    return DataStore.from_df(pandas_df)


def read_sql_table(
    table_name,
    con,
    schema=None,
    index_col=None,
    coerce_float=True,
    parse_dates=None,
    columns=None,
    chunksize=None,
    **kwargs,
) -> 'DataStoreType':
    """
    Read SQL database table into DataStore.

    Args:
        table_name: Name of SQL table
        con: Database connection
        schema: Schema name
        index_col: Column to use as index
        coerce_float: Convert values to float where possible
        parse_dates: Columns to parse as dates
        columns: Columns to read
        chunksize: Rows per chunk
        **kwargs: Additional arguments

    Returns:
        DataStore: Table data

    Example:
        >>> import datastore as ds
        >>> df = ds.read_sql_table("users", engine)
    """
    DataStore = _get_datastore_class()
    pandas_df = pd.read_sql_table(
        table_name,
        con,
        schema=schema,
        index_col=index_col,
        coerce_float=coerce_float,
        parse_dates=parse_dates,
        columns=columns,
        chunksize=chunksize,
        **kwargs,
    )
    return DataStore.from_df(pandas_df)


# ========== Data Manipulation Functions ==========


def concat(objs, axis=0, join='outer', ignore_index=False, keys=None, **kwargs):
    """
    Concatenate DataStore/DataFrame objects along a particular axis.

    Args:
        objs: Sequence of DataStore or DataFrame objects to concatenate
        axis: The axis to concatenate along (default 0)
        join: How to handle indexes on other axis ('outer' or 'inner')
        ignore_index: If True, do not use index values along concatenation axis
        keys: Sequence to use as keys for hierarchical index
        **kwargs: Additional arguments passed to pandas.concat()

    Returns:
        DataStore if any input is a DataStore, otherwise pandas DataFrame.
        This ensures compatibility when pandas internal code calls concat.

    Example:
        >>> from datastore import concat
        >>> result = concat([df1, df2, df3])
        >>> result = concat([df1, df2], axis=1)
    """
    DataStore = _get_datastore_class()

    # Check if any input is a DataStore (vs plain pandas DataFrame)
    has_datastore = False
    dfs = []
    for obj in objs:
        if isinstance(obj, DataStore):
            has_datastore = True
            dfs.append(obj.to_df())
        elif hasattr(obj, 'to_df') and not isinstance(obj, _pd.DataFrame):
            has_datastore = True
            dfs.append(obj.to_df())
        else:
            dfs.append(obj)

    result = _pd.concat(dfs, axis=axis, join=join, ignore_index=ignore_index, keys=keys, **kwargs)

    # Only wrap in DataStore if the user passed DataStore objects
    # This allows pandas internal code to work correctly
    if has_datastore:
        return DataStore.from_df(result)
    return result


def merge(
    left,
    right,
    how='inner',
    on=None,
    left_on=None,
    right_on=None,
    left_index=False,
    right_index=False,
    suffixes=('_x', '_y'),
    **kwargs,
):
    """
    Merge DataStore/DataFrame objects with a database-style join.

    Args:
        left: Left DataStore or DataFrame
        right: Right DataStore or DataFrame
        how: Type of merge ('left', 'right', 'outer', 'inner', 'cross')
        on: Column or index level names to join on
        left_on: Column(s) from left to use as keys
        right_on: Column(s) from right to use as keys
        left_index: Use index from left as join key
        right_index: Use index from right as join key
        suffixes: Suffix to apply to overlapping columns
        **kwargs: Additional arguments passed to pandas.merge()

    Returns:
        DataStore if any input is a DataStore, otherwise pandas DataFrame.
        This ensures compatibility when pandas internal code calls merge.

    Example:
        >>> from datastore import merge
        >>> result = merge(df1, df2, on='id')
        >>> result = merge(df1, df2, left_on='user_id', right_on='id')
    """
    DataStore = _get_datastore_class()

    # Check if any input is a DataStore
    has_datastore = isinstance(left, DataStore) or isinstance(right, DataStore)

    # Convert to DataFrames
    if isinstance(left, DataStore):
        left_df = left.to_df()
    elif hasattr(left, 'to_df') and not isinstance(left, _pd.DataFrame):
        has_datastore = True
        left_df = left.to_df()
    else:
        left_df = left

    if isinstance(right, DataStore):
        right_df = right.to_df()
    elif hasattr(right, 'to_df') and not isinstance(right, _pd.DataFrame):
        has_datastore = True
        right_df = right.to_df()
    else:
        right_df = right

    result = _pd.merge(
        left_df,
        right_df,
        how=how,
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        suffixes=suffixes,
        **kwargs,
    )

    # Only wrap in DataStore if the user passed DataStore objects
    if has_datastore:
        return DataStore.from_df(result)
    return result


def merge_asof(
    left,
    right,
    on=None,
    left_on=None,
    right_on=None,
    left_index=False,
    right_index=False,
    by=None,
    left_by=None,
    right_by=None,
    suffixes=('_x', '_y'),
    tolerance=None,
    allow_exact_matches=True,
    direction='backward',
) -> 'DataStoreType':
    """
    Merge by nearest key rather than equal keys.

    This is useful for merging on sorted data, like time series.

    Args:
        left: Left DataFrame/DataStore
        right: Right DataFrame/DataStore
        on: Column name to join on (must be sorted)
        left_on: Column name in left to join on
        right_on: Column name in right to join on
        left_index: Use left index as join key
        right_index: Use right index as join key
        by: Column(s) to group by before merge
        left_by: Column(s) in left to group by
        right_by: Column(s) in right to group by
        suffixes: Suffixes for overlapping columns
        tolerance: Maximum distance for merge
        allow_exact_matches: Allow matching with equal values
        direction: 'backward', 'forward', 'nearest'

    Returns:
        DataStore: Merged data

    Example:
        >>> import datastore as ds
        >>> left = ds.DataStore.from_df({'a': [1, 5, 10], 'left_val': ['a', 'b', 'c']})
        >>> right = ds.DataStore.from_df({'a': [1, 2, 3, 6, 7], 'right_val': [1, 2, 3, 6, 7]})
        >>> ds.merge_asof(left, right, on='a')
    """
    DataStore = _get_datastore_class()
    left_df = left.to_df() if hasattr(left, 'to_df') else left
    right_df = right.to_df() if hasattr(right, 'to_df') else right

    result = pd.merge_asof(
        left_df,
        right_df,
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        by=by,
        left_by=left_by,
        right_by=right_by,
        suffixes=suffixes,
        tolerance=tolerance,
        allow_exact_matches=allow_exact_matches,
        direction=direction,
    )
    return DataStore.from_df(result)


def merge_ordered(
    left,
    right,
    on=None,
    left_on=None,
    right_on=None,
    left_by=None,
    right_by=None,
    fill_method=None,
    suffixes=('_x', '_y'),
    how='outer',
) -> 'DataStoreType':
    """
    Merge with optional filling/interpolation for ordered data.

    Args:
        left: Left DataFrame/DataStore
        right: Right DataFrame/DataStore
        on: Column name to join on
        left_on: Column name in left to join on
        right_on: Column name in right to join on
        left_by: Column(s) in left to group by
        right_by: Column(s) in right to group by
        fill_method: 'ffill' to forward-fill
        suffixes: Suffixes for overlapping columns
        how: 'left', 'right', 'outer', 'inner'

    Returns:
        DataStore: Merged data

    Example:
        >>> import datastore as ds
        >>> left = ds.DataStore.from_df({'key': ['a', 'c', 'e'], 'lvalue': [1, 2, 3]})
        >>> right = ds.DataStore.from_df({'key': ['b', 'c', 'd'], 'rvalue': [4, 5, 6]})
        >>> ds.merge_ordered(left, right, on='key')
    """
    DataStore = _get_datastore_class()
    left_df = left.to_df() if hasattr(left, 'to_df') else left
    right_df = right.to_df() if hasattr(right, 'to_df') else right

    result = pd.merge_ordered(
        left_df,
        right_df,
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_by=left_by,
        right_by=right_by,
        fill_method=fill_method,
        suffixes=suffixes,
        how=how,
    )
    return DataStore.from_df(result)


# ========== Missing Value Functions ==========


def isna(obj):
    """
    Detect missing values for an array-like object or scalar.

    This function takes a scalar or array-like object and indicates
    whether values are missing (NaN in numeric arrays, None or NaN
    in object arrays, NaT in datetime-like).

    Args:
        obj: Scalar or array-like object to check for missing values

    Returns:
        bool or array-like of bool: Boolean value or array indicating missing values

    Example:
        >>> import datastore as ds
        >>> ds.isna(None)
        True
        >>> ds.isna(float('nan'))
        True
        >>> ds.isna(1)
        False
    """
    return pd.isna(obj)


def isnull(obj):
    """
    Detect missing values for an array-like object or scalar.

    This is an alias for isna().

    Args:
        obj: Scalar or array-like object to check for missing values

    Returns:
        bool or array-like of bool: Boolean value or array indicating missing values

    Example:
        >>> import datastore as ds
        >>> ds.isnull(None)
        True
    """
    return pd.isnull(obj)


def notna(obj):
    """
    Detect non-missing values for an array-like object or scalar.

    This function is the boolean inverse of isna().

    Args:
        obj: Scalar or array-like object to check for non-missing values

    Returns:
        bool or array-like of bool: Boolean value or array indicating non-missing values

    Example:
        >>> import datastore as ds
        >>> ds.notna(1)
        True
        >>> ds.notna(None)
        False
    """
    return pd.notna(obj)


def notnull(obj):
    """
    Detect non-missing values for an array-like object or scalar.

    This is an alias for notna().

    Args:
        obj: Scalar or array-like object to check for non-missing values

    Returns:
        bool or array-like of bool: Boolean value or array indicating non-missing values

    Example:
        >>> import datastore as ds
        >>> ds.notnull(1)
        True
    """
    return pd.notnull(obj)


# ========== Type Conversion Functions ==========


def to_datetime(arg, errors='raise', dayfirst=False, yearfirst=False, utc=False, format=None, **kwargs):
    """
    Convert argument to datetime.

    This function converts a scalar, array-like, Series or DataFrame/dict-like
    to a pandas datetime object.

    Args:
        arg: Object to convert to datetime
        errors: 'raise', 'coerce', 'ignore' - handling of errors
        dayfirst: If True, parse dates with day first (e.g., 10/11/12 -> Nov 10)
        yearfirst: If True, parse dates with year first
        utc: If True, return UTC DatetimeIndex
        format: strftime format string for parsing
        **kwargs: Additional arguments passed to pandas.to_datetime()

    Returns:
        datetime64, DatetimeIndex, or Series of datetime64

    Example:
        >>> import datastore as ds
        >>> ds.to_datetime('2024-01-15')
        Timestamp('2024-01-15 00:00:00')
        >>> ds.to_datetime(['2024-01-15', '2024-02-20'])
        DatetimeIndex(['2024-01-15', '2024-02-20'], dtype='datetime64[ns]', freq=None)
    """
    return pd.to_datetime(arg, errors=errors, dayfirst=dayfirst, yearfirst=yearfirst, utc=utc, format=format, **kwargs)


def to_numeric(arg, errors='raise', downcast=None):
    """
    Convert argument to a numeric type.

    Args:
        arg: Scalar, list, tuple, 1-d array, or Series to convert
        errors: 'raise', 'coerce', 'ignore' - handling of errors
        downcast: 'integer', 'signed', 'unsigned', 'float' - downcast dtype

    Returns:
        Numeric value or array of numeric values

    Example:
        >>> import datastore as ds
        >>> ds.to_numeric('1.5')
        1.5
        >>> ds.to_numeric(['1', '2', '3'])
        array([1., 2., 3.])
    """
    return pd.to_numeric(arg, errors=errors, downcast=downcast)


def to_timedelta(arg, unit=None, errors='raise'):
    """
    Convert argument to timedelta.

    Args:
        arg: String, timedelta, list-like, or Series to convert
        unit: Unit of the arg ('D', 'h', 'm', 's', 'ms', 'us', 'ns')
        errors: 'raise', 'coerce', 'ignore' - handling of errors

    Returns:
        Timedelta, TimedeltaIndex, or Series of timedelta64

    Example:
        >>> import datastore as ds
        >>> ds.to_timedelta('1 days')
        Timedelta('1 days 00:00:00')
        >>> ds.to_timedelta(1, unit='h')
        Timedelta('0 days 01:00:00')
    """
    return pd.to_timedelta(arg, unit=unit, errors=errors)


# ========== Date Range Functions ==========


def date_range(
    start=None, end=None, periods=None, freq=None, tz=None, normalize=False, name=None, inclusive='both', **kwargs
):
    """
    Return a fixed frequency DatetimeIndex.

    Args:
        start: Start date
        end: End date
        periods: Number of periods to generate
        freq: Frequency string (e.g., 'D', 'H', 'T', 'S')
        tz: Timezone name
        normalize: Normalize start/end dates to midnight
        name: Name of the resulting DatetimeIndex
        inclusive: 'both', 'neither', 'left', 'right'
        **kwargs: Additional arguments

    Returns:
        DatetimeIndex

    Example:
        >>> import datastore as ds
        >>> ds.date_range('2024-01-01', periods=3, freq='D')
        DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-03'], dtype='datetime64[ns]', freq='D')
    """
    return pd.date_range(
        start=start,
        end=end,
        periods=periods,
        freq=freq,
        tz=tz,
        normalize=normalize,
        name=name,
        inclusive=inclusive,
        **kwargs,
    )


def bdate_range(
    start=None,
    end=None,
    periods=None,
    freq='B',
    tz=None,
    normalize=True,
    name=None,
    weekmask=None,
    holidays=None,
    inclusive='both',
    **kwargs,
):
    """
    Return a fixed frequency DatetimeIndex with business day as default.

    Args:
        start: Start date
        end: End date
        periods: Number of periods
        freq: Frequency string (default 'B' for business day)
        tz: Timezone name
        normalize: Normalize start/end dates to midnight
        name: Name of the resulting DatetimeIndex
        weekmask: Valid business days
        holidays: Dates to exclude from calendar
        inclusive: 'both', 'neither', 'left', 'right'

    Returns:
        DatetimeIndex

    Example:
        >>> import datastore as ds
        >>> ds.bdate_range('2024-01-01', periods=5)
        DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'], ...)
    """
    return pd.bdate_range(
        start=start,
        end=end,
        periods=periods,
        freq=freq,
        tz=tz,
        normalize=normalize,
        name=name,
        weekmask=weekmask,
        holidays=holidays,
        inclusive=inclusive,
        **kwargs,
    )


def period_range(start=None, end=None, periods=None, freq=None, name=None):
    """
    Return a fixed frequency PeriodIndex.

    Args:
        start: Start period
        end: End period
        periods: Number of periods
        freq: Frequency string
        name: Name of the resulting PeriodIndex

    Returns:
        PeriodIndex

    Example:
        >>> import datastore as ds
        >>> ds.period_range('2024-01', periods=3, freq='M')
        PeriodIndex(['2024-01', '2024-02', '2024-03'], dtype='period[M]')
    """
    return pd.period_range(start=start, end=end, periods=periods, freq=freq, name=name)


def timedelta_range(start=None, end=None, periods=None, freq=None, name=None, closed=None):
    """
    Return a fixed frequency TimedeltaIndex.

    Args:
        start: Start timedelta
        end: End timedelta
        periods: Number of periods
        freq: Frequency string
        name: Name of the resulting TimedeltaIndex
        closed: 'left', 'right', or None

    Returns:
        TimedeltaIndex

    Example:
        >>> import datastore as ds
        >>> ds.timedelta_range('1 day', periods=3, freq='12h')
        TimedeltaIndex(['1 days 00:00:00', '1 days 12:00:00', '2 days 00:00:00'], ...)
    """
    return pd.timedelta_range(start=start, end=end, periods=periods, freq=freq, name=name, closed=closed)


def interval_range(start=None, end=None, periods=None, freq=None, name=None, closed='right'):
    """
    Return a fixed frequency IntervalIndex.

    Args:
        start: Start value
        end: End value
        periods: Number of periods
        freq: Numeric step or frequency string
        name: Name of the resulting IntervalIndex
        closed: 'left', 'right', 'both', 'neither'

    Returns:
        IntervalIndex

    Example:
        >>> import datastore as ds
        >>> ds.interval_range(start=0, end=5)
        IntervalIndex([(0, 1], (1, 2], (2, 3], (3, 4], (4, 5]], ...)
    """
    return pd.interval_range(start=start, end=end, periods=periods, freq=freq, name=name, closed=closed)


# ========== Data Binning and Categorization Functions ==========


def cut(
    x, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False, duplicates='raise', ordered=True
):
    """
    Bin values into discrete intervals.

    Use cut when you need to segment and sort data values into bins.
    This function is useful for going from a continuous variable to
    a categorical variable.

    Args:
        x: Input array to be binned
        bins: Number of bins or sequence of bin edges
        right: Include right edge of bin intervals
        labels: Labels for the returned bins
        retbins: Whether to return the bins
        precision: Precision for storing bins
        include_lowest: Include the lowest value in the first bin
        duplicates: 'raise', 'drop' - handling of duplicate bin edges
        ordered: Whether labels are ordered

    Returns:
        Categorical or Series; optionally (bins, labels) if retbins=True

    Example:
        >>> import datastore as ds
        >>> ds.cut([1, 7, 5, 4, 6, 3], 3)
        [(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0], (5.0, 7.0], (0.994, 3.0]]
    """
    return pd.cut(
        x,
        bins,
        right=right,
        labels=labels,
        retbins=retbins,
        precision=precision,
        include_lowest=include_lowest,
        duplicates=duplicates,
        ordered=ordered,
    )


def qcut(x, q, labels=None, retbins=False, precision=3, duplicates='raise'):
    """
    Quantile-based discretization function.

    Discretize variable into equal-sized buckets based on rank or
    based on sample quantiles.

    Args:
        x: Input array to be binned
        q: Number of quantiles or list of quantiles (e.g., [0, .25, .5, .75, 1.])
        labels: Labels for the returned bins
        retbins: Whether to return the bins
        precision: Precision for storing bins
        duplicates: 'raise', 'drop' - handling of duplicate bin edges

    Returns:
        Categorical or Series; optionally (bins, labels) if retbins=True

    Example:
        >>> import datastore as ds
        >>> ds.qcut(range(10), 4)  # Quartiles
        [(-0.001, 2.25], (-0.001, 2.25], (-0.001, 2.25], (2.25, 4.5], ...]
    """
    return pd.qcut(x, q, labels=labels, retbins=retbins, precision=precision, duplicates=duplicates)


def get_dummies(
    data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None
):
    """
    Convert categorical variable into dummy/indicator variables.

    Each unique value becomes a column with binary (0/1) values.

    Args:
        data: Data of which to get dummy indicators
        prefix: String to append DataFrame column names
        prefix_sep: Separator between prefix and column name
        dummy_na: Add a column to indicate NaNs
        columns: Column names to encode (if DataFrame)
        sparse: Return SparseDataFrame
        drop_first: Remove first level to avoid collinearity
        dtype: Data type for new columns

    Returns:
        DataFrame with dummy variables

    Example:
        >>> import datastore as ds
        >>> ds.get_dummies(['a', 'b', 'a'])
           a  b
        0  1  0
        1  0  1
        2  1  0
    """
    DataStore = _get_datastore_class()

    # Convert DataStore to DataFrame (pandas uses isinstance check internally)
    if hasattr(data, 'to_df'):
        data = data.to_df()

    result = pd.get_dummies(
        data,
        prefix=prefix,
        prefix_sep=prefix_sep,
        dummy_na=dummy_na,
        columns=columns,
        sparse=sparse,
        drop_first=drop_first,
        dtype=dtype,
    )
    return DataStore.from_df(result)


def factorize(values, sort=False, use_na_sentinel=True, size_hint=None):
    """
    Encode the object as an enumerated type or categorical variable.

    Args:
        values: Sequence to encode
        sort: Sort by values
        use_na_sentinel: Use -1 sentinel for NaN values
        size_hint: Hint for hashtable size

    Returns:
        codes: Integer codes
        uniques: Unique values

    Example:
        >>> import datastore as ds
        >>> codes, uniques = ds.factorize(['b', 'a', 'c', 'b'])
        >>> codes
        array([0, 1, 2, 0])
        >>> uniques
        array(['b', 'a', 'c'], dtype=object)
    """
    return pd.factorize(values, sort=sort, use_na_sentinel=use_na_sentinel, size_hint=size_hint)


def unique(values):
    """
    Return unique values based on a hash table.

    Args:
        values: 1d array-like

    Returns:
        ndarray or ExtensionArray of unique values

    Example:
        >>> import datastore as ds
        >>> ds.unique([1, 2, 2, 3, 1])
        array([1, 2, 3])
    """
    return pd.unique(values)


def value_counts(values, sort=True, ascending=False, normalize=False, bins=None, dropna=True):
    """
    Return a Series containing counts of unique values.

    Args:
        values: 1d array-like
        sort: Sort by frequency
        ascending: Sort in ascending order
        normalize: Return proportions instead of counts
        bins: Group into bins (numeric data only)
        dropna: Don't include NaN counts

    Returns:
        Series with counts/proportions

    Example:
        >>> import datastore as ds
        >>> ds.value_counts(['a', 'b', 'a', 'a'])
        a    3
        b    1
        dtype: int64
    """
    return pd.value_counts(values, sort=sort, ascending=ascending, normalize=normalize, bins=bins, dropna=dropna)


# ========== Reshaping Functions ==========


def melt(
    frame, id_vars=None, value_vars=None, var_name=None, value_name='value', col_level=None, ignore_index=True
) -> 'DataStoreType':
    """
    Unpivot a DataFrame from wide to long format.

    Args:
        frame: DataFrame or DataStore to melt
        id_vars: Columns to use as identifier variables
        value_vars: Columns to unpivot
        var_name: Name for the variable column
        value_name: Name for the value column
        col_level: If columns are MultiIndex, level to melt
        ignore_index: If True, original index is ignored

    Returns:
        DataStore: Unpivoted data

    Example:
        >>> import datastore as ds
        >>> df = ds.DataStore.from_df({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        >>> ds.melt(df, id_vars=['A'], value_vars=['B', 'C'])
    """
    DataStore = _get_datastore_class()
    if hasattr(frame, 'to_df'):
        frame = frame.to_df()
    result = pd.melt(
        frame,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
        col_level=col_level,
        ignore_index=ignore_index,
    )
    return DataStore.from_df(result)


def pivot(data, *, columns, index=None, values=None) -> 'DataStoreType':
    """
    Return reshaped DataFrame organized by given index / column values.

    Args:
        data: DataFrame or DataStore
        columns: Column(s) to use for new frame's columns
        index: Column(s) to use for new frame's index
        values: Column(s) to use for populating new frame's values

    Returns:
        DataStore: Reshaped data

    Example:
        >>> import datastore as ds
        >>> df = ds.DataStore.from_df({'A': ['foo', 'foo', 'bar'], 'B': ['one', 'two', 'one'], 'C': [1, 2, 3]})
        >>> ds.pivot(df, columns='B', index='A', values='C')
    """
    DataStore = _get_datastore_class()
    if hasattr(data, 'to_df'):
        data = data.to_df()
    result = pd.pivot(data, columns=columns, index=index, values=values)
    return DataStore.from_df(result)


def pivot_table(
    data,
    values=None,
    index=None,
    columns=None,
    aggfunc='mean',
    fill_value=None,
    margins=False,
    dropna=True,
    margins_name='All',
    observed=False,
    sort=True,
) -> 'DataStoreType':
    """
    Create a spreadsheet-style pivot table as a DataFrame.

    Args:
        data: DataFrame or DataStore
        values: Column(s) to aggregate
        index: Column(s) for pivot table index
        columns: Column(s) for pivot table columns
        aggfunc: Aggregation function(s)
        fill_value: Value to replace missing values
        margins: Add row/column margins (subtotals)
        dropna: Do not include columns with all NaN entries
        margins_name: Name of the row/column for margins
        observed: Only use observed values for categorical groupers
        sort: Sort result

    Returns:
        DataStore: Pivot table

    Example:
        >>> import datastore as ds
        >>> ds.pivot_table(df, values='D', index=['A'], columns=['C'], aggfunc='sum')
    """
    DataStore = _get_datastore_class()
    if hasattr(data, 'to_df'):
        data = data.to_df()
    result = pd.pivot_table(
        data,
        values=values,
        index=index,
        columns=columns,
        aggfunc=aggfunc,
        fill_value=fill_value,
        margins=margins,
        dropna=dropna,
        margins_name=margins_name,
        observed=observed,
        sort=sort,
    )
    return DataStore.from_df(result)


def crosstab(
    index,
    columns,
    values=None,
    rownames=None,
    colnames=None,
    aggfunc=None,
    margins=False,
    margins_name='All',
    dropna=True,
    normalize=False,
) -> 'DataStoreType':
    """
    Compute a simple cross tabulation of two (or more) factors.

    Args:
        index: Values to group by in rows
        columns: Values to group by in columns
        values: Array of values to aggregate
        rownames: Names for row grouping levels
        colnames: Names for column grouping levels
        aggfunc: Aggregation function
        margins: Add row/column margins
        margins_name: Name for margins
        dropna: Do not include NaN values
        normalize: Normalize by row/column/all

    Returns:
        DataStore: Cross tabulation

    Example:
        >>> import datastore as ds
        >>> a = ['foo', 'foo', 'bar', 'bar']
        >>> b = ['one', 'two', 'one', 'two']
        >>> ds.crosstab(a, b)
    """
    DataStore = _get_datastore_class()
    result = pd.crosstab(
        index,
        columns,
        values=values,
        rownames=rownames,
        colnames=colnames,
        aggfunc=aggfunc,
        margins=margins,
        margins_name=margins_name,
        dropna=dropna,
        normalize=normalize,
    )
    return DataStore.from_df(result)


def wide_to_long(df, stubnames, i, j, sep='', suffix=r'\d+') -> 'DataStoreType':
    """
    Unpivot a DataFrame from wide to long format.

    Args:
        df: DataFrame or DataStore
        stubnames: Stub names (prefix before numeric suffix)
        i: Column(s) to use as id variables
        j: Name of the sub-observation variable
        sep: Separator between stub and suffix
        suffix: Regex pattern for suffix

    Returns:
        DataStore: Reshaped data

    Example:
        >>> import datastore as ds
        >>> df = ds.DataStore.from_df({'A1970': [1], 'A1980': [2], 'B1970': [3], 'B1980': [4], 'id': [0]})
        >>> ds.wide_to_long(df, stubnames=['A', 'B'], i='id', j='year')
    """
    DataStore = _get_datastore_class()
    if hasattr(df, 'to_df'):
        df = df.to_df()
    result = pd.wide_to_long(df, stubnames, i, j, sep=sep, suffix=suffix)
    return DataStore.from_df(result)


# ========== Utility Functions ==========


def infer_freq(index):
    """
    Infer the most likely frequency given the input index.

    Args:
        index: DatetimeIndex or TimedeltaIndex

    Returns:
        str or None: Inferred frequency string

    Example:
        >>> import datastore as ds
        >>> idx = ds.date_range('2024-01-01', periods=5, freq='D')
        >>> ds.infer_freq(idx)
        'D'
    """
    return pd.infer_freq(index)


def json_normalize(
    data, record_path=None, meta=None, meta_prefix=None, record_prefix=None, errors='raise', sep='.', max_level=None
) -> 'DataStoreType':
    """
    Normalize semi-structured JSON data into a flat table.

    Args:
        data: Nested dict or list of dicts
        record_path: Path to list of records
        meta: Fields to use as metadata
        meta_prefix: Prefix for metadata columns
        record_prefix: Prefix for record columns
        errors: 'raise' or 'ignore'
        sep: Separator for nested column names
        max_level: Max depth to normalize

    Returns:
        DataStore: Normalized data

    Example:
        >>> import datastore as ds
        >>> data = [{'id': 1, 'info': {'name': 'Alice', 'age': 30}}]
        >>> ds.json_normalize(data)
    """
    DataStore = _get_datastore_class()
    result = pd.json_normalize(
        data,
        record_path=record_path,
        meta=meta,
        meta_prefix=meta_prefix,
        record_prefix=record_prefix,
        errors=errors,
        sep=sep,
        max_level=max_level,
    )
    return DataStore.from_df(result)


# ========== Configuration Functions ==========


def set_option(pat, value):
    """
    Set the value of a pandas option.

    Args:
        pat: Option name pattern
        value: New value for the option

    Example:
        >>> import datastore as ds
        >>> ds.set_option('display.max_rows', 100)
    """
    pd.set_option(pat, value)


def get_option(pat):
    """
    Get the value of a pandas option.

    Args:
        pat: Option name pattern

    Returns:
        Value of the option

    Example:
        >>> import datastore as ds
        >>> ds.get_option('display.max_rows')
        60
    """
    return pd.get_option(pat)


def reset_option(pat):
    """
    Reset one or more options to their default value.

    Args:
        pat: Option name pattern

    Example:
        >>> import datastore as ds
        >>> ds.reset_option('display.max_rows')
    """
    pd.reset_option(pat)


def describe_option(pat='', _print_desc=True):
    """
    Print the description of one or more options.

    Args:
        pat: Option name pattern (empty string for all)
        _print_desc: Whether to print description

    Returns:
        str or None: Description if _print_desc is False

    Example:
        >>> import datastore as ds
        >>> ds.describe_option('display.max_rows')
    """
    return pd.describe_option(pat, _print_desc=_print_desc)


def option_context(*args):
    """
    Context manager to temporarily set options.

    Args:
        *args: Pairs of (option_name, value)

    Returns:
        Context manager

    Example:
        >>> import datastore as ds
        >>> with ds.option_context('display.max_rows', 10):
        ...     print(df)
    """
    return pd.option_context(*args)


def show_versions(as_json=False):
    """
    Print various dependency versions.

    Args:
        as_json: If True, return JSON string instead of printing

    Example:
        >>> import datastore as ds
        >>> ds.show_versions()
    """
    return pd.show_versions(as_json=as_json)


# ========== Array Creation ==========


def array(data, dtype=None, copy=True):
    """
    Create a pandas ExtensionArray.

    Args:
        data: Data to create array from
        dtype: Optional dtype
        copy: Copy data if True

    Returns:
        ExtensionArray

    Example:
        >>> import datastore as ds
        >>> ds.array([1, 2, 3])
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
    """
    return pd.array(data, dtype=dtype, copy=copy)


# ========== DataFrame/Series Classes ==========
# These factory functions create DataStore objects, providing a pandas-like API.
# For monkey-patching compatibility (sys.modules['pandas'] = datastore), use PandasDataFrame below.


def DataFrame(data=None, index=None, columns=None, dtype=None, copy=None):
    """
    Create a DataStore from data (pandas DataFrame-like factory).

    This is the primary way to create a DataStore with a pandas-like API.
    Returns a DataStore object that supports both pandas-style and SQL-style operations.

    Args:
        data: Dict, list, ndarray, Iterable, or DataFrame
        index: Index or array-like for row labels
        columns: Column labels for the result
        dtype: Data type to force
        copy: Copy data from inputs

    Returns:
        DataStore: A DataStore object

    Example:
        >>> import datastore as ds
        >>> df = ds.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> df['A'].sum()  # Returns lazy result
    """
    DataStore = _get_datastore_class()
    pandas_df = _pd.DataFrame(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
    return DataStore.from_df(pandas_df)


# Alias for backward compatibility
make_datastore = DataFrame


def Series(data=None, index=None, dtype=None, name=None, copy=None):
    """
    Create a pandas Series.

    This is a pass-through to pandas Series for compatibility.
    For column operations, use DataStore column access (e.g., ds['column']).

    Args:
        data: Array-like, Iterable, dict, or scalar value
        index: Values must be hashable and have the same length as data
        dtype: Data type for the output Series
        name: The name to give to the Series
        copy: Copy input data

    Returns:
        pandas Series

    Example:
        >>> import datastore as ds
        >>> s = ds.Series([1, 2, 3], name='values')
    """
    return _pd.Series(data=data, index=index, dtype=dtype, name=name, copy=copy)


# ========== Pandas Core Types (Re-exported for convenience) ==========

# Index types
Index = _pd.Index
MultiIndex = _pd.MultiIndex
RangeIndex = _pd.RangeIndex
DatetimeIndex = _pd.DatetimeIndex
TimedeltaIndex = _pd.TimedeltaIndex
PeriodIndex = _pd.PeriodIndex
IntervalIndex = _pd.IntervalIndex
CategoricalIndex = _pd.CategoricalIndex

# Scalar types
Timestamp = _pd.Timestamp
Timedelta = _pd.Timedelta
Period = _pd.Period
Interval = _pd.Interval

# Data types
Categorical = _pd.Categorical
CategoricalDtype = _pd.CategoricalDtype
DatetimeTZDtype = _pd.DatetimeTZDtype
IntervalDtype = _pd.IntervalDtype
PeriodDtype = _pd.PeriodDtype
SparseDtype = _pd.SparseDtype
StringDtype = _pd.StringDtype
BooleanDtype = _pd.BooleanDtype
Int8Dtype = _pd.Int8Dtype
Int16Dtype = _pd.Int16Dtype
Int32Dtype = _pd.Int32Dtype
Int64Dtype = _pd.Int64Dtype
UInt8Dtype = _pd.UInt8Dtype
UInt16Dtype = _pd.UInt16Dtype
UInt32Dtype = _pd.UInt32Dtype
UInt64Dtype = _pd.UInt64Dtype
Float32Dtype = _pd.Float32Dtype
Float64Dtype = _pd.Float64Dtype

# NA handling
NA = _pd.NA
NaT = _pd.NaT

# Grouper
Grouper = _pd.Grouper

# NamedAgg for named aggregation
NamedAgg = _pd.NamedAgg

# ========== Module exports ==========

__all__ = [
    # DataFrame/Series Creation
    'DataFrame',
    'Series',
    'make_datastore',  # Explicit factory for creating DataStore objects
    # Pandas Core Types (re-exported)
    'Index',
    'MultiIndex',
    'RangeIndex',
    'DatetimeIndex',
    'TimedeltaIndex',
    'PeriodIndex',
    'IntervalIndex',
    'CategoricalIndex',
    'Timestamp',
    'Timedelta',
    'Period',
    'Interval',
    'Categorical',
    'CategoricalDtype',
    'DatetimeTZDtype',
    'IntervalDtype',
    'PeriodDtype',
    'SparseDtype',
    'StringDtype',
    'BooleanDtype',
    'Int8Dtype',
    'Int16Dtype',
    'Int32Dtype',
    'Int64Dtype',
    'UInt8Dtype',
    'UInt16Dtype',
    'UInt32Dtype',
    'UInt64Dtype',
    'Float32Dtype',
    'Float64Dtype',
    'NA',
    'NaT',
    'Grouper',
    'NamedAgg',
    # IO Functions
    'read_csv',
    'read_parquet',
    'read_json',
    'read_excel',
    'read_sql',
    'read_table',
    'read_feather',
    'read_orc',
    'read_pickle',
    'read_fwf',
    'read_html',
    'read_xml',
    'read_stata',
    'read_sas',
    'read_spss',
    'read_hdf',
    'read_sql_query',
    'read_sql_table',
    # Data Manipulation Functions
    'concat',
    'merge',
    'merge_asof',
    'merge_ordered',
    # Missing Value Functions
    'isna',
    'isnull',
    'notna',
    'notnull',
    # Type Conversion Functions
    'to_datetime',
    'to_numeric',
    'to_timedelta',
    # Date Range Functions
    'date_range',
    'bdate_range',
    'period_range',
    'timedelta_range',
    'interval_range',
    # Data Binning and Categorization Functions
    'cut',
    'qcut',
    'get_dummies',
    'factorize',
    'unique',
    'value_counts',
    # Reshaping Functions
    'melt',
    'pivot',
    'pivot_table',
    'crosstab',
    'wide_to_long',
    # Utility Functions
    'infer_freq',
    'json_normalize',
    # Configuration Functions
    'set_option',
    'get_option',
    'reset_option',
    'describe_option',
    'option_context',
    'show_versions',
    # Array Creation
    'array',
]
