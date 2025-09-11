# try import pyarrow and pandas, if failed, raise ImportError with suggestion
try:
    import pyarrow as pa  # noqa
    import pandas as pd  # noqa
except ImportError as e:
    print(f'ImportError: {e}')
    print('Please install pyarrow and pandas via "pip install pyarrow pandas"')
    raise ImportError('Failed to import pyarrow or pandas') from None

# check if pandas version >= 2.0.0
try:
    version_parts = pd.__version__.split('.')
    major_version = int(version_parts[0])
    if major_version < 2:
        print('Please upgrade pandas to version 2.0.0 or higher to have better performance')
except (ValueError, IndexError, AttributeError):
    pass

from .query import Table, pandas_read_parquet  # noqa: C0413

query = Table.queryStatic
sql = Table.queryStatic

__all__ = ["Table", "query", "sql", "pandas_read_parquet"]
