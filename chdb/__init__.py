import sys
import os


class ChdbError(Exception):
    """Base class for exceptions in this module."""


_arrow_format = set({"dataframe", "arrowtable"})
_process_result_format_funs = {
    "dataframe": lambda x: to_df(x),
    "arrowtable": lambda x: to_arrowTable(x),
}

# If any UDF is defined, the path of the UDF will be set to this variable
# and the path will be deleted when the process exits
# UDF config path will be f"{g_udf_path}/udf_config.xml"
# UDF script path will be f"{g_udf_path}/{func_name}.py"
g_udf_path = ""

chdb_version = ("0", "6", "0")
if sys.version_info[:2] >= (3, 7):
    # get the path of the current file
    current_path = os.path.dirname(os.path.abspath(__file__))
    # change the current working directory to the path of the current file
    # and import _chdb then change the working directory back
    cwd = os.getcwd()
    os.chdir(current_path)
    from . import _chdb  # noqa

    os.chdir(cwd)
    engine_version = str(_chdb.query("SELECT version();", "CSV").bytes())[3:-4]
else:
    raise NotImplementedError("Python 3.6 or lower version is not supported")

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = ".".join(map(str, chdb_version))
except:  # noqa
    __version__ = "unknown"


# return pyarrow table
def to_arrowTable(res):
    """convert res to arrow table"""
    # try import pyarrow and pandas, if failed, raise ImportError with suggestion
    try:
        import pyarrow as pa  # noqa
        import pandas as pd  # noqa
    except ImportError as e:
        print(f"ImportError: {e}")
        print('Please install pyarrow and pandas via "pip install pyarrow pandas"')
        raise ImportError("Failed to import pyarrow or pandas") from None
    if len(res) == 0:
        return pa.Table.from_batches([], schema=pa.schema([]))
    return pa.RecordBatchFileReader(res.bytes()).read_all()


# return pandas dataframe
def to_df(r):
    """convert arrow table to Dataframe"""
    t = to_arrowTable(r)
    return t.to_pandas(use_threads=True)


# wrap _chdb functions
def query(sql, output_format="CSV", path="", udf_path=""):
    global g_udf_path
    if udf_path != "":
        g_udf_path = udf_path
    lower_output_format = output_format.lower()
    result_func = _process_result_format_funs.get(lower_output_format, lambda x: x)
    if lower_output_format in _arrow_format:
        output_format = "Arrow"
    res = _chdb.query(sql, output_format, path=path, udf_path=g_udf_path)
    if res.has_error():
        raise ChdbError(res.error_message())
    return result_func(res)


# alias for query
sql = query

PyReader = _chdb.PyReader

__all__ = [
    "PyReader",
    "ChdbError",
    "query",
    "sql",
    "chdb_version",
    "engine_version",
    "to_df",
    "to_arrowTable",
]
