import sys
import os
import pyarrow as pa

chdb_version = (0, 1, 0)
if sys.version_info[:2] >= (3, 7):
    # get the path of the current file
    current_path = os.path.dirname(os.path.abspath(__file__))
    # change the current working directory to the path of the current file
    # and import _chdb then change the working directory back
    cwd = os.getcwd()
    os.chdir(current_path)
    from . import _chdb  # noqa
    os.chdir(cwd)
    engine_version = str(_chdb.query("SELECT version();", "CSV").get_memview().tobytes())[3:-4]
else:
    raise NotImplementedError("Python 3.6 or lower version is not supported")

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = ".".join(map(str, chdb_version))
except:  # pragma: no cover
    __version__ = "unknown"


def _to_arrowTable(res):
    """convert res to arrow table"""
    return pa.RecordBatchFileReader(res.get_memview()).read_all()

def to_df(r):
    """"convert arrow table to Dataframe"""
    t = _to_arrowTable(r)
    return t.to_pandas(use_threads=True)

# wrap _chdb functions
def query(sql, output_format="CSV", **kwargs):
    if output_format.lower() == "dataframe":
        r = _chdb.query(sql, "Arrow", **kwargs)
        return to_df(r)
    return _chdb.query(sql, output_format, **kwargs)
