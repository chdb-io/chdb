import sys
import os

chdb_version = (0, 1, 0)
if sys.version_info[:2] >= (3, 6):
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
    raise NotImplementedError("Python 3.5 or lower version is not supported")

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = ".".join(map(str, chdb_version))
except:  # pragma: no cover
    __version__ = "unknown"
