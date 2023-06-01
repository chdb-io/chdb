import os
import tempfile
import pandas as pd
import pyarrow as pa
from io import BytesIO
from chdb import query as chdb_query


class Table(object):
    """
    Table is a wrapper of multiple formats of data buffer, including parquet file path,
    parquet bytes, and pandas dataframe.
    """

    def __init__(self,
                 parquet_path: str = None,
                 temp_parquet_path: str = None,
                 parquet_memoryview: memoryview = None,
                 dataframe: pd.DataFrame = None,
                 arrow_table: pa.Table = None,
                 use_memfd: bool = False):
        """
        Initialize a Table object with one of parquet file path, parquet bytes, pandas dataframe or
        parquet table.
        """
        self._parquet_path = parquet_path
        self._temp_parquet_path = temp_parquet_path
        self._parquet_memoryview = parquet_memoryview
        self._dataframe = dataframe
        self._arrow_table = arrow_table
        self.use_memfd = use_memfd

    def __del__(self):
        try:
            if self._temp_parquet_path is not None:
                os.remove(self._temp_parquet_path)
        except:
            pass

    def to_pandas(self) -> pd.DataFrame:
        if self._dataframe is None:
            if self._arrow_table is not None:
                return self._arrow_table.to_pandas()
            elif self._parquet_memoryview is not None:
                # wrap bytes to ReadBuffer
                pq_reader = BytesIO(self._parquet_memoryview.tobytes())
                return pd.read_parquet(pq_reader, engine='pyarrow', dtype_backend='pyarrow')
            elif self._parquet_path is not None:
                return pd.read_parquet(self._parquet_path, engine='pyarrow', dtype_backend='pyarrow')
            elif self._temp_parquet_path is not None:
                return pd.read_parquet(self._temp_parquet_path, engine='pyarrow', dtype_backend='pyarrow')
            else:
                raise ValueError("No data buffer in Table object")
        return self._dataframe

    def __repr__(self):
        return repr(self.to_pandas())

    def __str__(self):
        return str(self.to_pandas())

    def query(self, sql, **kwargs):
        """
        Query on current Table object, return a new Table object.
        The `FROM` table name in SQL should always be `__table__`. eg:
            `SELECT * FROM __table__ WHERE ...`
        """
        # check if "__table__" is in sql
        if "__table__" not in sql:
            raise ValueError("SQL should always contain `FROM __table__`")

        if self._parquet_path is not None:  # if we have parquet file path, run chdb query on it directly is faster
            # replace "__table__" with file("self._parquet_path", Parquet)
            new_sql = sql.replace("__table__", f"file(\"{self._parquet_path}\", Parquet)")
            res = chdb_query(new_sql, "Parquet", **kwargs)
            return Table(parquet_memoryview=res.get_memview())
        elif self._temp_parquet_path is not None:
            # replace "__table__" with file("self._temp_parquet_path", Parquet)
            new_sql = sql.replace("__table__", f"file(\"{self._temp_parquet_path}\", Parquet)")
            res = chdb_query(new_sql, "Parquet", **kwargs)
            return Table(parquet_memoryview=res.get_memview())
        elif self._parquet_memoryview is not None:
            return self.queryParquetBuffer(sql, **kwargs)
        elif self._dataframe is not None:
            return self.queryDF(sql, **kwargs)
        elif self._arrow_table is not None:
            return self.queryArrowTable(sql, **kwargs)
        else:
            raise ValueError("Table object is not initialized correctly")

    def queryParquetBuffer(self, sql: str, **kwargs):
        if "__table__" not in sql:
            raise ValueError("SQL should always contain `FROM __table__`")
        if self._parquet_memoryview is None:
            raise ValueError("Parquet buffer is None")

        parquet_fd = -1
        temp_path = None
        # on Linux, try use memfd_create to create a file descriptor for the memoryview
        if self.use_memfd and os.name == "posix":
            try:
                parquet_fd = os.memfd_create("parquet_buffer", flags=os.MFD_CLOEXEC)
            except:
                pass
        # if memfd_create failed, use tempfile to create a file descriptor for the memoryview
        if parquet_fd == -1:
            parquet_fd, temp_path = tempfile.mkstemp()
        ffd = os.fdopen(parquet_fd, "wb")
        ffd.write(self._parquet_memoryview.tobytes())
        ffd.flush()
        ret = self._run_on_temp(parquet_fd, temp_path, sql, **kwargs)
        ffd.close()
        return ret

    def queryArrowTable(self, sql: str, **kwargs):
        if "__table__" not in sql:
            raise ValueError("SQL should always contain `FROM __table__`")
        if self._arrow_table is None:
            raise ValueError("Arrow table is None")

        arrow_fd = -1
        temp_path = None
        if self.use_memfd and os.name == "posix":
            try:
                arrow_fd = os.memfd_create("arrow_table", flags=os.MFD_CLOEXEC)
            except:
                pass
        if arrow_fd == -1:
            arrow_fd, temp_path = tempfile.mkstemp()
        ffd = os.fdopen(arrow_fd, "wb")
        with pa.RecordBatchFileWriter(ffd, self._arrow_table.schema) as writer:
            writer.write_table(self._arrow_table)
        ffd.flush()
        ret = self._run_on_temp(arrow_fd, temp_path, sql=sql, fmt="Arrow", **kwargs)
        ffd.close()
        return ret

    def queryDF(self, sql: str, **kwargs):
        if "__table__" not in sql:
            raise ValueError("SQL should always contain `FROM __table__`")
        if self._dataframe is None:
            raise ValueError("Dataframe is None")

        parquet_fd = -1
        temp_path = None
        if self.use_memfd and os.name == "posix":
            try:
                parquet_fd = os.memfd_create("parquet_buffer", flags=os.MFD_CLOEXEC)
            except:
                pass
        if parquet_fd == -1:
            parquet_fd, temp_path = tempfile.mkstemp()
        ffd = os.fdopen(parquet_fd, "wb")
        self._dataframe.to_parquet(ffd, engine='pyarrow', compression=None)
        ffd.flush()
        ret = self._run_on_temp(parquet_fd, temp_path, sql=sql, **kwargs)
        ffd.close()
        return ret

    def _run_on_temp(self, fd: int, temp_path: str = None, sql: str = None, fmt: str = "Parquet", **kwargs):
        # replace "__table__" with file("temp_path", Parquet) or file("/dev/fd/{parquet_fd}", Parquet)
        if temp_path is not None:
            new_sql = sql.replace("__table__", f"file(\"{temp_path}\", {fmt})")
        else:
            os.lseek(fd, 0, os.SEEK_SET)
            new_sql = sql.replace("__table__", f"file(\"/dev/fd/{fd}\", {fmt})")
        res = chdb_query(new_sql, "Parquet", **kwargs)
        return Table(parquet_memoryview=res.get_memview())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run SQL on parquet file')
    parser.add_argument('parquet_path', type=str, help='path to parquet file')
    parser.add_argument('sql', type=str, help='SQL to run')
    parser.add_argument('--use-memfd', action='store_true', help='use memfd_create to create file descriptor')
    args = parser.parse_args()

    table = Table(parquet_path=args.parquet_path, use_memfd=args.use_memfd)
    print(table.query(args.sql))
