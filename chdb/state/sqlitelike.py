import io
from typing import Optional, Any
from chdb import _chdb

# try import pyarrow if failed, raise ImportError with suggestion
try:
    import pyarrow as pa  # noqa
except ImportError as e:
    print(f"ImportError: {e}")
    print('Please install pyarrow via "pip install pyarrow"')
    raise ImportError("Failed to import pyarrow") from None


class Connection:
    def __init__(self, connection_string: str):
        # print("Connection", connection_string)
        self._cursor: Optional[Cursor] = None
        self._conn = _chdb.connect(connection_string)

    def cursor(self) -> "Cursor":
        self._cursor = Cursor(self._conn)
        return self._cursor

    def query(self, query: str, format: str = "ArrowStream") -> Any:
        return self._conn.query(query, format)

    def close(self) -> None:
        # print("close")
        if self._cursor:
            self._cursor.close()
        self._conn.close()


class Cursor:
    def __init__(self, connection):
        self._conn = connection
        self._cursor = self._conn.cursor()
        self._current_table: Optional[pa.Table] = None
        self._current_row: int = 0

    def execute(self, query: str) -> None:
        self._cursor.execute(query)
        result_mv = self._cursor.get_memview()
        # print("get_result", result_mv)
        if self._cursor.has_error():
            raise Exception(self._cursor.error_message())
        if self._cursor.data_size() == 0:
            self._current_table = None
            self._current_row = 0
            return
        arrow_data = result_mv.tobytes()
        reader = pa.ipc.open_stream(io.BytesIO(arrow_data))
        self._current_table = reader.read_all()
        self._current_row = 0

    def commit(self) -> None:
        self._cursor.commit()

    def fetchone(self) -> Optional[tuple]:
        if not self._current_table or self._current_row >= len(self._current_table):
            return None

        row_dict = {
            col: self._current_table.column(col)[self._current_row].as_py()
            for col in self._current_table.column_names
        }
        self._current_row += 1
        return tuple(row_dict.values())

    def fetchmany(self, size: int = 1) -> tuple:
        if not self._current_table:
            return tuple()

        rows = []
        for _ in range(size):
            if (row := self.fetchone()) is None:
                break
            rows.append(row)
        return tuple(rows)

    def fetchall(self) -> tuple:
        if not self._current_table:
            return tuple()

        remaining_rows = []
        while (row := self.fetchone()) is not None:
            remaining_rows.append(row)
        return tuple(remaining_rows)

    def close(self) -> None:
        self._cursor.close()

    def __iter__(self):
        return self

    def __next__(self) -> tuple:
        row = self.fetchone()
        if row is None:
            raise StopIteration
        return row


def connect(connection_string: str = ":memory:") -> Connection:
    """
    Create a connection to chDB backgroud server.
    Only one open connection is allowed per process. Use `close` to close the connection.
    If called with the same connection string, the same connection object will be returned.
    You can use the connection object to create cursor object. `cursor` method will return a cursor object.

    Args:
        connection_string (str, optional): Connection string. Defaults to ":memory:".
        Aslo support file path like:
          - ":memory:" (for in-memory database)
          - "test.db" (for relative path)
          - "file:test.db" (same as above)
          - "/path/to/test.db" (for absolute path)
          - "file:/path/to/test.db" (same as above)
          - "file:test.db?param1=value1&param2=value2" (for relative path with query params)
          - "///path/to/test.db?param1=value1&param2=value2" (for absolute path)

        Connection string args handling:
          Connection string can contain query params like "file:test.db?param1=value1&param2=value2"
          "param1=value1" will be passed to ClickHouse engine as start up args.

          For more details, see `clickhouse local --help --verbose`
          Some special args handling:
            - "mode=ro" would be "--readonly=1" for clickhouse (read-only mode)

    Returns:
        Connection: Connection object
    """
    return Connection(connection_string)
