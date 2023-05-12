import json
from . import err
from .cursors import Cursor
from . import converters

DEBUG = False
VERBOSE = False


class Connection(object):
    """
    Representation of a connection with chdb.

    The proper way to get an instance of this class is to call
    connect().

    Accepts several arguments:

    :param cursorclass: Custom cursor class to use.

    See `Connection <https://www.python.org/dev/peps/pep-0249/#connection-objects>`_ in the
    specification.
    """

    _closed = False

    def __init__(self, cursorclass=Cursor):

        self._resp = None

        # 1. pre-process params in init
        self.encoding = 'utf8'

        self.cursorclass = cursorclass

        self._result = None
        self._affected_rows = 0

        self.connect()

    def connect(self):
        self._closed = False
        self._execute_command("select 1;")
        self._read_query_result()

    def close(self):
        """
        Send the quit message and close the socket.

        See `Connection.close() <https://www.python.org/dev/peps/pep-0249/#Connection.close>`_
        in the specification.

        :raise Error: If the connection is already closed.
        """
        if self._closed:
            raise err.Error("Already closed")
        self._closed = True

    @property
    def open(self):
        """Return True if the connection is open"""
        return not self._closed

    def commit(self):
        """
        Commit changes to stable storage.

        See `Connection.commit() <https://www.python.org/dev/peps/pep-0249/#commit>`_
        in the specification.
        """
        return

    def rollback(self):
        """
        Roll back the current transaction.

        See `Connection.rollback() <https://www.python.org/dev/peps/pep-0249/#rollback>`_
        in the specification.
        """
        return

    def cursor(self, cursor=None):
        """
        Create a new cursor to execute queries with.

        :param cursor: The type of cursor to create; current only :py:class:`Cursor`
            None means use Cursor.
        """
        if cursor:
            return cursor(self)
        return self.cursorclass(self)

    # The following methods are INTERNAL USE ONLY (called from Cursor)
    def query(self, sql):
        if isinstance(sql, str):
            sql = sql.encode(self.encoding, 'surrogateescape')
        self._execute_command(sql)
        self._affected_rows = self._read_query_result()
        return self._affected_rows

    def _execute_command(self, sql):
        """
        :raise InterfaceError: If the connection is closed.
        :raise ValueError: If no username was specified.
        """
        if self._closed:
            raise err.InterfaceError("Connection closed")

        if isinstance(sql, str):
            sql = sql.encode(self.encoding)

        if isinstance(sql, bytearray):
            sql = bytes(sql)

        # drop last command return
        if self._resp is not None:
            self._resp = None

        if DEBUG:
            print("DEBUG: query:", sql)
        try:
            import chdb
            self._resp = chdb.query(sql, output_format="JSON").data()
        except Exception as error:
            raise err.InterfaceError("query err: %s" % error)

    def escape(self, obj, mapping=None):
        """Escape whatever value you pass to it.

        Non-standard, for internal use; do not use this in your applications.
        """
        if isinstance(obj, str):
            return "'" + self.escape_string(obj) + "'"
        if isinstance(obj, (bytes, bytearray)):
            ret = self._quote_bytes(obj)
            return ret
        return converters.escape_item(obj, mapping=mapping)

    def escape_string(self, s):
        return converters.escape_string(s)

    def _quote_bytes(self, s):
        return converters.escape_bytes(s)

    def _read_query_result(self):
        self._result = None
        result = CHDBResult(self)
        result.read()
        self._result = result
        return result.affected_rows

    def __enter__(self):
        """Context manager that returns a Cursor"""
        return self.cursor()

    def __exit__(self, exc, value, traceback):
        """On successful exit, commit. On exception, rollback"""
        if exc:
            self.rollback()
        else:
            self.commit()

    @property
    def resp(self):
        return self._resp


class CHDBResult(object):
    def __init__(self, connection):
        """
        :type connection: Connection
        """
        self.connection = connection
        self.affected_rows = 0
        self.insert_id = None
        self.warning_count = 0
        self.message = None
        self.field_count = 0
        self.description = None
        self.rows = None
        self.has_next = None

    def read(self):
        try:
            data = json.loads(self.connection.resp)
        except Exception as error:
            raise err.InterfaceError("Unsupported response format:" % error)

        try:
            self.field_count = len(data["meta"])
            description = []
            for meta in data["meta"]:
                fields = [meta["name"], meta["type"]]
                description.append(tuple(fields))
            self.description = tuple(description)

            rows = []
            for line in data["data"]:
                row = []
                for i in range(self.field_count):
                    column_data = converters.convert_column_data(self.description[i][1], line[self.description[i][0]])
                    row.append(column_data)
                rows.append(tuple(row))
            self.rows = tuple(rows)
        except Exception as error:
            raise err.InterfaceError("Read return data err:" % error)
