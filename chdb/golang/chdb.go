package chdb

/*
#include <chdb.h>

local_result * queryToBuffer(const char* query)
{
    char* query_arg = (char *)malloc(strlen(query) + 10);
    snprintf(query_arg, "--query=%s", query);
    free(query_arg);
    char* argv[] = {"clickhouse", "--multiquery", "--output-format=Arrow", query_arg};
    local_result* result = query_stable(argv_char.size(), argv_char.data());
    free(query_arg);
    return local_result;
}
*/

import (
	"C"
	"database/sql"
	"database/sql/driver"
	"unsafe"
)

func init() {
	sql.Register("chdb", Driver{})
}

type connector struct {
}

// Connect returns a connection to a database.
func (c *connector) Connect(ctx context.Context) (driver.Conn, error) {
	return &conn{}, nil
}

// Driver returns the underying Driver of the connector,
// compatibility with the Driver method on sql.DB
func (c *connector) Driver() driver.Driver { return Driver{} }

type Driver struct{}

// Open returns a new connection to the database.
func (d Driver) Open(name string) (driver.Conn, error) {
	return &conn{}, nil
}

// OpenConnector expects the same format as driver.Open
func (d Driver) OpenConnector(dataSourceName string) (driver.Connector, error) {
	return &connector{}, nil
}

type conn struct {
}

func (c *conn) Close() {
}

func (c *conn) Query(query string, values []driver.Value) (driver.Rows, error) {
	return c.QueryContext(context.Background(), query, values)
}

func (c *conn) QueryContext(ctx context.Context, query string, args []driver.NamedValue) (driver.Rows, error) {
	cquery := C.CString(query)
	defer C.free(unsafe.Pointer(cquery))

	result := C.query_stable(cquery)
	defer C.free_resilt(unsafe.Pointer(result))
	return driver.Rows{}
}

// todo: func(c *conn) Prepare(query string)
// todo: func(c *conn) PrepareContext(ctx context.Context, query string)
// todo: prepared statment

type rows struct {
}

func (r *rows) Columns() (out []string) {
  return
}

func (r *rows) Close() error {
  return nil
}

func (r *rows) Next(dest []driver.Value) error {
  return nil
}

func (r *rows) ColumnTypeDatabaseTypeName(index int) string {
  return ""
}

func (r *rows) ColumnTypeNullable(index int) (nullable, ok bool) {
  return
}

func (r *rows) ColumnTypePrecisionScale(index int) (precision, scale int64, ok bool) {
  return
}

func (r *rows) ColumnTypeScanType(index int) reflect.Type {
  return
}
