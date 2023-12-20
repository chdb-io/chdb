package chdb

/*
#include <chdb.h>
*/

import (
	"C"
	"database/sql"
	"database/sql/driver"
)

func init() {
	sql.Register("chdb", Driver{})
}

type connector struct {
}

// Connect returns a connection to a database.
func (c *connector) Connect(ctx context.Context) (driver.Conn, error) {
}

// Driver returns the underying Driver of the connector,
// compatibility with the Driver method on sql.DB
func (c *connector) Driver() driver.Driver { return Driver{} }

type Driver struct{}

// Open returns a new connection to the database.
func (d Driver) Open(name string) (driver.Conn, error) {

}

// OpenConnector expects the same format as driver.Open
func (d Driver) OpenConnector(dataSourceName string) (driver.Connector, error) {
}

type conn struct {
}

func (c *conn) Close() {
}

func (c *conn) Query(query string, values []driver.Value) (driver.Rows, error) {
}

func (c *conn) QueryContext(ctx context.Context, query string, args []driver.NamedValue) (driver.Rows, error) {
}

// todo: func(c *conn) Prepare(query string)
// todo: func(c *conn) PrepareContext(ctx context.Context, query string)
// todo: prepared statment

type rows struct {
}

func (r *rows) Columns() (out []string) {
}

func (r *rows) Close() error {
}

func (r *rows) Next(dest []driver.Value) error {
}

func (r *rows) ColumnTypeDatabaseTypeName(index int) string {
}

func (r *rows) ColumnTypeNullable(index int) (nullable, ok bool) {
}

func (r *rows) ColumnTypePrecisionScale(index int) (precision, scale int64, ok bool) {
}

func (r *rows) ColumnTypeScanType(index int) reflect.Type {
}
