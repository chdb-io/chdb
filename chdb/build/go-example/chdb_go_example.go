package main

/*
#include <stdlib.h>
#include "chdb.h"
*/
import "C"
import (
	"fmt"
	"os"
	"runtime"
	"unsafe"
)

// ChdbConnection wraps the C chdb_connection handle
type ChdbConnection struct {
	conn *C.chdb_connection
}

// ChdbResult wraps the C chdb_result handle
type ChdbResult struct {
	result *C.chdb_result
}

// Connect creates a new chDB connection
// args should be command-line style arguments like ["--path=/tmp/test.db"]
func Connect(args []string) (*ChdbConnection, error) {
	// Convert Go strings to C strings
	argc := C.int(len(args))
	argv := make([]*C.char, len(args))

	for i, arg := range args {
		argv[i] = C.CString(arg)
	}
	defer func() {
		for _, cstr := range argv {
			C.free(unsafe.Pointer(cstr))
		}
	}()

	// Call chdb_connect
	conn := C.chdb_connect(argc, &argv[0])
	if conn == nil {
		return nil, fmt.Errorf("failed to create chDB connection")
	}

	return &ChdbConnection{conn: conn}, nil
}

// Close closes the chDB connection and cleans up resources
func (c *ChdbConnection) Close() {
	if c.conn != nil {
		C.chdb_close_conn(c.conn)
		c.conn = nil
	}
}

// Query executes a SQL query and returns the result
func (c *ChdbConnection) Query(query, format string) (*ChdbResult, error) {
	if c.conn == nil {
		return nil, fmt.Errorf("connection is closed")
	}

	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))

	cFormat := C.CString(format)
	defer C.free(unsafe.Pointer(cFormat))

	result := C.chdb_query(*c.conn, cQuery, cFormat)
	if result == nil {
		return nil, fmt.Errorf("query execution failed")
	}

	// Check for errors
	errorMsg := C.chdb_result_error(result)
	if errorMsg != nil {
		errStr := C.GoString(errorMsg)
		C.chdb_destroy_query_result(result)
		return nil, fmt.Errorf("query error: %s", errStr)
	}

	return &ChdbResult{result: result}, nil
}

// GetBuffer returns the result data as a byte slice
func (r *ChdbResult) GetBuffer() []byte {
	if r.result == nil {
		return nil
	}

	buffer := C.chdb_result_buffer(r.result)
	length := C.chdb_result_length(r.result)

	if buffer == nil || length == 0 {
		return nil
	}

	// Convert C buffer to Go byte slice
	return C.GoBytes(unsafe.Pointer(buffer), C.int(length))
}

// GetString returns the result data as a string
func (r *ChdbResult) GetString() string {
	buffer := r.GetBuffer()
	if buffer == nil {
		return ""
	}
	return string(buffer)
}

// GetElapsed returns the query execution time in seconds
func (r *ChdbResult) GetElapsed() float64 {
	if r.result == nil {
		return 0
	}
	return float64(C.chdb_result_elapsed(r.result))
}

// GetRowsRead returns the number of rows read
func (r *ChdbResult) GetRowsRead() uint64 {
	if r.result == nil {
		return 0
	}
	return uint64(C.chdb_result_rows_read(r.result))
}

// GetBytesRead returns the number of bytes read
func (r *ChdbResult) GetBytesRead() uint64 {
	if r.result == nil {
		return 0
	}
	return uint64(C.chdb_result_bytes_read(r.result))
}

// Destroy cleans up the result resources
func (r *ChdbResult) Destroy() {
	if r.result != nil {
		C.chdb_destroy_query_result(r.result)
		r.result = nil
	}
}

func main() {
	// Set finalizer to ensure cleanup
	runtime.GC()

	fmt.Println("=== ChDB Go Example ===")

	// Create connection with in-memory database
	conn, err := Connect([]string{"--path=:memory:"})
	if err != nil {
		fmt.Printf("Failed to connect: %v\n", err)
		os.Exit(1)
	}
	defer conn.Close()

	fmt.Println("✓ Connected to chDB")

	// Create a table and insert data
	fmt.Println("\n1. Creating table and inserting data...")

	createQuery := `
		CREATE TABLE test_table (
			id UInt32,
			name String,
			value Float64
		) ENGINE = Memory
	`

	result, err := conn.Query(createQuery, "CSV")
	if err != nil {
		fmt.Printf("Create table failed: %v\n", err)
		os.Exit(1)
	}
	result.Destroy()

	// Insert some data
	insertQueries := []string{
		`INSERT INTO test_table (id, name, value) VALUES (1, 'Alice', 95.5)`,
		`INSERT INTO test_table (id, name, value) VALUES (2, 'Bob', 87.2)`,
		`INSERT INTO test_table (id, name, value) VALUES (3, 'Charlie', 92.8)`,
		`INSERT INTO test_table (id, name, value) VALUES (4, 'Diana', 98.1)`,
	}

	// Execute all insert queries
	for i, insertQuery := range insertQueries {
		fmt.Printf("Inserting record %d/%d...\n", i+1, len(insertQueries))
		result, err = conn.Query(insertQuery, "CSV")
		if err != nil {
			fmt.Printf("Insert failed for query %d: %v\n", i+1, err)
			os.Exit(1)
		}
		result.Destroy()
	}


	fmt.Println("✓ Table created and data inserted")

	// Query data
	fmt.Println("\n2. Querying data in CSV format...")

	selectQuery := "SELECT lengthUTF8(name), * FROM test_table ORDER BY value DESC"

	result, err = conn.Query(selectQuery, "CSV")
	if err != nil {
		fmt.Printf("Query failed: %v\n", err)
		os.Exit(1)
	}
	defer result.Destroy()

	fmt.Printf("CSV Result:\n%s\n", result.GetString())
	fmt.Printf("Execution time: %.3f seconds\n", result.GetElapsed())
	fmt.Printf("Rows read: %d\n", result.GetRowsRead())
	fmt.Printf("Bytes read: %d\n", result.GetBytesRead())

	fmt.Println("\n=== All examples completed successfully! ===")
}
