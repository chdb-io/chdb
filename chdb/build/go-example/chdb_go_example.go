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
	"reflect"
	"unsafe"

	"github.com/ClickHouse/ch-go/proto"
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

	// Test various ClickHouse SQL functions
	fmt.Println("\n2. Testing ClickHouse SQL Functions...")

	testQueries := []struct {
		name  string
		query string
	}{
		{
			"String Functions",
			`SELECT
				name,
				lengthUTF8(name) as name_length,
				upper(name) as upper_name,
				lower(name) as lower_name,
				reverse(name) as reversed_name,
				concat(name, '_test') as concat_name
			FROM test_table ORDER BY id`,
		},
		{
			"Math Functions",
			`SELECT
				id,
				value,
				round(value, 1) as rounded,
				ceil(value) as ceiling,
				floor(value) as floor_val,
				abs(value - 90) as abs_diff,
				sqrt(value) as square_root,
				pow(value, 2) as squared
			FROM test_table ORDER BY id`,
		},
		{
			"Conditional Functions",
			`SELECT
				name,
				value,
				if(value >= 95, 'Excellent', 'Good') as grade,
				multiIf(
					value >= 95, 'A',
					value >= 90, 'B',
					value >= 80, 'C',
					'D'
				) as letter_grade,
				greatest(value, 90.0) as min_90,
				least(value, 95.0) as max_95
			FROM test_table ORDER BY value DESC`,
		},
		{
			"Date/Time Functions",
			`SELECT
				name,
				now() as current_time,
				today() as current_date,
				toYear(now()) as current_year,
				toMonth(now()) as current_month,
				toDayOfWeek(today()) as day_of_week,
				formatDateTime(now(), '%Y-%m-%d %H:%M:%S') as formatted_time
			FROM test_table LIMIT 1`,
		},
		{
			"Array Functions",
			`SELECT
				[1, 2, 3, 4, 5] as numbers,
				arrayElement([1, 2, 3, 4, 5], 3) as third_element,
				arrayConcat([1, 2], [3, 4]) as concatenated,
				arrayReverse([1, 2, 3, 4, 5]) as reversed_array,
				arraySum([1, 2, 3, 4, 5]) as array_sum
			LIMIT 1`,
		},
		{
			"Aggregate Functions",
			`SELECT
				count() as total_count,
				avg(value) as average_value,
				min(value) as min_value,
				max(value) as max_value,
				sum(value) as sum_value,
				stddevPop(value) as std_deviation,
				groupArray(name) as all_names,
				groupConcat(name) as names_concat
			FROM test_table`,
		},
		{
			"Type Conversion Functions",
			`SELECT
				id,
				toString(id) as id_string,
				toFloat64(id) as id_float,
				toUInt32(value) as value_int,
				formatReadableSize(toUInt64(value * 1000000)) as readable_size,
				hex(id) as id_hex,
				bin(id) as id_binary
			FROM test_table ORDER BY id`,
		},
		{
			"Hash Functions",
			`SELECT
				name,
				cityHash64(name) as city_hash,
				sipHash64(name) as sip_hash,
				MD5(name) as md5_hash,
				SHA1(name) as sha1_hash,
				SHA256(name) as sha256_hash
			FROM test_table ORDER BY id`,
		},
		{
			"JSON Functions",
			`SELECT
				'{"name": "' || name || '", "score": ' || toString(value) || '}' as json_str,
				JSONExtractString('{"name": "' || name || '", "score": ' || toString(value) || '}', 'name') as extracted_name,
				JSONExtractFloat('{"name": "' || name || '", "score": ' || toString(value) || '}', 'score') as extracted_score,
				isValidJSON('{"name": "' || name || '", "score": ' || toString(value) || '}') as is_valid_json
			FROM test_table ORDER BY id`,
		},
		{
			"URL Functions",
			`SELECT
				'https://example.com/path?param=' || name as url,
				protocol('https://example.com/path?param=' || name) as url_protocol,
				domain('https://example.com/path?param=' || name) as url_domain,
				path('https://example.com/path?param=' || name) as url_path,
				queryString('https://example.com/path?param=' || name) as url_query
			FROM test_table LIMIT 2`,
		},
	}

	for i, test := range testQueries {
		fmt.Printf("\n%d. %s:\n", i+1, test.name)

		result, err := conn.Query(test.query, "Native")
		if err != nil {
			fmt.Printf("   ❌ Query failed: %v\n", err)
			os.Exit(1)
		}

		output := readNativeBuffer(result.GetBuffer())
		fmt.Printf("   ✓ Result:\n%s", output)
		fmt.Printf("   Execution time: %.3f seconds\n", result.GetElapsed())
		result.Destroy()
	}

	fmt.Println("\n=== All examples completed successfully! ===")
}

package main

/*
#include <stdlib.h>
#include "chdb.h"
*/
import "C"
import (
	"fmt"
	"os"
	"bytes"
	"runtime"
	"unsafe"
	"reflect"
	"github.com/ClickHouse/ch-go/proto"
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

	// Test various ClickHouse SQL functions
	fmt.Println("\n2. Testing ClickHouse SQL Functions...")

	testQueries := []struct {
		name  string
		query string
	}{
		{
			"String Functions",
			`SELECT
				name,
				lengthUTF8(name) as name_length,
				upper(name) as upper_name,
				lower(name) as lower_name,
				reverse(name) as reversed_name,
				concat(name, '_test') as concat_name
			FROM test_table ORDER BY id`,
		},
		{
			"Math Functions",
			`SELECT
				id,
				value,
				round(value, 1) as rounded,
				ceil(value) as ceiling,
				floor(value) as floor_val,
				abs(value - 90) as abs_diff,
				sqrt(value) as square_root,
				pow(value, 2) as squared
			FROM test_table ORDER BY id`,
		},
		{
			"Conditional Functions",
			`SELECT
				name,
				value,
				if(value >= 95, 'Excellent', 'Good') as grade,
				multiIf(
					value >= 95, 'A',
					value >= 90, 'B',
					value >= 80, 'C',
					'D'
				) as letter_grade,
				greatest(value, 90.0) as min_90,
				least(value, 95.0) as max_95
			FROM test_table ORDER BY value DESC`,
		},
		{
			"Date/Time Functions",
			`SELECT
				name,
				now() as current_time,
				today() as current_date,
				toYear(now()) as current_year,
				toMonth(now()) as current_month,
				toDayOfWeek(today()) as day_of_week,
				formatDateTime(now(), '%Y-%m-%d %H:%M:%S') as formatted_time
			FROM test_table LIMIT 1`,
		},
		{
			"Array Functions",
			`SELECT
				[1, 2, 3, 4, 5] as numbers,
				arrayElement([1, 2, 3, 4, 5], 3) as third_element,
				arrayConcat([1, 2], [3, 4]) as concatenated,
				arrayReverse([1, 2, 3, 4, 5]) as reversed_array,
				arraySum([1, 2, 3, 4, 5]) as array_sum
			LIMIT 1`,
		},
		{
			"Aggregate Functions",
			`SELECT
				count() as total_count,
				avg(value) as average_value,
				min(value) as min_value,
				max(value) as max_value,
				sum(value) as sum_value,
				stddevPop(value) as std_deviation,
				groupArray(name) as all_names,
				groupConcat(name) as names_concat
			FROM test_table`,
		},
		{
			"Type Conversion Functions",
			`SELECT
				id,
				toString(id) as id_string,
				toFloat64(id) as id_float,
				toUInt32(value) as value_int,
				formatReadableSize(toUInt64(value * 1000000)) as readable_size,
				hex(id) as id_hex,
				bin(id) as id_binary
			FROM test_table ORDER BY id`,
		},
		{
			"Hash Functions",
			`SELECT
				name,
				cityHash64(name) as city_hash,
				sipHash64(name) as sip_hash,
				MD5(name) as md5_hash,
				SHA1(name) as sha1_hash,
				SHA256(name) as sha256_hash
			FROM test_table ORDER BY id`,
		},
		{
			"JSON Functions",
			`SELECT
				'{"name": "' || name || '", "score": ' || toString(value) || '}' as json_str,
				JSONExtractString('{"name": "' || name || '", "score": ' || toString(value) || '}', 'name') as extracted_name,
				JSONExtractFloat('{"name": "' || name || '", "score": ' || toString(value) || '}', 'score') as extracted_score,
				isValidJSON('{"name": "' || name || '", "score": ' || toString(value) || '}') as is_valid_json
			FROM test_table ORDER BY id`,
		},
		{
			"URL Functions",
			`SELECT
				'https://example.com/path?param=' || name as url,
				protocol('https://example.com/path?param=' || name) as url_protocol,
				domain('https://example.com/path?param=' || name) as url_domain,
				path('https://example.com/path?param=' || name) as url_path,
				queryString('https://example.com/path?param=' || name) as url_query
			FROM test_table LIMIT 2`,
		},
	}

	for i, test := range testQueries {
		fmt.Printf("\n%d. %s:\n", i+1, test.name)

		result, err := conn.Query(test.query, "Native")
		if err != nil {
			fmt.Printf("   ❌ Query failed: %v\n", err)
			os.Exit(1)
		}

		output := readNativeBuffer(result.GetBuffer())
		fmt.Printf("   ✓ Result:\n%s", output)
		fmt.Printf("   Execution time: %.3f seconds\n", result.GetElapsed())
		result.Destroy()
	}

	fmt.Println("\n=== All examples completed successfully! ===")
}


func readNativeBuffer(data []byte) error {
	r := proto.NewReader(bytes.NewReader(data))

	// basic reader, it's possible to add the schema instead of using case/switch
	var (
		block   proto.Block
		results proto.Results
	)
	err := results.DecodeResult(r, 54451, block)
	if err != nil {
		fmt.Println("decode result block error: ", err)
	}
	fmt.Println("print raws")
	rs := results.Auto()
	inspect(rs)
	inspect(results)
	fmt.Println(results.Rows())

	if err := block.DecodeRawBlock(r, 54451, results.Auto()); err != nil {
		fmt.Println("decode raw block error : ", err)
	}
	fmt.Println("print raws block")
	inspect(results)
	print(results)
	for i := range results {
		result := results[i]
		fmt.Println(result.Name)
		inspect(result)
		inspect(result.Data)
		fmt.Println("Data.Row() : ", result.Data.Rows())
		val := reflect.ValueOf(result.Data).Elem()
		fmt.Println("Data val", val)
		fmt.Println("Data ", result.Data)
		fmt.Println("Data Type", result.Data.Type())

		switch col := result.Data.(type) {

		case *proto.ColUInt8:
			fmt.Println("  UInt8 values:", col.Row(0))
		case *proto.ColUInt16:
			fmt.Println("  UInt16 values:", col.Row(0))
		case *proto.ColUInt32:
			fmt.Println("  UInt32 values:", col.Row(0))
		case *proto.ColUInt64:
			fmt.Println("  UInt64 values:", col.Row(0))
		case *proto.ColInt8:
			fmt.Println("  Int8 values:", col.Row(0))
		case *proto.ColNullable[string]:
			fmt.Println("  UInt8 values:", col.Row(0))
		case *proto.ColNullable[float64]:
			fmt.Println("  UInt8 values:", col.Row(0))
		case *proto.ColNullable[int32]:
			fmt.Println("  UInt8 values:", col.Row(0))
		case *proto.ColNullable[bool]:
			fmt.Println("  UInt8 values:", col.Row(0))
		case *proto.ColStr:
			fmt.Println("  ColStr:", col.Row(0))
		case *proto.ColFloat64:
			fmt.Println("  ColFloat64:", col.Row(0))
		case *proto.ColArr[uint8]:
			fmt.Println("  ColArr[uint8]:", col.Row(0))
		case *proto.ColArr[string]:
			fmt.Println("  ColArr[string]:", col.Row(0))
		case *proto.ColFixedStr16:
			fmt.Println("  ColFixedStr16:", col.Row(0))
		case  *proto.ColDateTime:
			fmt.Println("  *proto.ColDateTime:", col.Row(0))
		case *proto.ColDate:
			fmt.Println("  *proto.ColDate:", col.Row(0))
		default:
			fmt.Printf("  Unhandled type: %T\n", col)
		}
	}
	return nil
}

func inspect(v ...interface{}) {
	for _, v := range v {
		fmt.Printf("%T %#v \n", v, v)
	}
}
