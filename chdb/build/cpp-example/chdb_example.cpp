#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "chdb.h"

int main(void) {
    // 1. Prepare connection arguments
    // Session will keep the state of query.
    // If path is None, it will create a temporary directory and use it as the database path
    // and the temporary directory will be removed when the session is closed.
    // You can also pass in a path to create a database at that path where will keep your data.

    // You can also use a connection string to pass in the path and other parameters.
    // Examples:
    //     - ":memory:" (for in-memory database)
    //     - "test.db" (for relative path)
    //     - "file:test.db" (same as above)
    //     - "/path/to/test.db" (for absolute path)
    //     - "file:/path/to/test.db" (same as above)
    //     - "file:test.db?param1=value1&param2=value2" (for relative path with query params)
    //     - "file::memory:?verbose&log-level=test" (for in-memory database with query params)
    //     - "///path/to/test.db?param1=value1&param2=value2" (for absolute path)

    // Connection string args handling:
    //     Connection string can contain query params like "file:test.db?param1=value1&param2=value2"
    //     "param1=value1" will be passed to ClickHouse engine as start up args.

    //     For more details, see `clickhouse local --help --verbose`
    //     Some special args handling:
    //     - "mode=ro" would be "--readonly=1" for clickhouse (read-only mode)

    // Important:
    //     - There can be only one session at a time. If you want to create a new session, you need to close the existing one.
    //     - Creating a new session will close the existing one.
    const char *argv[] = {
        "chdb_example",      // Program name
        // "--path=chdb_example"
    };
    int argc = 1;

    // 2. Create database connection
    chdb_connection *conn = chdb_connect(argc, const_cast<char**>(argv));
    if (!conn) {
        fprintf(stderr, "Failed to create database connection\n");
        return EXIT_FAILURE;
    }
    printf("Connected to in-memory database successfully\n");

    // 3. Execute SQL query
    const char *query = "CREATE TABLE test_table(id UInt32, name String, value Float64) ENGINE = Memory";
    chdb_query(*conn, query, "CSV");

    query = "INSERT INTO test_table (id, name, value) VALUES (1, 'Alice', 95.5)";
    chdb_query(*conn, query, "CSV");

    // 4. Process query results
    chdb_result *result = chdb_query(*conn, "SELECT * FROM test_table", "CSV");
    // The output supports various other rich formats, such as CSV, JSON, etc.
    // For more information, please see https://clickhouse.com/docs/interfaces/formats#formats-overview
    // chdb_result *result = chdb_query(*conn, query, "CSV");
    if (!result) {
        fprintf(stderr, "Query execution failed\n");
        chdb_close_conn(conn);
        return EXIT_FAILURE;
    }

    // 4. Check for query errors
    const char *error = chdb_result_error(result);
    if (error) {
        fprintf(stderr, "Query error: %s\n", error);
        chdb_destroy_query_result(result);
        chdb_close_conn(conn);
        return EXIT_FAILURE;
    }

    // 5. Process query results
    printf("Query executed successfully. Retrieved %lu rows\n", chdb_result_rows_read(result));

    char *data = chdb_result_buffer(result);
    size_t data_len = chdb_result_length(result);

    printf("Result (%zu bytes):\n%.*s\n", data_len, (int)data_len, data);

    // 6. Clean up resources
    chdb_destroy_query_result(result);
    chdb_close_conn(conn);

    printf("Resources released. Program completed.\n");

    return EXIT_SUCCESS;
}
