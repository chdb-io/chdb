#pragma once

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
extern "C" {
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#endif

#define CHDB_EXPORT __attribute__((visibility("default")))

#ifndef CHDB_NO_DEPRECATED
// WARNING: The following structs are deprecated and will be removed in a future version.
struct local_result
{
    char * buf;
    size_t len;
    void * _vec; // std::vector<char> *, for freeing
    double elapsed;
    uint64_t rows_read;
    uint64_t bytes_read;
};

#ifdef __cplusplus
struct local_result_v2
{
    char * buf = nullptr;
    size_t len = 0;
    void * _vec = nullptr; // std::vector<char> *, for freeing
    double elapsed = 0.0;
    uint64_t rows_read = 0;
    uint64_t bytes_read = 0;
    char * error_message = nullptr;
};
#else
struct local_result_v2
{
    char * buf;
    size_t len;
    void * _vec; // std::vector<char> *, for freeing
    double elapsed;
    uint64_t rows_read;
    uint64_t bytes_read;
    char * error_message;
};
#endif

/**
 * Connection structure for chDB
 * Contains ChdbClient instance and connection state
 */
struct chdb_conn
{
    void * server; /* ChdbClient instance */
    bool connected; /* Connection state flag */
};

typedef struct
{
	void * internal_data;
} chdb_streaming_result;

#endif

// Return state enumeration for chDB API functions
typedef enum chdb_state
{
    CHDBSuccess = 0,
    CHDBError = 1
} chdb_state;

// Opaque handle for query results.
// Internal data structure managed by chDB implementation.
// Users should only interact through API functions.
typedef struct chdb_result_
{
	void * internal_data;
} chdb_result;

// Connection handle wrapping database session state.
// Internal data structure managed by chDB implementation.
// Users should only interact through API functions.
typedef struct chdb_connection_
{
	void * internal_data;
} * chdb_connection;

// Holds an arrow array stream.
typedef struct chdb_arrow_stream_
{
	void * internal_data;
} * chdb_arrow_stream;

// Holds an arrow schema.
typedef struct chdb_arrow_schema_
{
	void * internal_data;
} * chdb_arrow_schema;

// Holds an arrow array.
typedef struct chdb_arrow_array_
{
	void * internal_data;
} * chdb_arrow_array;

#ifndef CHDB_NO_DEPRECATED
// WARNING: The following interfaces are deprecated and will be removed in a future version.
CHDB_EXPORT struct local_result * query_stable(int argc, char ** argv);
CHDB_EXPORT void free_result(struct local_result * result);

CHDB_EXPORT struct local_result_v2 * query_stable_v2(int argc, char ** argv);
CHDB_EXPORT void free_result_v2(struct local_result_v2 * result);

/**
 * Creates a new chDB connection.
 * Only one active connection is allowed per process.
 * Creating a new connection with different path requires closing existing connection.
 *
 * @param argc Number of command-line arguments
 * @param argv Command-line arguments array (--path=<db_path> to specify database location)
 * @return Pointer to connection pointer, or NULL on failure
 * @note Default path is ":memory:" if not specified
 */
CHDB_EXPORT struct chdb_conn ** connect_chdb(int argc, char ** argv);

/**
 * Closes an existing chDB connection and cleans up resources.
 * Thread-safe function that handles connection shutdown and cleanup.
 *
 * @param conn Pointer to connection pointer to close
 */
CHDB_EXPORT void close_conn(struct chdb_conn ** conn);

/**
 * Executes a query on the given connection.
 * Thread-safe function that handles query execution in a separate thread.
 *
 * @param conn Connection to execute query on
 * @param query SQL query string to execute
 * @param format Output format string (e.g., "CSV", default format)
 * @return Query result structure containing output or error message
 * @note Returns error result if connection is invalid or closed
 */
CHDB_EXPORT struct local_result_v2 * query_conn(struct chdb_conn * conn, const char * query, const char * format);

/**
 * Executes a query on the given connection with explicit length parameters.
 * @brief Thread-safe query execution with binary-safe string handling
 * @param conn Connection to execute query on
 * @param query SQL query string to execute (may contain null bytes)
 * @param query_len Length of query string in bytes
 * @param format Output format string (e.g., "CSV", default format)
 * @param format_len Length of format string in bytes
 * @return Query result structure containing output or error message
 * @note Returns error result if connection is invalid or closed
 * @note This function is binary-safe and can handle queries containing null bytes
 */
CHDB_EXPORT struct local_result_v2 *
query_conn_n(struct chdb_conn * conn, const char * query, size_t query_len, const char * format, size_t format_len);

/**
 * Executes a streaming query on the given connection.
 * @brief Initializes streaming query execution and returns result handle
 * @param conn Connection to execute query on
 * @param query SQL query string to execute
 * @param format Output format string (e.g. "CSV", default format)
 * @return Streaming result handle containing query state or error message
 * @note Returns error result if connection is invalid or closed
 */
CHDB_EXPORT chdb_streaming_result * query_conn_streaming(struct chdb_conn * conn, const char * query, const char * format);

/**
 * Executes a streaming query on the given connection with explicit length parameters.
 * @brief Initializes streaming query execution with binary-safe string handling
 * @param conn Connection to execute query on
 * @param query SQL query string to execute (may contain null bytes)
 * @param query_len Length of query string in bytes
 * @param format Output format string (e.g., "CSV", default format)
 * @param format_len Length of format string in bytes
 * @return Streaming result handle containing query state or error message
 * @note Returns error result if connection is invalid or closed
 * @note This function is binary-safe and can handle queries containing null bytes
 * @note Use chdb_streaming_fetch_result() to retrieve data chunks from the streaming query
 */
CHDB_EXPORT chdb_streaming_result *
query_conn_streaming_n(struct chdb_conn * conn, const char * query, size_t query_len, const char * format, size_t format_len);

/**
 * Retrieves error message from streaming result.
 * @brief Gets error message associated with streaming query execution
 * @param result Streaming result handle from query_conn_streaming()
 * @return Null-terminated error message string, or NULL if no error occurred
 */
CHDB_EXPORT const char * chdb_streaming_result_error(chdb_streaming_result * result);

/**
 * Fetches next chunk of streaming results.
 * @brief Iterates through streaming query results
 * @param conn Active connection handle
 * @param result Streaming result handle from query_conn_streaming()
 * @return Materialized result chunk with data
 * @note Returns empty result when stream ends
 */
CHDB_EXPORT struct local_result_v2 * chdb_streaming_fetch_result(struct chdb_conn * conn, chdb_streaming_result * result);

/**
 * Cancels ongoing streaming query.
 * @brief Aborts streaming query execution and cleans up resources
 * @param conn Active connection handle
 * @param result Streaming result handle to cancel
 */
CHDB_EXPORT void chdb_streaming_cancel_query(struct chdb_conn * conn, chdb_streaming_result * result);

/**
 * Releases resources associated with streaming result.
 * @brief Destroys streaming result handle and frees allocated memory
 * @param result Streaming result handle to destroy
 * @warning Must be called even if query was finished or canceled
 */
 CHDB_EXPORT void chdb_destroy_result(chdb_streaming_result * result);

#endif

/**
 * Creates a new chDB connection.
 * Only one active connection is allowed per process.
 * Creating a new connection with different path requires closing existing connection.
 *
 * @param argc Number of command-line arguments
 * @param argv Command-line arguments array (--path=<db_path> to specify database location)
 * @return Pointer to connection pointer, or NULL on failure
 * @note Default path is ":memory:" if not specified
 */
CHDB_EXPORT chdb_connection * chdb_connect(int argc, char ** argv);

/**
 * Closes an existing chDB connection and cleans up resources.
 * Thread-safe function that handles connection shutdown and cleanup.
 *
 * @param conn Pointer to connection pointer to close
 */
 CHDB_EXPORT void chdb_close_conn(chdb_connection * conn);

/**
 * Executes a query on the given connection.
 * Thread-safe function that handles query execution in a separate thread.
 *
 * @param conn Connection to execute query on
 * @param query SQL query string to execute
 * @param format Output format string (e.g., "CSV", default format)
 * @return Query result structure containing output or error message
 * @note Returns error result if connection is invalid or closed
 */
CHDB_EXPORT chdb_result * chdb_query(chdb_connection conn, const char * query, const char * format);

/**
 * Executes a query on the given connection with explicit length parameters.
 * @brief Thread-safe query execution with binary-safe string handling
 * @param conn Connection to execute query on
 * @param query SQL query string to execute (may contain null bytes)
 * @param query_len Length of query string in bytes
 * @param format Output format string (e.g., "CSV", default format)
 * @param format_len Length of format string in bytes
 * @return Query result structure containing output or error message
 * @note Returns error result if connection is invalid or closed
 * @note This function is binary-safe and can handle queries containing null bytes
 * @note Use chdb_result_* functions to access result data and metadata
 */
CHDB_EXPORT chdb_result * chdb_query_n(chdb_connection conn, const char * query, size_t query_len, const char * format, size_t format_len);

/**
 * @brief Execute a query with command-line interface
 * @param argc Argument count (same as main()'s argc)
 * @param argv Argument vector (same as main()'s argv)
 * @return Query result structure containing output or error message
 */
CHDB_EXPORT chdb_result * chdb_query_cmdline(int argc, char ** argv);

/**
 * Executes a streaming query on the given connection.
 * @brief Initializes streaming query execution and returns result handle
 * @param conn Connection to execute query on
 * @param query SQL query string to execute
 * @param format Output format string (e.g. "CSV", default format)
 * @return Streaming result handle containing query state or error message
 * @note Returns error result if connection is invalid or closed
 */
CHDB_EXPORT chdb_result * chdb_stream_query(chdb_connection conn, const char * query, const char * format);

/**
 * Executes a streaming query with explicit string lengths (binary-safe).
 * @brief Initializes streaming query execution with specified buffer lengths
 * @param conn Connection to execute query on
 * @param query SQL query buffer (may contain null bytes)
 * @param query_len Length of query buffer in bytes
 * @param format Output format buffer (may contain null bytes)
 * @param format_len Length of format buffer in bytes
 * @return Streaming result handle containing query state or error message
 * @note Strings do not need to be null-terminated
 * @note Use this function when dealing with queries/formats containing null bytes
 */
CHDB_EXPORT chdb_result *
chdb_stream_query_n(chdb_connection conn, const char * query, size_t query_len, const char * format, size_t format_len);

/**
 * Fetches next chunk of streaming results.
 * @brief Iterates through streaming query results
 * @param conn Active connection handle
 * @param result Streaming result handle from query_conn_streaming()
 * @return Materialized result chunk with data
 * @note Returns empty result when stream ends
 */
CHDB_EXPORT chdb_result * chdb_stream_fetch_result(chdb_connection conn, chdb_result * result);

/**
 * Cancels ongoing streaming query.
 * @brief Aborts streaming query execution and cleans up resources
 * @param conn Active connection handle
 * @param result Streaming result handle to cancel
 */
CHDB_EXPORT void chdb_stream_cancel_query(chdb_connection conn, chdb_result * result);

/**
 * Destroys a query result and releases all associated resources
 * @param result The result handle to destroy
 */
CHDB_EXPORT void chdb_destroy_query_result(chdb_result * result);

/**
 * Gets pointer to the result data buffer
 * @param result The query result handle
 * @return Read-only pointer to the result data
 */
CHDB_EXPORT char * chdb_result_buffer(chdb_result * result);

/**
 * Gets the length of the result data
 * @param result The query result handle
 * @return Size of result data in bytes
 */
CHDB_EXPORT size_t chdb_result_length(chdb_result * result);

/**
 * Gets query execution time
 * @param result The query result handle
 * @return Elapsed time in seconds
 */
CHDB_EXPORT double chdb_result_elapsed(chdb_result * result);

/**
 * Gets total rows in query result
 * @param result The query result handle
 * @return Number of rows contained in the result set
 */
CHDB_EXPORT uint64_t chdb_result_rows_read(chdb_result * result);

/**
 * Gets the total bytes occupied by the result set in internal binary format
 * @param result The query result handle
 * @return Number of bytes occupied by the result set in internal binary representation
 */
CHDB_EXPORT uint64_t chdb_result_bytes_read(chdb_result * result);

/**
 * Gets rows read from storage engine
 * @param result The query result handle
 * @return Number of rows read from storage
 */
CHDB_EXPORT uint64_t chdb_result_storage_rows_read(chdb_result * result);

/**
 * Gets bytes read from storage engine
 * @param result The query result handle
 * @return Number of bytes read from storage engine
 */
CHDB_EXPORT uint64_t chdb_result_storage_bytes_read(chdb_result * result);

/**
 * Retrieves error message from query execution
 * @param result The query result handle
 * @return Null-terminated error description, NULL if no error
 */
CHDB_EXPORT const char * chdb_result_error(chdb_result * result);

//===--------------------------------------------------------------------===//
// Arrow Integration
//===--------------------------------------------------------------------===//

/**
 * Registers an Arrow stream as an arrow stream table function with the given name
 * @param conn The connection on which to execute the registration
 * @param table_name Name to register for the arrow stream table function
 * @param arrow_stream chdb Arrow stream handle
 * @return CHDBSuccess on success, CHDBError on failure
 */
CHDB_EXPORT chdb_state chdb_arrow_scan(
    chdb_connection conn, const char * table_name,
    chdb_arrow_stream arrow_stream);

/**
 * Registers an Arrow array as an arrow stream table function with the given name
 * @param conn The connection on which to execute the registration
 * @param table_name Name to register for the arrow stream table function
 * @param arrow_schema chdb Arrow schema handle
 * @param arrow_array chdb Arrow array handle
 * @return CHDBSuccess on success, CHDBError on failure
 */
CHDB_EXPORT chdb_state chdb_arrow_array_scan(
    chdb_connection conn, const char * table_name,
    chdb_arrow_schema arrow_schema, chdb_arrow_array arrow_array);

/**
 * Unregisters an arrow stream table function that was previously registered via chdb_arrow_scan
 * @param conn The connection on which to execute the unregister operation
 * @param table_name Name of the arrow stream table function to unregister
 * @return CHDBSuccess on success, CHDBError on failure
 */
CHDB_EXPORT chdb_state chdb_arrow_unregister_table(chdb_connection conn, const char * table_name);

#ifdef __cplusplus
}
#endif
