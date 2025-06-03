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
 * Contains server instance, connection state, and query processing queue
 */
struct chdb_conn
{
    void * server; /* ClickHouse LocalServer instance */
    bool connected; /* Connection state flag */
    void * queue; /* Query processing queue */
};

#endif

// Opaque handle for query results.
// Internal data structure managed by chDB implementation.
// Users should only interact through API functions.
typedef struct {
	void * internal_data;
} chdb_result;

// Connection handle wrapping database session state.
// Internal data structure managed by chDB implementation.
// Users should only interact through API functions.
typedef struct _chdb_connection {
	void * internal_data;
} * chdb_connection;

// Opaque handle for streaming query results.
// Internal data structure managed by chDB implementation.
// Users should only interact through API functions.
typedef struct {
	void * internal_data;
} chdb_streaming_result;

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
 * Executes a streaming query on the given connection.
 * @brief Initializes streaming query execution and returns result handle
 * @param conn Connection to execute query on
 * @param query SQL query string to execute
 * @param format Output format string (e.g. "CSV", default format)
 * @return Streaming result handle containing query state or error message
 * @note Returns error result if connection is invalid or closed
 */
CHDB_EXPORT chdb_streaming_result * chdb_stream_query(chdb_connection conn, const char * query, const char * format);

/**
 * Fetches next chunk of streaming results.
 * @brief Iterates through streaming query results
 * @param conn Active connection handle
 * @param result Streaming result handle from query_conn_streaming()
 * @return Materialized result chunk with data
 * @note Returns empty result when stream ends
 */
CHDB_EXPORT chdb_result * chdb_stream_fetch_result(chdb_connection conn, chdb_streaming_result * result);

/**
 * Cancels ongoing streaming query.
 * @brief Aborts streaming query execution and cleans up resources
 * @param conn Active connection handle
 * @param result Streaming result handle to cancel
 */
CHDB_EXPORT void chdb_stream_cancel_query(chdb_connection conn, chdb_streaming_result * result);

/**
 * Retrieves error message from streaming result.
 * @brief Gets error message associated with streaming query execution
 * @param result Streaming result handle from query_conn_streaming()
 * @return Null-terminated error message string, or NULL if no error occurred
 */
CHDB_EXPORT const char * chdb_streaming_result_error(chdb_streaming_result * result);

/**
 * Releases resources associated with streaming result.
 * @brief Destroys streaming result handle and frees allocated memory
 * @param result Streaming result handle to destroy
 * @warning Must be called even if query was finished or canceled
 */
CHDB_EXPORT void chdb_destroy_result(chdb_streaming_result * result);

#ifdef __cplusplus
}
#endif
