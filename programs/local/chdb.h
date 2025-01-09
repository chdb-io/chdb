#pragma once

#ifdef __cplusplus
#    include <condition_variable>
#    include <cstddef>
#    include <cstdint>
#    include <mutex>
#    include <queue>
#    include <string>
extern "C" {
#else
#    include <stdbool.h>
#    include <stddef.h>
#    include <stdint.h>
#endif

#define CHDB_EXPORT __attribute__((visibility("default")))
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

CHDB_EXPORT struct local_result * query_stable(int argc, char ** argv);
CHDB_EXPORT void free_result(struct local_result * result);

CHDB_EXPORT struct local_result_v2 * query_stable_v2(int argc, char ** argv);
CHDB_EXPORT void free_result_v2(struct local_result_v2 * result);

#ifdef __cplusplus
struct query_request
{
    std::string query;
    std::string format;
};

struct query_queue
{
    std::mutex mutex;
    std::condition_variable query_cv; // For query submission
    std::condition_variable result_cv; // For query result retrieval
    query_request current_query;
    local_result_v2 * current_result = nullptr;
    bool has_query = false;
    bool shutdown = false;
    bool cleanup_done = false;
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

#ifdef __cplusplus
}
#endif
