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
    std::condition_variable cv; // Single condition variable for all synchronization
    std::queue<query_request> queries;
    std::queue<local_result_v2 *> results;
    bool shutdown = false;
    bool cleanup_done = false; // Flag to indicate server cleanup is complete
};
#endif

struct chdb_conn
{
    void * server; // LocalServer * server;
    bool connected;
    void * queue; // query_queue*
};

CHDB_EXPORT struct chdb_conn ** connect_chdb(int argc, char ** argv);
CHDB_EXPORT void close_conn(struct chdb_conn ** conn);
CHDB_EXPORT struct local_result_v2 * query_conn(struct chdb_conn * conn, const char * query, const char * format);

#ifdef __cplusplus
}
#endif
