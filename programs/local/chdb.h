#pragma once

#ifdef __cplusplus
#    include <cstddef>
#    include <cstdint>
extern "C" {
#else
#    include <stddef.h>
#    include <stdint.h>
#endif

#define CHDB_EXPORT __attribute__((visibility("default")))
struct CHDB_EXPORT local_result
{
    char * buf;
    size_t len;
    void * _vec; // std::vector<char> *, for freeing
    double elapsed;
    uint64_t rows_read;
    uint64_t bytes_read;
};

struct CHDB_EXPORT local_resultV2
{
    char * buf;
    size_t len;
    void * _vec; // std::vector<char> *, for freeing
    double elapsed;
    uint64_t rows_read;
    uint64_t bytes_read;
    char * error_message;
};

CHDB_EXPORT struct local_result * query_stable(int argc, char ** argv);
CHDB_EXPORT void free_result(struct local_result * result);

CHDB_EXPORT struct local_resultV2 * query_stableV2(int argc, char ** argv);
CHDB_EXPORT void free_resultV2(struct local_resultV2 * result);

#ifdef __cplusplus
}
#endif
