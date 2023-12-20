#pragma once

#ifdef __cplusplus
#    include <cstddef>
#    include <cstdint>
extern "C" {
#else
#    include <stddef.h>
#    include <stdint.h>
#endif

struct __attribute__((visibility("default"))) local_result
{
    char * buf;
    size_t len;
    void * _vec; // std::vector<char> *, for freeing
    double elapsed;
    uint64_t rows_read;
    uint64_t bytes_read;
};

__attribute__((visibility("default"))) struct local_result * query_stable(int argc, char ** argv);
__attribute__((visibility("default"))) void free_result(struct local_result * result);

#ifdef __cplusplus
}
#endif
