#pragma once

#include <cstddef>

// Forward declarations for jemalloc extended functions
extern "C" {
void * mallocx(size_t size, int flags);
void * rallocx(void * ptr, size_t size, int flags);
void   sdallocx(void * ptr, size_t size, int flags);
int    mallctl(const char * name, void * oldp, size_t * oldlenp, void * newp, size_t newlen);
void   malloc_stats_print(void (*write_cb)(void *, const char *), void * cbopaque, const char * opts);
}
