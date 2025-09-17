#pragma once

#include <base/defines.h>
#include <cstddef>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-identifier"

// NOLINTBEGIN

#if defined(SANITIZER) || defined(SANITIZE_COVERAGE) || defined(OS_DARWIN) || defined(OS_FREEBSD)

#define __real_malloc(size) ::malloc(size)
#define __real_calloc(nmemb, size) ::calloc(nmemb, size)
#define __real_realloc(ptr, size) ::realloc(ptr, size)
#define __real_posix_memalign(memptr, alignment, size) ::posix_memalign(memptr, alignment, size)
#define __real_aligned_alloc(alignment, size) ::aligned_alloc(alignment, size)
#define __real_valloc(size) ::valloc(size)
#if defined (OS_DARWIN)
#define __real_memalign(alignment, size) ::aligned_alloc(alignment, size)
#else
#define __real_memalign(alignment, size) ::memalign(alignment, size)
#endif
#define __real_free ::free
#if !defined(USE_MUSL) && defined(OS_LINUX)
#define __real_pvalloc(size) ::pvalloc(size)
#endif

#else

extern "C" void * __real_malloc(size_t size);
extern "C" void * __real_calloc(size_t nmemb, size_t size);
extern "C" void * __real_realloc(void * ptr, size_t size);
extern "C" int    __real_posix_memalign(void ** memptr, size_t alignment, size_t size);
extern "C" void * __real_aligned_alloc(size_t alignment, size_t size);
extern "C" void * __real_valloc(size_t size);
extern "C" void * __real_memalign(size_t alignment, size_t size);
extern "C" void   __real_free(void * ptr);
#if !defined(USE_MUSL) && defined(OS_LINUX)
extern "C" void * __real_pvalloc(size_t size);
#endif

// jemalloc extended functions
extern "C" void * mallocx(size_t size, int flags);
extern "C" void * rallocx(void * ptr, size_t size, int flags);
extern "C" void   sdallocx(void * ptr, size_t size, int flags);
extern "C" int    mallctl(const char * name, void * oldp, size_t * oldlenp, void * newp, size_t newlen);
extern "C" void   malloc_stats_print(void (*write_cb)(void *, const char *), void * cbopaque, const char * opts);

#endif

// NOLINTEND

#pragma clang diagnostic pop
