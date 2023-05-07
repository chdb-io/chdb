#include <stdlib.h>
#include <jemalloc/jemalloc.h> // include jemalloc header file

#ifdef __GNUC__
// #    define likely(x) __builtin_expect(!!(x), 1)
#    define unlikely(x) __builtin_expect(!!(x), 0)
#else
// #    define likely(x) !!(x)
#    define unlikely(x) !!(x)
#endif

extern void * __real_malloc(size_t size);
extern void __real_free(void * ptr);
extern void * __real_calloc(size_t nmemb, size_t size);
extern void * __real_realloc(void * ptr, size_t size);
extern int __real_posix_memalign(void ** memptr, size_t alignment, size_t size);
extern void * __real_aligned_alloc(size_t alignment, size_t size);
extern void * __real_valloc(size_t size);
extern void * __real_pvalloc(size_t size);
extern void * __real_memalign(size_t alignment, size_t size);


void * __wrap_malloc(size_t size)
{
    return je_malloc(size);
}

void __wrap_free(void * ptr)
{
    int arena_ind;
    if (unlikely(ptr == NULL))
    {
        return;
    }
    // in some glibc functions, the returned buffer is allocated by glibc malloc
    // so we need to free it by glibc free.
    // eg. getcwd, see: https://man7.org/linux/man-pages/man3/getcwd.3.html
    // so we need to check if the buffer is allocated by jemalloc
    // if not, we need to free it by glibc free
    arena_ind = je_mallctl("arenas.lookup", NULL, NULL, &ptr, sizeof(ptr));
    if (unlikely(arena_ind != 0)) {
        __real_free(ptr);
        return;
    }
    je_free(ptr);
}

void * __wrap_calloc(size_t nmemb, size_t size)
{
    return je_calloc(nmemb, size);
}

void * __wrap_realloc(void * ptr, size_t size)
{
    return je_realloc(ptr, size);
}

int __wrap_posix_memalign(void ** memptr, size_t alignment, size_t size)
{
    return je_posix_memalign(memptr, alignment, size);
}

void * __wrap_aligned_alloc(size_t alignment, size_t size)
{
    return je_aligned_alloc(alignment, size);
}

void * __wrap_valloc(size_t size)
{
    return je_valloc(size);
}

void * __wrap_pvalloc(size_t size)
{
    return je_pvalloc(size);
}

void * __wrap_memalign(size_t alignment, size_t size)
{
    return je_memalign(alignment, size);
}

void * malloc(size_t size)
{
    return je_malloc(size); // call je_malloc function
}

void free(void * ptr)
{
    je_free(ptr); // call je_free function
}

void * calloc(size_t nmemb, size_t size)
{
    return je_calloc(nmemb, size); // call je_calloc function
}

void * realloc(void * ptr, size_t size)
{
    return je_realloc(ptr, size); // call je_realloc function
}

int posix_memalign(void ** memptr, size_t alignment, size_t size)
{
    return je_posix_memalign(memptr, alignment, size); // call je_posix_memalign function
}

void * aligned_alloc(size_t alignment, size_t size)
{
    return je_aligned_alloc(alignment, size); // call je_aligned_alloc function
}

void * valloc(size_t size)
{
    return je_valloc(size); // call je_valloc function
}

void * pvalloc(size_t size)
{
    return je_pvalloc(size); // call je_pvalloc function
}

void * memalign(size_t alignment, size_t size)
{
    return je_memalign(alignment, size); // call je_memalign function
}

size_t sallocx(const void * ptr, int flags)
{
    return je_sallocx(ptr, flags); // call je_sallocx function
}

void sdallocx(void * ptr, size_t size, int flags)
{
    je_sdallocx(ptr, size, flags); // call je_sdallocx function
}

size_t nallocx(size_t size, int flags)
{
    return je_nallocx(size, flags); // call je_nallocx function
}

int mallctl(const char * name, void * oldp, size_t * oldlenp, void * newp, size_t newlen)
{
    return je_mallctl(name, oldp, oldlenp, newp, newlen); // call je_mallctl function
}

