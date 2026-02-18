#include <cassert>
#include <new>
#include "config.h"

#include <Common/AllocationInterceptors.h>
#include <Common/CurrentThread.h>
#include <Common/memory.h>

#if defined(OS_DARWIN) && (USE_JEMALLOC)
/// In case of OSX jemalloc register itself as a default zone allocator.
///
/// Sure jemalloc will register itself, since zone_register() declared with
/// constructor attribute (since zone_register is also forbidden from
/// optimizing out), however those constructors will be called before
/// constructors for global variable initializers (__cxx_global_var_init()).
///
/// So to make jemalloc under OSX more stable, we will call it explicitly from
/// global variable initializers so that each allocation will use it.
/// (NOTE: It is ok to call it twice, since zone_register() is a no-op if the
/// default zone is already replaced with something.)
///
/// Refs: https://github.com/jemalloc/jemalloc/issues/708

extern "C"
{
    extern void zone_register();
}

static struct InitializeJemallocZoneAllocatorForOSX
{
    InitializeJemallocZoneAllocatorForOSX()
    {
        zone_register();
        /// jemalloc() initializes itself only on malloc()
        /// and so if some global initializer will have free(nullptr)
        /// jemalloc may trigger some internal assertion.
        ///
        /// To prevent this, we explicitly call malloc(free()) here.
        if (void * ptr = malloc(0))
        {
            free(ptr);
        }
    }
} initializeJemallocZoneAllocatorForOSX;
#endif

/// Replace default new/delete with memory tracking versions.
/// @sa https://en.cppreference.com/w/cpp/memory/new/operator_new
///     https://en.cppreference.com/w/cpp/memory/new/operator_delete

/// new

void * operator new(std::size_t size)
{
    AllocationTrace trace;
    std::size_t actual_size = Memory::trackMemory(size, trace);
    void * ptr = Memory::newImpl(size);
    trace.onAlloc(ptr, actual_size);
    return ptr;
}

void * operator new(std::size_t size, std::align_val_t align)
{
    AllocationTrace trace;
    std::size_t actual_size = Memory::trackMemory(size, trace, align);
    void * ptr = Memory::newImpl(size, align);
    trace.onAlloc(ptr, actual_size);
    return ptr;
}

void * operator new[](std::size_t size)
{
    AllocationTrace trace;
    std::size_t actual_size = Memory::trackMemory(size, trace);
    void * ptr =  Memory::newImpl(size);
    trace.onAlloc(ptr, actual_size);
    return ptr;
}

void * operator new[](std::size_t size, std::align_val_t align)
{
    AllocationTrace trace;
    std::size_t actual_size = Memory::trackMemory(size, trace, align);
    void * ptr = Memory::newImpl(size, align);
    trace.onAlloc(ptr, actual_size);
    return ptr;
}

void * operator new(std::size_t size, const std::nothrow_t &) noexcept
{
    AllocationTrace trace;
    std::size_t actual_size = Memory::trackMemory(size, trace);
    void * ptr = Memory::newNoExcept(size);
    trace.onAlloc(ptr, actual_size);
    return ptr;
}

void * operator new[](std::size_t size, const std::nothrow_t &) noexcept
{
    AllocationTrace trace;
    std::size_t actual_size = Memory::trackMemory(size, trace);
    void * ptr = Memory::newNoExcept(size);
    trace.onAlloc(ptr, actual_size);
    return ptr;
}

void * operator new(std::size_t size, std::align_val_t align, const std::nothrow_t &) noexcept
{
    AllocationTrace trace;
    std::size_t actual_size = Memory::trackMemory(size, trace, align);
    void * ptr = Memory::newNoExcept(size, align);
    trace.onAlloc(ptr, actual_size);
    return ptr;
}

void * operator new[](std::size_t size, std::align_val_t align, const std::nothrow_t &) noexcept
{
    AllocationTrace trace;
    std::size_t actual_size = Memory::trackMemory(size, trace, align);
    void * ptr = Memory::newNoExcept(size, align);
    trace.onAlloc(ptr, actual_size);
    return ptr;
}

#if USE_JEMALLOC

extern "C" void __real_free(void * ptr);

inline ALWAYS_INLINE bool isJemallocMemory(void * ptr)
{
    int arena_ind = je_mallctl("arenas.lookup", nullptr, nullptr, &ptr, sizeof(ptr));
    return arena_ind == 0; // arena_ind == 0 means jemalloc memory
}

/// Safely handle memory that may not have been allocated by jemalloc.
///
/// This function addresses a critical memory management issue in libpybind11nonlimitedapi_chdb.so:
/// - The 'new' operator inside the library may not be overridden by chdb's custom allocator
/// - However, the 'delete' operator is overridden by chdb's custom implementation
/// - This mismatch can cause crashes when trying to free memory that wasn't allocated by jemalloc
///
/// To prevent crashes, this function checks if the memory pointer was allocated by jemalloc
/// before attempting to free it. If the memory was allocated by a different allocator,
/// it uses the system's default free() function instead.
///
/// Note: We don't update memory tracking for non-jemalloc memory since it was likely
/// never tracked by our system in the first place.
inline ALWAYS_INLINE bool tryFreeNonJemallocMemory(void * ptr)
{
    if (unlikely(ptr == nullptr))
        return true;

    int arena_ind = je_mallctl("arenas.lookup", nullptr, nullptr, &ptr, sizeof(ptr));
    if (unlikely(arena_ind != 0))
    {
        __real_free(ptr);
        return true;
    }

    return false; // Not handled - should continue with jemalloc path
}

namespace Memory
{
thread_local bool disable_memory_check{false};
}

inline ALWAYS_INLINE bool tryFreeNonJemallocMemoryConditional(void * ptr)
{
    if (unlikely(ptr == nullptr))
        return true;

    if (likely(Memory::disable_memory_check))
        return false;

    int arena_ind = je_mallctl("arenas.lookup", nullptr, nullptr, &ptr, sizeof(ptr));
    if (unlikely(arena_ind != 0))
    {
        __real_free(ptr);
        return true;
    }

    return false; // Not handled - should continue with jemalloc path
}

#endif

/// delete

/// C++17 std 21.6.2.1 (11)
/// If a function without a size parameter is defined, the program should also define the corresponding function with a size parameter.
/// If a function with a size parameter is defined, the program shall also define the corresponding version without the size parameter.

/// cppreference:
/// It's unspecified whether size-aware or size-unaware version is called when deleting objects of
/// incomplete type and arrays of non-class and trivially-destructible class types.

void operator delete(void * ptr) noexcept
{
#if USE_JEMALLOC
    if (tryFreeNonJemallocMemoryConditional(ptr))
        return;
#endif
    AllocationTrace trace;
    std::size_t actual_size = Memory::untrackMemory(ptr, trace);
    trace.onFree(ptr, actual_size);
    Memory::deleteImpl(ptr);
}

void operator delete(void * ptr, std::align_val_t align) noexcept
{
#if USE_JEMALLOC
    if (tryFreeNonJemallocMemoryConditional(ptr))
        return;
#endif
    AllocationTrace trace;
    std::size_t actual_size = Memory::untrackMemory(ptr, trace, 0, align);
    trace.onFree(ptr, actual_size);
    Memory::deleteImpl(ptr);
}

void operator delete[](void * ptr) noexcept
{
#if USE_JEMALLOC
    if (tryFreeNonJemallocMemoryConditional(ptr))
        return;
#endif
    AllocationTrace trace;
    std::size_t actual_size = Memory::untrackMemory(ptr, trace);
    trace.onFree(ptr, actual_size);
    Memory::deleteImpl(ptr);
}

void operator delete[](void * ptr, std::align_val_t align) noexcept
{
#if USE_JEMALLOC
    if (tryFreeNonJemallocMemoryConditional(ptr))
        return;
#endif
    AllocationTrace trace;
    std::size_t actual_size = Memory::untrackMemory(ptr, trace, 0, align);
    trace.onFree(ptr, actual_size);
    Memory::deleteImpl(ptr);
}

void operator delete(void * ptr, std::size_t size) noexcept
{
#if USE_JEMALLOC
    if (tryFreeNonJemallocMemoryConditional(ptr))
        return;
#endif
    AllocationTrace trace;
    std::size_t actual_size = Memory::untrackMemory(ptr, trace, size);
    trace.onFree(ptr, actual_size);
    Memory::deleteSized(ptr, size);
}

void operator delete(void * ptr, std::size_t size, std::align_val_t align) noexcept
{
#if USE_JEMALLOC
    if (tryFreeNonJemallocMemoryConditional(ptr))
        return;
#endif
    AllocationTrace trace;
    std::size_t actual_size = Memory::untrackMemory(ptr, trace, size, align);
    trace.onFree(ptr, actual_size);
    Memory::deleteSized(ptr, size, align);
}

void operator delete[](void * ptr, std::size_t size) noexcept
{
#if USE_JEMALLOC
    if (tryFreeNonJemallocMemoryConditional(ptr))
        return;
#endif
    AllocationTrace trace;
    std::size_t actual_size = Memory::untrackMemory(ptr, trace, size);
    trace.onFree(ptr, actual_size);
    Memory::deleteSized(ptr, size);
}

void operator delete[](void * ptr, std::size_t size, std::align_val_t align) noexcept
{
#if USE_JEMALLOC
    if (tryFreeNonJemallocMemoryConditional(ptr))
        return;
#endif
    AllocationTrace trace;
    std::size_t actual_size = Memory::untrackMemory(ptr, trace, size, align);
    trace.onFree(ptr, actual_size);
    Memory::deleteSized(ptr, size, align);
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-identifier"

extern "C" void * __wrap_malloc(size_t size) // NOLINT
{
    AllocationTrace trace;
    std::size_t actual_size = Memory::trackMemoryFromC(size, trace);
#if USE_JEMALLOC
    void * ptr = je_malloc(size);
#else
    void * ptr = __real_malloc(size);
#endif
    if (unlikely(!ptr))
    {
        trace = CurrentMemoryTracker::free(actual_size);
        return nullptr;
    }
    trace.onAlloc(ptr, actual_size);
    return ptr;
}

extern "C" void * __wrap_calloc(size_t number_of_members, size_t size) // NOLINT
{
    size_t real_size = 0;
    if (__builtin_mul_overflow(number_of_members, size, &real_size))
        return nullptr;

    AllocationTrace trace;
    size_t actual_size = Memory::trackMemoryFromC(real_size, trace);
#if USE_JEMALLOC
    void * res = je_calloc(number_of_members, size);
#else
    void * res = __real_calloc(number_of_members, size);
#endif
    if (unlikely(!res))
    {
        trace = CurrentMemoryTracker::free(actual_size);
        return nullptr;
    }
    trace.onAlloc(res, actual_size);
    return res;
}

extern "C" void * __wrap_realloc(void * ptr, size_t size) // NOLINT
{
    if (ptr)
    {
#if USE_JEMALLOC
        if (!isJemallocMemory(ptr))
        {
            return __real_realloc(ptr, size);
        }
#endif
        AllocationTrace trace;
        size_t actual_size = Memory::untrackMemory(ptr, trace);
        trace.onFree(ptr, actual_size);
    }
    AllocationTrace trace;
    size_t actual_size = Memory::trackMemoryFromC(size, trace);
#if USE_JEMALLOC
    void * res = je_realloc(ptr, size);
#else
    void * res = __real_realloc(ptr, size);
#endif
    if (unlikely(!res))
    {
        trace = CurrentMemoryTracker::free(actual_size);
        return nullptr;
    }
    trace.onAlloc(res, actual_size);
    return res;
}

extern "C" int __wrap_posix_memalign(void ** memptr, size_t alignment, size_t size) // NOLINT
{
    AllocationTrace trace;
    size_t actual_size = Memory::trackMemoryFromC(size, trace, static_cast<std::align_val_t>(alignment));
#if USE_JEMALLOC
    int res = je_posix_memalign(memptr, alignment, size);
#else
    int res = __real_posix_memalign(memptr, alignment, size);
#endif
    if (unlikely(res != 0))
    {
        trace = CurrentMemoryTracker::free(actual_size);
        return res;
    }
    trace.onAlloc(*memptr, actual_size);
    return res;
}

extern "C" void * __wrap_aligned_alloc(size_t alignment, size_t size) // NOLINT
{
    AllocationTrace trace;
    size_t actual_size = Memory::trackMemoryFromC(size, trace, static_cast<std::align_val_t>(alignment));
#if USE_JEMALLOC
    void * res = je_aligned_alloc(alignment, size);
#else
    void * res = __real_aligned_alloc(alignment, size);
#endif
    if (unlikely(!res))
    {
        trace = CurrentMemoryTracker::free(actual_size);
        return nullptr;
    }
    trace.onAlloc(res, actual_size);
    return res;
}

#if !defined(OS_FREEBSD)
extern "C" void * __wrap_valloc(size_t size) // NOLINT
{
    AllocationTrace trace;
    size_t actual_size = Memory::trackMemoryFromC(size, trace);
#if USE_JEMALLOC
    void * res = je_valloc(size);
#else
    void * res = __real_valloc(size);
#endif
    if (unlikely(!res))
    {
        trace = CurrentMemoryTracker::free(actual_size);
        return nullptr;
    }
    trace.onAlloc(res, actual_size);
    return res;
}

extern "C" void * __wrap_reallocarray(void * ptr, size_t number_of_members, size_t size) // NOLINT
{
    size_t real_size = 0;
    if (__builtin_mul_overflow(number_of_members, size, &real_size))
        return nullptr;
#if USE_JEMALLOC
    return je_realloc(ptr, real_size);
#else
    return __wrap_realloc(ptr, real_size);
#endif
}

extern "C" void __wrap_free(void * ptr) // NOLINT
{
#if USE_JEMALLOC
    if (tryFreeNonJemallocMemory(ptr))
        return;
#endif
    AllocationTrace trace;
    size_t actual_size = Memory::untrackMemory(ptr, trace);
    trace.onFree(ptr, actual_size);
#if USE_JEMALLOC
    je_free(ptr);
#else
    __real_free(ptr);
#endif
}

#if !defined(OS_DARWIN) && !defined(OS_FREEBSD)
extern "C" void * __wrap_memalign(size_t alignment, size_t size) // NOLINT
{
    AllocationTrace trace;
    size_t actual_size = Memory::trackMemoryFromC(size, trace, static_cast<std::align_val_t>(alignment));
#if USE_JEMALLOC
    void * res = je_memalign(alignment, size);
#else
    void * res = __real_memalign(alignment, size);
#endif
    if (unlikely(!res))
    {
        trace = CurrentMemoryTracker::free(actual_size);
        return nullptr;
    }
    trace.onAlloc(res, actual_size);
    return res;
}
#endif

#if !defined(USE_MUSL) && defined(OS_LINUX)
extern "C" void * __wrap_pvalloc(size_t size) // NOLINT
{
    AllocationTrace trace;
    size_t actual_size = Memory::trackMemoryFromC(size, trace);
#if USE_JEMALLOC
    void * res = je_pvalloc(size);
#else
    void * res = __real_pvalloc(size);
#endif
    if (unlikely(!res))
    {
        trace = CurrentMemoryTracker::free(actual_size);
        return nullptr;
    }
    trace.onAlloc(res, actual_size);
    return res;
}
#endif

#endif // !defined(OS_FREEBSD)

#pragma clang diagnostic pop
