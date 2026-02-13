#include <Common/MemoryTracker.h>
#include <Common/CurrentThread.h>
#include <Common/MemoryTrackerBlockerInThread.h>

#include <Common/CurrentMemoryTracker.h>

std::atomic<bool> g_memory_tracking_disabled{false};

#ifdef MEMORY_TRACKER_DEBUG_CHECKS
thread_local bool memory_tracker_always_throw_logical_error_on_allocation = false;
#endif

extern bool chdb_embedded_server_initialized;

namespace DB
{
namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}
}

namespace
{

MemoryTracker * getMemoryTracker()
{
    if (unlikely(g_memory_tracking_disabled.load(std::memory_order_relaxed)))
        return nullptr;

    if (auto * thread_memory_tracker = DB::CurrentThread::getMemoryTracker())
        return thread_memory_tracker;

    if (likely(chdb_embedded_server_initialized))
        return &total_memory_tracker;

    /// Note, we cannot use total_memory_tracker earlier (i.e. just after static variable initialized without this check),
    /// since the initialization order of static objects is not defined, and total_memory_tracker may not be initialized yet.
    /// So here we relying on MainThreadStatus initialization.
    if (DB::MainThreadStatus::initialized())
        return &total_memory_tracker;

    return nullptr;
}

}

using DB::current_thread;

AllocationTrace CurrentMemoryTracker::allocImpl(Int64 size, bool throw_if_memory_exceeded)
{
#ifdef MEMORY_TRACKER_DEBUG_CHECKS
    if (unlikely(memory_tracker_always_throw_logical_error_on_allocation))
    {
        memory_tracker_always_throw_logical_error_on_allocation = false;
        throw DB::Exception(DB::ErrorCodes::LOGICAL_ERROR, "Memory tracker: allocations not allowed.");
    }
#endif

    if (auto * memory_tracker = getMemoryTracker())
    {
        /// Ignore untracked_memory if:
        ///  * total_memory_tracker only, or
        ///  * MemoryTrackerBlockerInThread is active.
        ///    memory_tracker->allocImpl needs to be called for these bytes with the same blocker
        ///    state as we currently have.
        ///    E.g. suppose allocImpl is called twice: first for 2 MB with blocker set to
        ///    VariableContext::User, then for 3 MB with no blocker. This should increase the
        ///    Global memory tracker by 5 MB and the User memory tracker by 3 MB. So we can't group
        ///    these two calls into one memory_tracker->allocImpl call. Without this `if`, the first
        ///    allocImpl call would increment untracked_memory, and the second call would
        ///    incorrectly report all 5 MB with no blocker, so the User memory tracker would be
        ///    incorrectly increased by 5 MB instead of 3 MB.
        ///    (Alternatively, we could maintain `untracked_memory` value separately for each
        ///     possible blocker state, i.e. per VariableContext.)
        if (!current_thread)
        {
            /// total_memory_tracker only, ignore untracked_memory
            return memory_tracker->allocImpl(size, throw_if_memory_exceeded);
        }
        else
        {
            VariableContext blocker_level = MemoryTrackerBlockerInThread::getLevel();
            if (blocker_level != current_thread->untracked_memory_blocker_level) 
            {
                current_thread->flushUntrackedMemory();
            }
            current_thread->untracked_memory_blocker_level = blocker_level;
            Int64 previous_untracked_memory = current_thread->untracked_memory;
            current_thread->untracked_memory += size;
            if (current_thread->untracked_memory > current_thread->untracked_memory_limit)
            {
                Int64 current_untracked_memory = current_thread->untracked_memory;
                current_thread->untracked_memory = 0;

                try
                {
                    return memory_tracker->allocImpl(current_untracked_memory, throw_if_memory_exceeded);
                }
                catch (...)
                {
                    current_thread->untracked_memory += previous_untracked_memory;
                    throw;
                }
            }
        }

        /// return AllocationTrace(memory_tracker->getSampleProbability(size));
        return AllocationTrace(0);
    }

    return AllocationTrace(0);
}

void CurrentMemoryTracker::check()
{
    if (auto * memory_tracker = getMemoryTracker())
        std::ignore = memory_tracker->allocImpl(0, true);
}

AllocationTrace CurrentMemoryTracker::alloc(Int64 size)
{
    bool throw_if_memory_exceeded = true;
    return allocImpl(size, throw_if_memory_exceeded);
}

AllocationTrace CurrentMemoryTracker::allocNoThrow(Int64 size)
{
    bool throw_if_memory_exceeded = false;
    return allocImpl(size, throw_if_memory_exceeded);
}

AllocationTrace CurrentMemoryTracker::free(Int64 size)
{
    if (auto * memory_tracker = getMemoryTracker())
    {
        if (!current_thread)
        {
            return memory_tracker->free(size);
        }

        VariableContext blocker_level = MemoryTrackerBlockerInThread::getLevel();
        if (blocker_level != current_thread->untracked_memory_blocker_level)
        {
            current_thread->flushUntrackedMemory();
        }
        current_thread->untracked_memory_blocker_level = blocker_level;

        current_thread->untracked_memory -= size;
        if (current_thread->untracked_memory < -current_thread->untracked_memory_limit)
        {
            Int64 untracked_memory = current_thread->untracked_memory;
            current_thread->untracked_memory = 0;
            return memory_tracker->free(-untracked_memory);
        }

        /// return AllocationTrace(memory_tracker->getSampleProbability(size));
        return AllocationTrace(0);
    }

    return AllocationTrace(0);
}

void CurrentMemoryTracker::injectFault()
{
    if (auto * memory_tracker = getMemoryTracker())
        memory_tracker->injectFault();
}
