#pragma once

#include "ArrowSchema.h"

#include <memory>
#include <arrow/c/abi.h>

namespace CHDB
{

class ArrowArrayWrapper
{
public:
    ArrowArray arrow_array;

    ArrowArrayWrapper()
    {
        reset();
    }

    ~ArrowArrayWrapper();

    void reset()
    {
        arrow_array.length = 0;
        arrow_array.release = nullptr;
    }

    /// Non-copyable but moveable
    ArrowArrayWrapper(const ArrowArrayWrapper &) = delete;
    ArrowArrayWrapper & operator=(const ArrowArrayWrapper &) = delete;
    ArrowArrayWrapper(ArrowArrayWrapper && other) noexcept;
    ArrowArrayWrapper & operator=(ArrowArrayWrapper && other) noexcept;
};

class ArrowArrayStreamWrapper
{
public:
    ArrowArrayStream arrow_array_stream;

    explicit ArrowArrayStreamWrapper(bool should_release = true)
        : should_release_on_destroy(should_release) {
        arrow_array_stream.release = nullptr;
    }

    ~ArrowArrayStreamWrapper();

    /// Non-copyable but moveable
    ArrowArrayStreamWrapper(const ArrowArrayStreamWrapper&) = delete;
    ArrowArrayStreamWrapper& operator=(const ArrowArrayStreamWrapper&) = delete;
    ArrowArrayStreamWrapper(ArrowArrayStreamWrapper&& other) noexcept;
    ArrowArrayStreamWrapper& operator=(ArrowArrayStreamWrapper&& other) noexcept;

    /// Get schema from the stream
    void getSchema(ArrowSchemaWrapper & schema);

    /// Get next chunk from the stream
    std::unique_ptr<ArrowArrayWrapper> getNextChunk();

    /// Get last error message
    const char* getError();

    /// Check if stream is valid
    bool isValid() const { return arrow_array_stream.release != nullptr; }

    /// Set whether to release on destruction
    void setShouldRelease(bool should_release) { should_release_on_destroy = should_release; }

    /// Get whether will release on destruction
    bool getShouldRelease() const { return should_release_on_destroy; }

private:
    bool should_release_on_destroy = true;
};

} // namespace CHDB
