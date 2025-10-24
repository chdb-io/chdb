#include "ArrowStreamWrapper.h"

#include <Common/Exception.h>
#include <base/defines.h>

namespace DB
{

namespace ErrorCodes
{
extern const int BAD_ARGUMENTS;
}

}

using namespace DB;

namespace CHDB
{

/// ArrowArrayWrapper implementation
ArrowArrayWrapper::~ArrowArrayWrapper()
{
    if (arrow_array.release)
    {
        arrow_array.release(&arrow_array);
    }
}

ArrowArrayWrapper::ArrowArrayWrapper(ArrowArrayWrapper && other) noexcept
    : arrow_array(other.arrow_array)
{
    other.arrow_array.release = nullptr;
}

ArrowArrayWrapper & ArrowArrayWrapper::operator=(ArrowArrayWrapper && other) noexcept
{
    if (this != &other)
    {
        if (arrow_array.release)
        {
            arrow_array.release(&arrow_array);
        }
        arrow_array = other.arrow_array;
        other.arrow_array.release = nullptr;
    }
    return *this;
}

/// ArrowArrayStreamWrapper implementation
ArrowArrayStreamWrapper::~ArrowArrayStreamWrapper()
{
    if (should_release_on_destroy && arrow_array_stream.release)
    {
        arrow_array_stream.release(&arrow_array_stream);
    }
}

ArrowArrayStreamWrapper::ArrowArrayStreamWrapper(ArrowArrayStreamWrapper&& other) noexcept
    : arrow_array_stream(other.arrow_array_stream)
    , should_release_on_destroy(other.should_release_on_destroy)
{
    other.arrow_array_stream.release = nullptr;
    other.should_release_on_destroy = true;
}

ArrowArrayStreamWrapper & ArrowArrayStreamWrapper::operator=(ArrowArrayStreamWrapper && other) noexcept
{
    if (this != &other)
    {
        if (should_release_on_destroy && arrow_array_stream.release)
        {
            arrow_array_stream.release(&arrow_array_stream);
        }
        arrow_array_stream = other.arrow_array_stream;
        should_release_on_destroy = other.should_release_on_destroy;
        other.arrow_array_stream.release = nullptr;
        other.should_release_on_destroy = true;
    }
    return *this;
}

void ArrowArrayStreamWrapper::getSchema(ArrowSchemaWrapper& schema)
{
    if (!isValid())
    {
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "ArrowArrayStream is not valid");
    }

    if (arrow_array_stream.get_schema(&arrow_array_stream, &schema.arrow_schema) != 0)
    {
        throw Exception(ErrorCodes::BAD_ARGUMENTS,
                        "Failed to get schema from ArrowArrayStream: {}", getError());
    }

    if (!schema.arrow_schema.release)
    {
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "Released schema returned from ArrowArrayStream");
    }
}

std::unique_ptr<ArrowArrayWrapper> ArrowArrayStreamWrapper::getNextChunk()
{
    chassert(isValid());

    auto chunk = std::make_unique<ArrowArrayWrapper>();

    /// Get next non-empty chunk, skipping empty ones
    do
    {
        chunk->reset();
        if (arrow_array_stream.get_next(&arrow_array_stream, &chunk->arrow_array) != 0)
        {
            throw Exception(ErrorCodes::BAD_ARGUMENTS,
                            "Failed to get next chunk from ArrowArrayStream: {}", getError());
        }

        /// Check if we've reached the end of the stream
        if (!chunk->arrow_array.release)
        {
            return nullptr;
        }
    }
    while (chunk->arrow_array.length == 0);

    return chunk;
}

const char* ArrowArrayStreamWrapper::getError()
{
    if (!isValid())
    {
        return "ArrowArrayStream is not valid";
    }

    return arrow_array_stream.get_last_error(&arrow_array_stream);
}

} // namespace CHDB
