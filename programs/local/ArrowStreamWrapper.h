#pragma once

#include <memory>
#include <arrow/c/abi.h>
#include <pybind11/pybind11.h>
#include <Core/Names.h>

namespace CHDB
{

/// Wrapper for Arrow C Data Interface structures with RAII resource management
class ArrowSchemaWrapper
{
public:
    ArrowSchema arrow_schema;

    ArrowSchemaWrapper() {
        arrow_schema.release = nullptr;
    }

    ~ArrowSchemaWrapper();

    /// Non-copyable but moveable
    ArrowSchemaWrapper(const ArrowSchemaWrapper &) = delete;
    ArrowSchemaWrapper & operator=(const ArrowSchemaWrapper &) = delete;
    ArrowSchemaWrapper(ArrowSchemaWrapper && other) noexcept;
    ArrowSchemaWrapper & operator=(ArrowSchemaWrapper && other) noexcept;
};

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

    ArrowArrayStreamWrapper() {
        arrow_array_stream.release = nullptr;
    }

    ~ArrowArrayStreamWrapper();

    // Non-copyable but moveable
    ArrowArrayStreamWrapper(const ArrowArrayStreamWrapper&) = delete;
    ArrowArrayStreamWrapper& operator=(const ArrowArrayStreamWrapper&) = delete;
    ArrowArrayStreamWrapper(ArrowArrayStreamWrapper&& other) noexcept;
    ArrowArrayStreamWrapper& operator=(ArrowArrayStreamWrapper&& other) noexcept;

    /// Get schema from the stream
    void getSchema(ArrowSchemaWrapper& schema);

    /// Get next chunk from the stream
    std::unique_ptr<ArrowArrayWrapper> getNextChunk();

    /// Get last error message
    const char* getError();

    /// Check if stream is valid
    bool isValid() const { return arrow_array_stream.release != nullptr; }
};

/// Factory class for creating ArrowArrayStream from Python objects
class PyArrowStreamFactory
{
public:
    static std::unique_ptr<ArrowArrayStreamWrapper> createFromPyObject(
        pybind11::object & py_obj,
        const DB::Names & column_names);

private:
    static std::unique_ptr<ArrowArrayStreamWrapper> createFromTable(
        pybind11::object & table,
        const DB::Names & column_names);
};

} // namespace CHDB
