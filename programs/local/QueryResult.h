#pragma once

#include <memory>
#include <vector>
#include <utility>

#include "config.h"
#include <base/types.h>

#if USE_PYTHON
#include <Processors/Chunk.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
namespace DB
{
    class Block;
}
#endif

namespace CHDB
{

enum class QueryResultType : uint8_t
{
    RESULT_TYPE_MATERIALIZED = 0,
    RESULT_TYPE_STREAMING = 1,
    RESULT_TYPE_CHUNK = 2,
    RESULT_TYPE_DATAFRAME = 3,
    RESULT_TYPE_NONE = 4
};

class QueryResult
{
public:
    explicit QueryResult(QueryResultType type, String error_message_ = "")
        : result_type(type), error_message(std::move(error_message_))
    {}

    virtual ~QueryResult() = default;

    QueryResultType getType() const { return result_type; }
    const String & getError() const { return error_message; }
    virtual bool isEmpty() const = 0;

protected:
    QueryResultType result_type;
    String error_message;
};

class StreamQueryResult : public QueryResult
{
public:
    explicit StreamQueryResult(String error_message_ = "")
        : QueryResult(QueryResultType::RESULT_TYPE_STREAMING, std::move(error_message_))
    {}

    bool isEmpty() const override
    {
        return false;
    }
};

using ResultBuffer = std::unique_ptr<std::vector<char>>;

class MaterializedQueryResult : public QueryResult
{
public:
    explicit MaterializedQueryResult(
        ResultBuffer result_buffer_,
        double elapsed_,
        uint64_t rows_read_,
        uint64_t bytes_read_,
        uint64_t storage_rows_read_,
        uint64_t storage_bytes_read_)
        : QueryResult(QueryResultType::RESULT_TYPE_MATERIALIZED),
        result_buffer(std::move(result_buffer_)),
        elapsed(elapsed_),
        rows_read(rows_read_),
        bytes_read(bytes_read_),
        storage_rows_read(storage_rows_read_),
        storage_bytes_read(storage_bytes_read_)
    {}

    explicit MaterializedQueryResult(String error_message_)
        : QueryResult(QueryResultType::RESULT_TYPE_MATERIALIZED, std::move(error_message_))
    {}

    bool isEmpty() const override
    {
        return rows_read == 0;
    }

    String string()
    {
        if (!result_buffer)
            return {};

        return String(result_buffer->begin(), result_buffer->end());
    }

public:
    ResultBuffer result_buffer;
    double elapsed;
    uint64_t rows_read;
    uint64_t bytes_read;
    uint64_t storage_rows_read;
    uint64_t storage_bytes_read;
};

#if USE_PYTHON
class ChunkQueryResult : public QueryResult
{
public:
    explicit ChunkQueryResult(
        std::vector<DB::Chunk> chunks_,
        std::shared_ptr<const DB::Block> header_,
        double elapsed_,
        uint64_t rows_read_,
        uint64_t bytes_read_,
        uint64_t storage_rows_read_,
        uint64_t storage_bytes_read_)
        : QueryResult(QueryResultType::RESULT_TYPE_CHUNK),
        chunks(std::move(chunks_)),
        header(header_),
        elapsed(elapsed_),
        rows_read(rows_read_),
        bytes_read(bytes_read_),
        storage_rows_read(storage_rows_read_),
        storage_bytes_read(storage_bytes_read_)
    {}

    explicit ChunkQueryResult(String error_message_)
        : QueryResult(QueryResultType::RESULT_TYPE_CHUNK, std::move(error_message_))
    {}

    bool isEmpty() const override
    {
        return rows_read == 0;
    }

public:
    std::vector<DB::Chunk> chunks;
    std::shared_ptr<const DB::Block> header;
    double elapsed;
    uint64_t rows_read;
    uint64_t bytes_read;
    uint64_t storage_rows_read;
    uint64_t storage_bytes_read;
};

class DataFrameQueryResult : public QueryResult
{
public:
    explicit DataFrameQueryResult(
        py::handle dataframe_,
        uint64_t rows_read)
        : QueryResult(QueryResultType::RESULT_TYPE_DATAFRAME),
        dataframe(dataframe_),
        is_empty(rows_read == 0)
    {}

    bool isEmpty() const override
    {
        return is_empty;
    }

    py::handle dataframe;
    bool is_empty;
};
#endif

using QueryResultPtr = std::unique_ptr<QueryResult>;
using MaterializedQueryResultPtr = std::unique_ptr<MaterializedQueryResult>;
using StreamQueryResultPtr = std::unique_ptr<StreamQueryResult>;
#if USE_PYTHON
using ChunkQueryResultPtr = std::unique_ptr<ChunkQueryResult>;
using DataFrameQueryResultPtr = std::unique_ptr<DataFrameQueryResult>;
#endif

} // namespace CHDB
