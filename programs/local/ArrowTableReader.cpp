#include "ArrowTableReader.h"

#include <Common/Exception.h>
#include <Processors/Formats/Impl/ArrowColumnToCHColumn.h>
#include <arrow/c/bridge.h>
#include <base/defines.h>
#include <pybind11/pybind11.h>
#include <pybind11/gil.h>

namespace DB
{

namespace ErrorCodes
{
extern const int PY_EXCEPTION_OCCURED;
}

}

namespace py = pybind11;
using namespace DB;

namespace CHDB
{

ArrowTableReader::ArrowTableReader(
    py::object & data_source_,
    const DB::Block & sample_block_,
    const DB::FormatSettings & format_settings_,
    size_t num_streams_,
    size_t max_block_size_)
    : sample_block(sample_block_),
    format_settings(format_settings_),
    num_streams(num_streams_),
    max_block_size(max_block_size_),
    scan_states(num_streams_)
{
    initializeStream(data_source_);
}

void ArrowTableReader::initializeStream(py::object & data_source_)
{
    try
    {
        /// Create Arrow stream from Python object
        arrow_stream = PyArrowStreamFactory::createFromPyObject(data_source_, sample_block.getNames());

        if (!arrow_stream || !arrow_stream->isValid())
        {
            throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED,
                            "Failed to create valid ArrowArrayStream from Python object");
        }
    }
    catch (const py::error_already_set & e)
    {
        throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED,
                        "Failed to initialize Arrow stream from Python object: {}", e.what());
    }

    /// Get schema from stream
    arrow_stream->getSchema(schema);
    auto arrow_schema_result = arrow::ImportSchema(&schema.arrow_schema);
    if (!arrow_schema_result.ok())
    {
        throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED,
                        "Failed to import Arrow schema during initialization: {}", arrow_schema_result.status().message());
    }
    cached_arrow_schema = arrow_schema_result.ValueOrDie();
}

Chunk ArrowTableReader::readNextChunk(size_t stream_index)
{
    if (stream_index >= num_streams)
    {
        throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED,
                        "Stream index {} is out of range [0, {})", stream_index, num_streams);
    }

    auto & state = scan_states[stream_index];

    if (state.exhausted)
    {
        return {};
    }

    try
    {
        /// If we don't have a current array or it's exhausted, get the next one
        if (!state.current_array || state.current_offset >= static_cast<size_t>(state.current_array->arrow_array.length))
        {
            auto next_array = getNextArrowArray();
            if (!next_array)
            {
                state.exhausted = true;
                return {};
            }
            state.current_array = std::move(next_array);
            state.current_offset = 0;
            state.cached_record_batch.reset();
        }

        /// Calculate how many rows to read from current array
        size_t available_rows = static_cast<size_t>(state.current_array->arrow_array.length) - state.current_offset;
        size_t rows_to_read = std::min(max_block_size, available_rows);

        /// Convert the slice to chunk
        auto chunk = convertArrowArrayToChunk(*state.current_array, state.current_offset, rows_to_read, stream_index);

        /// Update offset
        state.current_offset += rows_to_read;

        return chunk;
    }
    catch (const Exception &)
    {
        state.exhausted = true;
        throw;
    }
}

std::unique_ptr<ArrowArrayWrapper> ArrowTableReader::getNextArrowArray()
{
    std::lock_guard<std::mutex> lock(stream_mutex);

    if (global_stream_exhausted || !arrow_stream || !arrow_stream->isValid())
    {
        return nullptr;
    }

    try
    {
        auto arrow_array = arrow_stream->getNextChunk();

        if (!arrow_array || arrow_array->arrow_array.length == 0)
        {
            global_stream_exhausted = true;
            return nullptr;
        }

        return arrow_array;
    }
    catch (const Exception &)
    {
        global_stream_exhausted = true;
        throw;
    }
}

Chunk ArrowTableReader::convertArrowArrayToChunk(const ArrowArrayWrapper & arrow_array_wrapper, size_t offset, size_t count, size_t stream_index)
{
    chassert(arrow_array_wrapper.arrow_array.length && count && offset < arrow_array_wrapper.arrow_array.length);
    chassert(count <= arrow_array_wrapper.arrow_array.length - offset);
    chassert(stream_index < num_streams);

    auto & state = scan_states[stream_index];
    std::shared_ptr<arrow::RecordBatch> record_batch;

    /// Check if we have a cached RecordBatch for this ArrowArray
    if (!state.cached_record_batch)
    {
        /// Import the full ArrowArray to RecordBatch and cache it
        ArrowArray array_copy = arrow_array_wrapper.arrow_array;

        /// Set a dummy release function to prevent Arrow from freeing the underlying data
        static auto dummy_release = [](ArrowArray* array)
        {
            // No-op: ArrowArrayWrapper will handle the actual cleanup
            // But we must set release to nullptr to follow Arrow C ABI convention
            array->release = nullptr;
        };
        array_copy.release = dummy_release;

        /// Import the full Arrow array to Arrow RecordBatch
        auto arrow_batch_result = arrow::ImportRecordBatch(&array_copy, cached_arrow_schema);
        if (!arrow_batch_result.ok())
        {
            throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED,
                            "Failed to import Arrow RecordBatch: {}", arrow_batch_result.status().message());
        }

        state.cached_record_batch = arrow_batch_result.ValueOrDie();
    }

    /// Use the cached RecordBatch and slice it
    record_batch = state.cached_record_batch;
    auto sliced_batch = record_batch->Slice(offset, count);
    auto arrow_table = arrow::Table::FromRecordBatches({sliced_batch}).ValueOrDie();

    /// Use ArrowColumnToCHColumn to convert the batch
    ArrowColumnToCHColumn converter(
        sample_block,
        "Arrow",
        format_settings.arrow.allow_missing_columns,
        format_settings.null_as_default,
        format_settings.date_time_overflow_behavior,
        format_settings.parquet.allow_geoparquet_parser,
        format_settings.arrow.case_insensitive_column_matching,
        false
    );

    return converter.arrowTableToCHChunk(arrow_table, sliced_batch->num_rows(), nullptr);
}

} // namespace CHDB
