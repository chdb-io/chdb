#pragma once

#include "ArrowStreamWrapper.h"

#include <Core/Block.h>
#include <Processors/Chunk.h>
#include <Formats/FormatSettings.h>
#include <pybind11/pybind11.h>
#include <arrow/type.h>

namespace CHDB
{

/// Scan state for each stream
struct ArrowScanState
{
    /// Current Arrow array being processed
    std::unique_ptr<ArrowArrayWrapper> current_array;
    /// Current offset within the array
    size_t current_offset = 0;
    /// Whether this stream is exhausted
    bool exhausted = false;
    /// Cached imported RecordBatch to avoid repeated imports
    std::shared_ptr<arrow::RecordBatch> cached_record_batch;

    void reset()
    {
        current_array.reset();
        current_offset = 0;
        exhausted = false;
        cached_record_batch.reset();
    }
};

class ArrowTableReader;
using ArrowTableReaderPtr = std::shared_ptr<ArrowTableReader>;

class ArrowTableReader
{
public:
    ArrowTableReader(
        pybind11::object & data_source_,
        const DB::Block & sample_block_,
        const DB::FormatSettings & format_settings_,
        size_t num_streams_,
        size_t max_block_size_);

    ~ArrowTableReader() = default;

    /// Read next chunk from the specified stream
    DB::Chunk readNextChunk(size_t stream_index);

private:
    /// Initialize the Arrow stream from Python object
    void initializeStream(pybind11::object & data_source_);

    /// Convert Arrow array slice to ClickHouse chunk
    DB::Chunk convertArrowArrayToChunk(const ArrowArrayWrapper & arrow_array, size_t offset, size_t count, size_t stream_index);

    /// Get next Arrow array from stream
    std::unique_ptr<ArrowArrayWrapper> getNextArrowArray();

    DB::Block sample_block;
    DB::FormatSettings format_settings;
    std::unique_ptr<ArrowArrayStreamWrapper> arrow_stream;
    ArrowSchemaWrapper schema;

    /// Cached Arrow schema to avoid repeated imports
    std::shared_ptr<arrow::Schema> cached_arrow_schema;

    /// Multi-stream scanning parameters
    size_t num_streams;
    size_t max_block_size;

    /// Scan states for each stream
    std::vector<ArrowScanState> scan_states;

    /// Global stream state
    bool global_stream_exhausted = false;
    size_t total_rows_hint = 0;

    /// Mutex for thread-safe access to arrow_stream
    mutable std::mutex stream_mutex;
};

} // namespace CHDB
