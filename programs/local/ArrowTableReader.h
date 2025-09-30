#pragma once

#include "ArrowScanState.h"
#include "ArrowStreamWrapper.h"

#include <Core/Block.h>
#include <Processors/Chunk.h>
#include <Formats/FormatSettings.h>
#include <arrow/type.h>

namespace CHDB
{

class ArrowTableReader;
using ArrowTableReaderPtr = std::shared_ptr<ArrowTableReader>;

class ArrowTableReader
{
public:
    ArrowTableReader(
        std::unique_ptr<ArrowArrayStreamWrapper> arrow_stream_,
        const DB::Block & sample_block_,
        const DB::FormatSettings & format_settings_,
        size_t num_streams_,
        size_t max_block_size_);

    ~ArrowTableReader() = default;

    /// Read next chunk from the specified stream
    DB::Chunk readNextChunk(size_t stream_index);

private:
    /// Initialize the Arrow stream from ArrowArrayStreamWrapper
    void initializeStream();

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

    /// Mutex for thread-safe access to arrow_stream
    mutable std::mutex stream_mutex;
};

} // namespace CHDB
