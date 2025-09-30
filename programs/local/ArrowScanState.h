#pragma once

#include "ArrowStreamWrapper.h"

#include <memory>
#include <arrow/type.h>

namespace CHDB
{

/// Scan state for each stream - shared between ArrowTableReader and ArrowStreamReader
struct ArrowScanState
{
    /// Current Arrow array being processed (for ArrowTableReader)
    std::unique_ptr<ArrowArrayWrapper> current_array;
    /// Current offset within the array
    size_t current_offset = 0;
    /// Whether this stream is exhausted
    bool exhausted = false;
    /// Cached imported RecordBatch to avoid repeated imports
    std::shared_ptr<arrow::RecordBatch> cached_record_batch;

    virtual ~ArrowScanState() = default;

    virtual void reset()
    {
        current_array.reset();
        current_offset = 0;
        exhausted = false;
        cached_record_batch.reset();
    }
};

} // namespace CHDB
