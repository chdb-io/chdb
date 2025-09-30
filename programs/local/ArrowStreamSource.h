#pragma once

#include "ArrowTableReader.h"

#include <Processors/ISource.h>
#include <arrow/c/abi.h>
#include <Poco/Logger.h>

namespace DB
{

class ArrowStreamSource : public ISource
{
public:
    ArrowStreamSource(
        const Block & sample_block_,
        CHDB::ArrowTableReaderPtr arrow_table_reader_,
        size_t stream_index_);

    String getName() const override { return "ArrowStream"; }

    Chunk generate() override;

private:
    CHDB::ArrowTableReaderPtr arrow_table_reader;
    Block sample_block;
    size_t stream_index;
    Poco::Logger * logger = &Poco::Logger::get("ArrowStreamSource");
};

}
