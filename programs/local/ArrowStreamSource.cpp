#include "ArrowStreamSource.h"
#include <base/defines.h>

#include <Common/Exception.h>

namespace DB
{

namespace ErrorCodes
{
extern const int BAD_ARGUMENTS;
}

ArrowStreamSource::ArrowStreamSource(
    const Block & sample_block_,
    CHDB::ArrowTableReaderPtr arrow_table_reader_,
    size_t stream_index_)
    : ISource(std::make_shared<Block>(sample_block_.cloneEmpty()))
    , arrow_table_reader(arrow_table_reader_)
    , sample_block(sample_block_)
    , stream_index(stream_index_)
{
}

Chunk ArrowStreamSource::generate()
{
    chassert(arrow_table_reader);

    if (sample_block.getNames().empty())
        return {};

    try
    {
        auto chunk = arrow_table_reader->readNextChunk(stream_index);
        return chunk;
    }
    catch (const Exception &)
    {
        throw;
    }
    catch (const std::exception & e)
    {
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "ArrowStreamSource error: {}", e.what());
    }
    catch (...)
    {
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "ArrowStreamSource unknown exception");
    }
}

}
