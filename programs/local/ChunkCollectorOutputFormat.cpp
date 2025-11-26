#include "ChunkCollectorOutputFormat.h"

#include <IO/NullWriteBuffer.h>
#include <Processors/Port.h>
#include <Client/ClientBase.h>
#include <base/defines.h>

using namespace DB;

namespace CHDB
{

NullWriteBuffer ChunkCollectorOutputFormat::out;

ChunkCollectorOutputFormat::ChunkCollectorOutputFormat(
    SharedHeader shared_header,
    std::vector<Chunk> & chunks_storage)
    : IOutputFormat(shared_header, out)
    , chunks(chunks_storage)
{}

void ChunkCollectorOutputFormat::consume(Chunk chunk)
{
    chunks.emplace_back(std::move(chunk));
}

void ChunkCollectorOutputFormat::consumeTotals(Chunk totals)
{
    chunks.emplace_back(std::move(totals));
}

void ChunkCollectorOutputFormat::consumeExtremes(Chunk extremes)
{
    chunks.emplace_back(std::move(extremes));
}

void ChunkCollectorOutputFormat::finalizeImpl()
{
}

/// Create ChunkCollectorOutputFormat for use with function pointer
std::shared_ptr<IOutputFormat> createDataFrameOutputFormat(SharedHeader header, std::vector<Chunk> & chunks_storage)
{
    return std::make_shared<ChunkCollectorOutputFormat>(header, chunks_storage);
}

/// Registration function to be called during initialization
void registerDataFrameOutputFormat()
{
    ClientBase::setDataFrameFormatCreator(&createDataFrameOutputFormat);
}

}
