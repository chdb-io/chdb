#include "ChunkCollectorOutputFormat.h"
#include "PandasDataFrameBuilder.h"

#include <IO/NullWriteBuffer.h>
#include <Processors/Port.h>
#include <base/defines.h>

namespace DB
{

NullWriteBuffer ChunkCollectorOutputFormat::out;

ChunkCollectorOutputFormat::ChunkCollectorOutputFormat(
    const Block & header,
    PandasDataFrameBuilder & builder)
    : IOutputFormat(header, out)
    , dataframe_builder(builder)
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
    // Add all collected chunks to the builder
    for (const auto & chunk : chunks)
    {
        dataframe_builder.addChunk(chunk);
    }

    // Finalize the DataFrame generation
    dataframe_builder.finalize();

    chunks.clear();
}

}
