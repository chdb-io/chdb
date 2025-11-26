#include "ChunkCollectorOutputFormat.h"
#include "PandasDataFrameBuilder.h"

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
    PandasDataFrameBuilder & builder)
    : IOutputFormat(shared_header, out)
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

/// Global dataframe builder
static std::shared_ptr<PandasDataFrameBuilder> g_dataframe_builder = nullptr;

PandasDataFrameBuilder * getGlobalDataFrameBuilder()
{
    return g_dataframe_builder.get();
}

void setGlobalDataFrameBuilder(std::shared_ptr<PandasDataFrameBuilder> builder)
{
    g_dataframe_builder = builder;
}

void resetGlobalDataFrameBuilder()
{
    if (g_dataframe_builder)
    {
        py::gil_scoped_acquire acquire;
        g_dataframe_builder.reset();
    }
}

/// create ChunkCollectorOutputFormat for use with function pointer
std::shared_ptr<IOutputFormat> createDataFrameOutputFormat(SharedHeader header)
{
    /// Create a PandasDataFrameBuilder and set it globally
    auto dataframe_builder = std::make_shared<PandasDataFrameBuilder>(*header);
    resetGlobalDataFrameBuilder();
    setGlobalDataFrameBuilder(dataframe_builder);

    /// Create and return the format with the builder
    return std::make_shared<ChunkCollectorOutputFormat>(header, *dataframe_builder);
}

/// Registration function to be called during initialization
void registerDataFrameOutputFormat()
{
    ClientBase::setDataFrameFormatCreator(&createDataFrameOutputFormat);
}

}
