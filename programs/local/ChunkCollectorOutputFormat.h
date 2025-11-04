#pragma once

#include <vector>
#include <Core/NamesAndTypes.h>
#include <Processors/Formats/IOutputFormat.h>
#include <Processors/Port.h>

namespace DB
{
class NullWriteBuffer;
}

namespace CHDB
{

class PandasDataFrameBuilder;

/// OutputFormat that collects all chunks into memory for further processing
/// Does not write to WriteBuffer, instead accumulates data for conversion to pandas DataFrame objects
class ChunkCollectorOutputFormat : public DB::IOutputFormat
{
public:
    ChunkCollectorOutputFormat(DB::SharedHeader shared_header, PandasDataFrameBuilder & builder);

    String getName() const override { return "ChunkCollectorOutputFormat"; }

    void onCancel() noexcept override
    {
        chunks.clear();
    }

protected:
    void consume(DB::Chunk chunk) override;

    void consumeTotals(DB::Chunk totals) override;

    void consumeExtremes(DB::Chunk extremes) override;

    void finalizeImpl() override;

private:
    std::vector<DB::Chunk> chunks;

    PandasDataFrameBuilder & dataframe_builder;

    static DB::NullWriteBuffer out;
};

/// Registration function to be called during initialization
void registerDataFrameOutputFormat();

/// Get the global dataframe builder
PandasDataFrameBuilder & getGlobalDataFrameBuilder();

/// Set the global dataframe builder
void setGlobalDataFrameBuilder(std::shared_ptr<PandasDataFrameBuilder> builder);

/// Reset the global dataframe builder
void resetGlobalDataFrameBuilder();

}
