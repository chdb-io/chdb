#pragma once

#include <vector>
#include <Processors/Formats/IOutputFormat.h>

namespace DB
{

class NullWriteBuffer;
class PandasDataFrameBuilder;

/// OutputFormat that collects all chunks into memory for further processing
/// Does not write to WriteBuffer, instead accumulates data for conversion to pandas DataFrame objects
class ChunkCollectorOutputFormat : public IOutputFormat
{
public:
    ChunkCollectorOutputFormat(const Block & header, PandasDataFrameBuilder & builder);

    String getName() const override { return "ChunkCollectorOutputFormat"; }

    void onCancel() noexcept override
    {
        chunks.clear();
    }

protected:
    void consume(Chunk chunk) override;

    void consumeTotals(Chunk totals) override;

    void consumeExtremes(Chunk extremes) override;

    void finalizeImpl() override;

private:
    std::vector<Chunk> chunks;

    PandasDataFrameBuilder & dataframe_builder;

    /// Is not used.
    static NullWriteBuffer out;
};

}
