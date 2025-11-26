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

class ChunkCollectorOutputFormat : public DB::IOutputFormat
{
public:
    ChunkCollectorOutputFormat(DB::SharedHeader shared_header, std::vector<DB::Chunk> & chunks_storage);

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
    std::vector<DB::Chunk> & chunks;

    static DB::NullWriteBuffer out;
};

/// Registration function to be called during initialization
void registerDataFrameOutputFormat();

}
