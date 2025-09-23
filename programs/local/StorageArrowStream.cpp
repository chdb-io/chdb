#include "StorageArrowStream.h"
#include "ArrowStreamSource.h"
#include "ArrowStreamWrapper.h"
#include "ArrowTableReader.h"

#include <any>
#include <Formats/FormatFactory.h>
#include <base/defines.h>

namespace DB
{

namespace ErrorCodes
{
extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
}

void registerStorageArrowStream(StorageFactory & factory)
{
    factory.registerStorage(
        "ArrowStream",
        [](const StorageFactory::Arguments & args) -> StoragePtr
        {
            if (args.engine_args.size() != 1)
                throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH, "ArrowStream engine requires 1 argument: ArrowStreamInfo object");

            CHDB::ArrowStreamRegistry::ArrowStreamInfo stream_info = std::any_cast<CHDB::ArrowStreamRegistry::ArrowStreamInfo>(args.engine_args[0]);
            return std::make_shared<StorageArrowStream>(args.table_id, stream_info, args.columns, args.getLocalContext());
        },
        {
            .supports_settings = false,
            .supports_parallel_insert = false,
        });
}

StorageArrowStream::StorageArrowStream(
    const StorageID & storage_id_,
    const CHDB::ArrowStreamRegistry::ArrowStreamInfo & stream_info_,
    const ColumnsDescription & columns_,
    ContextPtr context_)
    : IStorage(storage_id_)
    , WithContext(context_)
    , stream_info(stream_info_)
{
    StorageInMemoryMetadata storage_metadata;
    storage_metadata.setColumns(columns_);
    setInMemoryMetadata(storage_metadata);
}

Pipe StorageArrowStream::read(
    const Names & column_names,
    const StorageSnapshotPtr & storage_snapshot,
    SelectQueryInfo & /*query_info*/,
    ContextPtr /*context*/,
    QueryProcessingStage::Enum /*processed_stage*/,
    size_t max_block_size,
    size_t num_streams)
{
    chassert(stream_info.stream);
    storage_snapshot->check(column_names);

    Block sample_block = prepareSampleBlock(column_names, storage_snapshot);
    auto format_settings = getFormatSettings(getContext());

    /// Create ArrowArrayStreamWrapper from the registered stream
    auto arrow_stream_wrapper = std::make_unique<CHDB::ArrowArrayStreamWrapper>(false);
    arrow_stream_wrapper->arrow_array_stream = *stream_info.stream;

    auto arrow_table_reader = std::make_shared<CHDB::ArrowTableReader>(
        std::move(arrow_stream_wrapper),
        sample_block,
        format_settings,
        num_streams,
        max_block_size
    );

    Pipes pipes;
    for (size_t stream = 0; stream < num_streams; ++stream)
    {
        pipes.emplace_back(std::make_shared<ArrowStreamSource>(
            sample_block, arrow_table_reader, stream));
    }
    return Pipe::unitePipes(std::move(pipes));
}

Block StorageArrowStream::prepareSampleBlock(const Names & column_names, const StorageSnapshotPtr & storage_snapshot)
{
    Block sample_block;
    for (const String & column_name : column_names)
    {
        auto column_data = storage_snapshot->metadata->getColumns().getPhysical(column_name);
        sample_block.insert({column_data.type, column_data.name});
    }
    return sample_block;
}

}
