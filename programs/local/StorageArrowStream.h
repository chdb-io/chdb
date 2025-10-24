#pragma once

#include "ArrowStreamRegistry.h"

#include <Storages/IStorage.h>
#include <Storages/StorageFactory.h>
#include <QueryPipeline/Pipe.h>
#include <Poco/Logger.h>

namespace DB
{

void registerStorageArrowStream(StorageFactory & factory);

class StorageArrowStream : public IStorage, public WithContext
{
public:
    StorageArrowStream(
        const StorageID & storage_id_,
        const CHDB::ArrowStreamRegistry::ArrowStreamInfo & stream_info_,
        const ColumnsDescription & columns_,
        ContextPtr context_);

    ~StorageArrowStream() override = default;

    std::string getName() const override { return "ArrowStream"; }

    Pipe read(
        const Names & column_names,
        const StorageSnapshotPtr & storage_snapshot,
        SelectQueryInfo & query_info,
        ContextPtr context,
        QueryProcessingStage::Enum processed_stage,
        size_t max_block_size,
        size_t num_streams) override;

    Block prepareSampleBlock(const Names & column_names, const StorageSnapshotPtr & storage_snapshot);

private:
    CHDB::ArrowStreamRegistry::ArrowStreamInfo stream_info;
    Poco::Logger * logger = &Poco::Logger::get("StorageArrowStream");
};

}
