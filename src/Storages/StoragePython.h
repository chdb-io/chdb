#pragma once

#include <memory>
#include <Storages/IStorage.h>
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace DB
{

class StoragePython final : public IStorage
{
public:
    StoragePython(const StorageID & table_id_, const String & python_class_name_, const ColumnsDescription & columns_, ContextPtr context_);

    std::string getName() const override { return "Python"; }

    // Override the read method in IStorage
    Pipe read(
        const Names & column_names,
        const StorageSnapshotPtr & storage_snapshot,
        SelectQueryInfo & query_info,
        ContextPtr context,
        QueryProcessingStage::Enum processed_stage,
        size_t max_block_size,
        size_t num_streams) override;

    // Override the write method in IStorage
    SinkToStoragePtr
    write(const ASTPtr & query, const StorageMetadataPtr & metadata_snapshot, ContextPtr context, bool async_insert) override;

private:
    String python_class_name;
    py::object python_class_instance; // To store the instance of the Python class
};

}
