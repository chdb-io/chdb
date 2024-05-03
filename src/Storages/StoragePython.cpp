#include <Columns/IColumn.h>
#include <Functions/FunctionsConversion.h>
#include <Interpreters/evaluateConstantExpression.h>
#include <Processors/Sources/PythonSource.h>
#include <Storages/IStorage.h>
#include <Storages/StorageFactory.h>
#include <Storages/StoragePython.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Common/Exception.h>

#include <any>

namespace DB
{

namespace ErrorCodes
{
extern const int BAD_ARGUMENTS;
extern const int LOGICAL_ERROR;
extern const int BAD_TYPE_OF_FIELD;
}

namespace py = pybind11;


StoragePython::StoragePython(
    const StorageID & table_id_,
    const ColumnsDescription & columns_,
    const ConstraintsDescription & constraints_,
    std::shared_ptr<PyReader> reader_,
    ContextPtr context_)
    : IStorage(table_id_), reader(std::move(reader_)), WithContext(context_->getGlobalContext())
{
    StorageInMemoryMetadata storage_metadata;
    storage_metadata.setColumns(columns_);
    storage_metadata.setConstraints(constraints_);
    setInMemoryMetadata(storage_metadata);
}

Pipe StoragePython::read(
    const Names & column_names,
    const StorageSnapshotPtr & storage_snapshot,
    SelectQueryInfo & /*query_info*/,
    ContextPtr /*context_*/,
    QueryProcessingStage::Enum /*processed_stage*/,
    size_t max_block_size,
    size_t /*num_streams*/)
{
    storage_snapshot->check(column_names);

    Block sample_block = prepareSampleBlock(column_names, storage_snapshot);

    return Pipe(std::make_shared<PythonSource>(reader, sample_block, max_block_size));
}

Block StoragePython::prepareSampleBlock(const Names & column_names, const StorageSnapshotPtr & storage_snapshot)
{
    Block sample_block;
    for (const String & column_name : column_names)
    {
        auto column_data = storage_snapshot->metadata->getColumns().getPhysical(column_name);
        sample_block.insert({column_data.type, column_data.name});
    }
    return sample_block;
}

void registerStoragePython(StorageFactory & factory)
{
    factory.registerStorage(
        "Python",
        [](const StorageFactory::Arguments & args) -> StoragePtr
        {
            if (args.engine_args.size() != 1)
                throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH, "Python engine requires 1 argument: PyReader object");

            auto reader = std::dynamic_pointer_cast<PyReader>(std::any_cast<std::shared_ptr<PyReader>>(args.engine_args[0]));
            return std::make_shared<StoragePython>(args.table_id, args.columns, args.constraints, reader, args.getLocalContext());
        },
        {.supports_settings = true, .supports_parallel_insert = false});
}
}
