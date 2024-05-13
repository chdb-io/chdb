#include <Columns/IColumn.h>
#include <DataTypes/DataTypeDate.h>
#include <DataTypes/DataTypeDate32.h>
#include <DataTypes/DataTypeDateTime.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <Functions/FunctionsConversion.h>
#include <Interpreters/evaluateConstantExpression.h>
#include <Processors/Sources/PythonSource.h>
#include <Storages/ColumnsDescription.h>
#include <Storages/IStorage.h>
#include <Storages/StorageFactory.h>
#include <Storages/StoragePython.h>
#include <base/types.h>
#include <pybind11/functional.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <re2/re2.h>
#include <Poco/Logger.h>
#include <Common/Exception.h>
#include <Common/logger_useful.h>

#include <any>

namespace DB
{

namespace ErrorCodes
{
extern const int BAD_ARGUMENTS;
extern const int LOGICAL_ERROR;
extern const int BAD_TYPE_OF_FIELD;
}


StoragePython::StoragePython(
    const StorageID & table_id_,
    const ColumnsDescription & columns_,
    const ConstraintsDescription & constraints_,
    py::object reader_,
    ContextPtr context_)
    : IStorage(table_id_), reader(reader_), WithContext(context_->getGlobalContext())
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
    py::gil_scoped_acquire acquire;
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

ColumnsDescription StoragePython::getTableStructureFromData(py::object reader)
{
    if (!reader)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Python reader not initialized");
    py::gil_scoped_acquire acquire;
    auto schema = reader.attr("get_schema")().cast<std::vector<std::pair<std::string, std::string>>>();

    auto * logger = &Poco::Logger::get("StoragePython");
    if (logger->debug())
    {
        LOG_DEBUG(logger, "Schema content:");
        for (const auto & item : schema)
            LOG_DEBUG(logger, "Column: {}, Type: {}", String(item.first), String(item.second));
    }

    NamesAndTypesList names_and_types;

    // Define regular expressions for different data types
    RE2 pattern_int(R"(\bint(\d+))");
    RE2 pattern_generic_int(R"(\bint\b|<class 'int'>)"); // Matches generic 'int'
    RE2 pattern_uint(R"(\buint(\d+))");
    RE2 pattern_float(R"(\b(float|double)(\d+))");
    RE2 pattern_decimal128(R"(decimal128\((\d+),\s*(\d+)\))");
    RE2 pattern_decimal256(R"(decimal256\((\d+),\s*(\d+)\))");
    RE2 pattern_date32(R"(\bdate32\b)");
    RE2 pattern_date64(R"(\bdate64\b)");
    RE2 pattern_time32(R"(\btime32\b)");
    RE2 pattern_time64_us(R"(\btime64\[us\]\b)");
    RE2 pattern_time64_ns(R"(\btime64\[ns\]\b)");
    RE2 pattern_string_binary(
        R"(\bstring\b|<class 'str'>|str|DataType\(string\)|DataType\(binary\)|binary\[pyarrow\]|dtype\[object_\]|dtype\('O'\))");

    // Iterate through each pair of name and type string in the schema
    for (const auto & [name, typeStr] : schema)
    {
        std::shared_ptr<IDataType> data_type;

        std::string bits, precision, scale;
        if (RE2::PartialMatch(typeStr, pattern_int, &bits))
        {
            if (bits == "8")
                data_type = std::make_shared<DataTypeInt8>();
            else if (bits == "16")
                data_type = std::make_shared<DataTypeInt16>();
            else if (bits == "32")
                data_type = std::make_shared<DataTypeInt32>();
            else if (bits == "64")
                data_type = std::make_shared<DataTypeInt64>();
            else if (bits == "128")
                data_type = std::make_shared<DataTypeInt128>();
            else if (bits == "256")
                data_type = std::make_shared<DataTypeInt256>();
        }
        else if (RE2::PartialMatch(typeStr, pattern_uint, &bits))
        {
            if (bits == "8")
                data_type = std::make_shared<DataTypeUInt8>();
            else if (bits == "16")
                data_type = std::make_shared<DataTypeUInt16>();
            else if (bits == "32")
                data_type = std::make_shared<DataTypeUInt32>();
            else if (bits == "64")
                data_type = std::make_shared<DataTypeUInt64>();
            else if (bits == "128")
                data_type = std::make_shared<DataTypeUInt128>();
            else if (bits == "256")
                data_type = std::make_shared<DataTypeUInt256>();
        }
        else if (RE2::PartialMatch(typeStr, pattern_generic_int))
        {
            data_type = std::make_shared<DataTypeInt64>(); // Default to 64-bit integers for generic 'int'
        }
        else if (RE2::PartialMatch(typeStr, pattern_float, &bits))
        {
            if (bits == "32")
                data_type = std::make_shared<DataTypeFloat32>();
            else if (bits == "64")
                data_type = std::make_shared<DataTypeFloat64>();
        }
        else if (RE2::PartialMatch(typeStr, pattern_decimal128, &precision, &scale))
        {
            data_type = std::make_shared<DataTypeDecimal128>(std::stoi(precision), std::stoi(scale));
        }
        else if (RE2::PartialMatch(typeStr, pattern_decimal256, &precision, &scale))
        {
            data_type = std::make_shared<DataTypeDecimal256>(std::stoi(precision), std::stoi(scale));
        }
        else if (RE2::PartialMatch(typeStr, pattern_date32))
        {
            data_type = std::make_shared<DataTypeDate32>();
        }
        else if (RE2::PartialMatch(typeStr, pattern_date64))
        {
            data_type = std::make_shared<DataTypeDateTime64>(3); // date64 corresponds to DateTime64(3)
        }
        else if (RE2::PartialMatch(typeStr, pattern_time32))
        {
            data_type = std::make_shared<DataTypeDateTime>();
        }
        else if (RE2::PartialMatch(typeStr, pattern_time64_us))
        {
            data_type = std::make_shared<DataTypeDateTime64>(6); // time64[us] corresponds to DateTime64(6)
        }
        else if (RE2::PartialMatch(typeStr, pattern_time64_ns))
        {
            data_type = std::make_shared<DataTypeDateTime64>(9); // time64[ns] corresponds to DateTime64(9)
        }
        else if (RE2::PartialMatch(typeStr, pattern_string_binary))
        {
            data_type = std::make_shared<DataTypeString>();
        }
        else
        {
            throw Exception(ErrorCodes::TYPE_MISMATCH, "Unrecognized data type: {}", typeStr);
        }

        names_and_types.push_back({name, data_type});
    }

    return ColumnsDescription(names_and_types);
}

void registerStoragePython(StorageFactory & factory)
{
    factory.registerStorage(
        "Python",
        [](const StorageFactory::Arguments & args) -> StoragePtr
        {
            if (args.engine_args.size() != 1)
                throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH, "Python engine requires 1 argument: PyReader object");

            py::object reader = std::any_cast<py::object>(args.engine_args[0]);
            return std::make_shared<StoragePython>(args.table_id, args.columns, args.constraints, reader, args.getLocalContext());
        },
        {.supports_settings = true, .supports_parallel_insert = false});
}
}
