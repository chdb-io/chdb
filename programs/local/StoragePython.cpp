#include "StoragePython.h"
#include "NumpyType.h"
#include "PandasDataFrame.h"
#include "PybindWrapper.h"
#include "PythonSource.h"
#include "PyArrowTable.h"
#include "PyArrowStreamFactory.h"
#include "PythonUtils.h"

#include <DataTypes/DataTypeDate32.h>
#include <DataTypes/DataTypeDateTime.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypeObject.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesDecimal.h>
#include <DataTypes/DataTypesNumber.h>
#include <Formats/FormatFactory.h>
#include <Storages/ColumnsDescription.h>
#include <Storages/VirtualColumnsDescription.h>
#include <base/types.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <re2/re2.h>
#include <Poco/Logger.h>
#include <Common/Exception.h>
#include <Common/logger_useful.h>
#if USE_JEMALLOC
#include <Common/memory.h>
#endif

using namespace CHDB;

namespace DB
{

namespace ErrorCodes
{
extern const int BAD_ARGUMENTS;
extern const int LOGICAL_ERROR;
extern const int BAD_TYPE_OF_FIELD;
extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
extern const int TYPE_MISMATCH;
}


StoragePython::StoragePython(
    const StorageID & table_id_,
    const ColumnsDescription & columns_,
    const ConstraintsDescription & constraints_,
    ContextPtr context_,
    bool is_pandas_df_,
    CHDB::DataSourceWrapperPtr data_source_wrapper_)
    : IStorage(table_id_), WithContext(context_)
    , is_pandas_df(is_pandas_df_), data_source_wrapper(std::move(data_source_wrapper_))
{
    StorageInMemoryMetadata storage_metadata;
    storage_metadata.setColumns(columns_);
    storage_metadata.setConstraints(constraints_);
    setInMemoryMetadata(storage_metadata);

    if (is_pandas_df)
    {
        VirtualColumnsDescription virtuals;
        virtuals.addEphemeral("_row_id", std::make_shared<DataTypeUInt64>(), "Row index in the Pandas DataFrame");
        setVirtuals(std::move(virtuals));
    }
}

Pipe StoragePython::read(
    const Names & column_names,
    const StorageSnapshotPtr & storage_snapshot,
    SelectQueryInfo & /*query_info*/,
    ContextPtr /*context_*/,
    QueryProcessingStage::Enum /*processed_stage*/,
    size_t max_block_size,
    size_t num_streams)
{
    storage_snapshot->check(column_names);

    std::vector<bool> is_virtual_column(column_names.size(), false);
    for (size_t i = 0; i < column_names.size(); ++i)
    {
        if (isVirtualColumn(column_names[i], storage_snapshot->metadata))
            is_virtual_column[i] = true;
    }

    Block sample_block = prepareSampleBlock(column_names, storage_snapshot, is_virtual_column);
    auto format_settings = getFormatSettings(getContext());

    auto & data_source = data_source_wrapper->getDataSource();
    if (isInheritsFromPyReader(data_source))
    {
        return Pipe(
            std::make_shared<PythonSource>(data_source_wrapper, true, false, sample_block, column_cache, data_source_row_count, max_block_size, 0, 1, format_settings));
    }

    ArrowTableReaderPtr arrow_table_reader;
    {
        py::gil_scoped_acquire acquire;
        if (PyArrowTable::isPyArrowTable(data_source))
        {
            auto arrow_stream = PyArrowStreamFactory::createFromPyObject(data_source, sample_block.getNames());
            arrow_table_reader = std::make_shared<ArrowTableReader>(
                std::move(arrow_stream), sample_block,
                format_settings, num_streams, max_block_size);
        }
    }

    if (!arrow_table_reader)
        prepareColumnCache(column_names, sample_block, this->is_pandas_df, is_virtual_column);

    Pipes pipes;
    for (size_t stream = 0; stream < num_streams; ++stream)
        pipes.emplace_back(std::make_shared<PythonSource>(
            data_source_wrapper, false, this->is_pandas_df, sample_block, column_cache, data_source_row_count, max_block_size, stream, num_streams, format_settings, arrow_table_reader));
    return Pipe::unitePipes(std::move(pipes));
}

Block StoragePython::prepareSampleBlock(
    const Names & column_names,
    const StorageSnapshotPtr & storage_snapshot,
    const std::vector<bool> & is_virtual_column)
{
    Block sample_block;
    for (size_t i = 0; i < column_names.size(); ++i)
    {
        if (is_virtual_column[i])
        {
            auto virtual_column = storage_snapshot->virtual_columns->get(column_names[i]);
            sample_block.insert({virtual_column.type, virtual_column.name});
        }
        else
        {
            auto column_data = storage_snapshot->metadata->getColumns().getPhysical(column_names[i]);
            sample_block.insert({column_data.type, column_data.name});
        }
    }
    return sample_block;
}

void StoragePython::prepareColumnCache(
    const Names & names,
    const Block & sample_block,
    const bool is_pandas_df,
    const std::vector<bool> & is_virtual_column)
{
    py::gil_scoped_acquire acquire;
#if USE_JEMALLOC
    ::Memory::MemoryCheckScope memory_check_scope;
#endif

    auto & data_source = data_source_wrapper->getDataSource();

    bool need_rebuild = (column_cache == nullptr) || (column_cache->size() != names.size());
    if (!need_rebuild)
    {
        for (size_t i = 0; i < names.size(); ++i)
        {
            if ((*column_cache)[i].name != names[i])
            {
                need_rebuild = true;
                break;
            }
        }
    }

    if (need_rebuild)
    {
        column_cache = std::make_shared<PyColumnVec>(names.size());
        for (size_t i = 0; i < names.size(); ++i)
        {
            auto & col = (*column_cache)[i];
            if (is_virtual_column[i])
            {
                col.is_virtual = true;
                continue;
            }

            const auto & col_name = names[i];
            col.name = col_name;
            try
            {
                if (is_pandas_df)
                {
                    CHDB::PandasDataFrame::fillColumn(data_source, col_name, col, *data_source_wrapper);
                }
                else
                {
                    py::object col_data = data_source[py::str(col_name)];
                    col.buf = const_cast<void *>(tryGetPyArray(col_data, col.data, col.tmp, col.py_type, col.row_count));
                }
                if (col.buf == nullptr)
                    throw Exception(
                        ErrorCodes::PY_EXCEPTION_OCCURED, "Convert to array failed for column {} type {}", col_name, col.py_type);
                col.dest_type = sample_block.getByPosition(i).type;
                data_source_row_count = col.row_count;
            }
            catch (const Exception & e)
            {
                LOG_ERROR(logger, "Error processing column {}: {}", col_name, e.what());
                throw;
            }
        }

        if (data_source_row_count == 0 && is_pandas_df)
            data_source_row_count = py::len(data_source);
    }
}

ColumnsDescription StoragePython::getTableStructureFromData(std::vector<std::pair<std::string, std::string>> & schema, const ContextPtr & context)
{
    py::gil_assert();

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
    RE2 pattern_bool(R"(\bBool|bool)");
    RE2 pattern_float(R"(\b(float|double)(\d+)?)");
    RE2 pattern_decimal128(R"(decimal128\((\d+),\s*(\d+)\))");
    RE2 pattern_decimal256(R"(decimal256\((\d+),\s*(\d+)\))");
    RE2 pattern_date32(R"(\bdate32\b)");
    RE2 pattern_datatime64s(R"(\bdatetime64\[s\]|timestamp\[s\])");
    RE2 pattern_date64(R"(\bdate64\b|datetime64\[ms\]|timestamp\[ms\])");
    RE2 pattern_time32(R"(\btime32\b)");
    RE2 pattern_time64_us(R"(\btime64\[us\]\b|datetime64\[us\]|<M8\[us\])");
    RE2 pattern_time64_ns(R"(\btime64\[ns\]\b|datetime64\[ns\]|<M8\[ns\])");
    RE2 pattern_string_binary(
        R"(\bstring\b|<class 'str'>|str|DataType\(string\)|DataType\(binary\)|binary\[pyarrow\]|dtype\[object_\]|
dtype\('S|dtype\('O|<class 'bytes'>|<class 'bytearray'>|<class 'memoryview'>|<class 'numpy.bytes_'>|<class 'numpy.str_'>|<class 'numpy.void)");
    RE2 pattern_null(R"(\bnull\b)");
    RE2 pattern_json(R"((?i)(\bjson\b|struct<))");

    // Iterate through each pair of name and type string in the schema
    for (const auto & [name, typeStr] : schema)
    {
        std::shared_ptr<IDataType> data_type;

        std::string type_capture, bits, precision, scale;
        if (context->getQueryContext() && context->getQueryContext()->isJSONSupported() && RE2::PartialMatch(typeStr, pattern_json))
        {
            auto nested_type = std::make_shared<DataTypeObject>(DataTypeObject::SchemaFormat::JSON);
            data_type = std::make_shared<DataTypeNullable>(std::move(nested_type));
        }
        else if (RE2::PartialMatch(typeStr, pattern_int, &bits))
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
        else if (RE2::PartialMatch(typeStr, pattern_bool))
        {
            data_type = std::make_shared<DataTypeUInt8>();
        }
        else if (RE2::PartialMatch(typeStr, pattern_generic_int))
        {
            data_type = std::make_shared<DataTypeInt64>(); // Default to 64-bit integers for generic 'int'
        }
        else if (RE2::PartialMatch(typeStr, pattern_float, &type_capture, &bits))
        {
            if (bits == "32")
                data_type = std::make_shared<DataTypeFloat32>();
            else if (bits == "64")
                data_type = std::make_shared<DataTypeFloat64>();
            else if (bits.empty())
                data_type = std::make_shared<DataTypeFloat64>(); // Default to 64-bit floating point numbers
            else
                throw Exception(ErrorCodes::TYPE_MISMATCH, "Unrecognized floating point type: {}", typeStr);
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
        else if (RE2::PartialMatch(typeStr, pattern_datatime64s))
        {
            data_type = std::make_shared<DataTypeDateTime64>(0); // datetime64[s] corresponds to DateTime64(0)
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
        else if (RE2::PartialMatch(typeStr, pattern_null))
        {
            // ClickHouse uses a separate file with NULL masks in addition to normal file with values.
            // Entries in masks file allow ClickHouse to distinguish between NULL and a default value of
            // corresponding data type for each table row. Because of an additional file we can't make it
            // in Python, so we have to use String type for NULLs.
            // https://clickhouse.com/docs/en/sql-reference/data-types/nullable#storage-features
            data_type = std::make_shared<DataTypeString>();
        }
        else
        {
            throw Exception(ErrorCodes::TYPE_MISMATCH, "Unrecognized data type: {} on column {}", typeStr, name);
        }

        names_and_types.push_back({name, data_type});
    }

    return ColumnsDescription(names_and_types);
}

std::vector<std::pair<std::string, std::string>> PyReader::getSchemaFromPyObj(const py::object data)
{
    std::vector<std::pair<std::string, std::string>> schema;
    if (!py::hasattr(data, "__class__"))
    {
        throw Exception(
            ErrorCodes::UNKNOWN_FORMAT,
            "Unknown data type for schema inference. Consider inheriting PyReader and overriding get_schema().");
    }

    auto type_name = data.attr("__class__").attr("__name__").cast<std::string>();

    if (py::isinstance<py::dict>(data))
    {
        // If the data is a Python dictionary
        for (auto item : data.cast<py::dict>())
        {
            std::string key = py::str(item.first).cast<std::string>();
            py::list values = py::cast<py::list>(item.second);
            std::string dtype = py::str(values[0].attr("__class__").attr("__name__")).cast<std::string>();
            if (!values.empty())
                schema.emplace_back(key, dtype);
        }
        return schema;
    }

    if (py::hasattr(data, "dtypes"))
    {
        // If the data is a Pandas DataFrame
        py::object dtypes = data.attr("dtypes");
        py::list columns = data.attr("columns");
        for (size_t i = 0; i < py::len(columns); ++i)
        {
            std::string name = py::str(columns[i]).cast<std::string>();
            std::string dtype = py::str(py::repr(dtypes[columns[i]])).cast<std::string>();
            schema.emplace_back(name, dtype);
        }
        return schema;
    }

    if (py::hasattr(data, "schema"))
    {
        // If the data is a Pyarrow Table
        py::object tbl_schema = data.attr("schema");
        auto names = tbl_schema.attr("names").cast<py::list>();
        auto types = tbl_schema.attr("types").cast<py::list>();
        for (size_t i = 0; i < py::len(names); ++i)
        {
            std::string name = py::str(names[i]).cast<std::string>();
            std::string dtype = py::str(types[i]).cast<std::string>();
            schema.emplace_back(name, dtype);
        }
        return schema;
    }

    /// TODO: current implementation maybe cause dictionary update sequence error
    if (type_name == "recarray")
    {
        // if it's numpy.recarray
        py::object dtype = data.attr("dtype");
        py::list fields = dtype.attr("fields");
        py::dict fields_dict = fields.cast<py::dict>();
        // fields_dict looks like:
        //   {'TIME': (dtype('int64'), 0),
        //    'FX' : (dtype('int64'), 8),
        //    'FY' : (dtype('int64'), 16),
        //    'FZ' : (dtype('S68'), 24)}
        for (auto field : fields_dict)
        {
            std::string name = field.first.cast<std::string>();
            std::string dtype_str = py::str(field.second).cast<std::string>();
            schema.emplace_back(name, dtype_str);
        }
        return schema;
    }

    throw Exception(
        ErrorCodes::UNKNOWN_FORMAT,
        "Unknown data type {} for schema inference. Consider inheriting PyReader and overriding get_schema().",
        py::str(data.attr("__class__")).cast<std::string>());
}

}
