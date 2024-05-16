#include <Columns/ColumnDecimal.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnVectorHelper.h>
#include <Columns/IColumn.h>
#include <DataTypes/DataTypeDecimalBase.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesDecimal.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/IDataType.h>
#include <Interpreters/ExpressionActions.h>
#include <Processors/Sources/PythonSource.h>
#include <Storages/StoragePython.h>
#include <base/Decimal.h>
#include <base/Decimal_fwd.h>
#include <base/types.h>
#include <pybind11/gil.h>
#include <pybind11/pytypes.h>
#include <Poco/Logger.h>
#include <Common/Exception.h>
#include <Common/logger_useful.h>

namespace DB
{
PythonSource::PythonSource(py::object reader_, const Block & sample_block_, const UInt64 max_block_size_)
    : ISource(sample_block_.cloneEmpty()), reader(reader_), max_block_size(max_block_size_)
{
    description.init(sample_block_);
}

template <typename T>
void insert_from_pyobject(py::object obj, const MutableColumnPtr & column)
{
    if (py::isinstance<py::list>(obj))
    {
        py::list list = obj.cast<py::list>();
        for (auto && item : list)
            column->insert(item.cast<T>());
        return;
    }

    if (py::isinstance<py::array>(obj))
    {
        // if column type is ColumnString, we need to handle it like list
        if constexpr (std::is_same_v<T, String>)
        {
            py::array array = obj.cast<py::array>();
            for (auto && item : array)
                column->insert(item.cast<std::string>());
            return;
        }
        py::array array = obj.cast<py::array>();
        column->reserve(size_t(array.size()));
        // get the raw data from the array and memcpy it into the column
        ColumnVectorHelper * helper = static_cast<ColumnVectorHelper *>(column.get());
        helper->appendRawData<sizeof(T)>(static_cast<const char *>(array.data()), array.size());
        LOG_DEBUG(&Poco::Logger::get("TableFunctionPython"), "Read {} bytes", array.size() * array.itemsize());
        return;
    }
    else
    {
        throw Exception(
            ErrorCodes::BAD_TYPE_OF_FIELD,
            "Unsupported type {} for value {}",
            obj.get_type().attr("__name__").cast<std::string>(),
            py::str(obj).cast<std::string>());
    }
}

template <typename T>
ColumnPtr convert_and_insert(py::object obj, UInt32 scale = 0)
{
    MutableColumnPtr column;
    if constexpr (std::is_same_v<T, DateTime64> || std::is_same_v<T, Decimal128> || std::is_same_v<T, Decimal256>)
        column = ColumnDecimal<T>::create(0, scale);
    else if constexpr (std::is_same_v<T, String>)
        column = ColumnString::create();
    else
        column = ColumnVector<T>::create();

    if (py::isinstance<py::list>(obj) || py::isinstance<py::array>(obj))
    {
        insert_from_pyobject<T>(obj, column);
        return column;
    }

    std::string type_name = obj.attr("__class__").attr("__name__").cast<std::string>();
    if (type_name == "Series")
    {
        py::object values = obj.attr("values");
        if (py::isinstance<py::array>(values))
        {
            insert_from_pyobject<T>(values, column);
            return column;
        }

        // Handle ArrowExtensionArray and similar structures with to_numpy method
        // this will introduce about 25% overhead for the case when the data is already in numpy array:
        // See:
        //      Read parquet file into memory. Time cost: 0.6645004749298096 s
        //      Parquet file size: 1395695970 bytes
        //      Read parquet file as old pandas dataframe. Time cost: 9.46176028251648 s
        //      Dataframe size: 4700000128 bytes
        //      Read parquet file as pandas dataframe(arrow). Time cost: 1.6119577884674072 s
        //      Dataframe size: 7025418615 bytes
        //      Convert old dataframe to numpy array. Time cost: 2.574920654296875e-05 s
        //      Convert dataframe(arrow) to numpy array. Time cost: 0.017014503479003906 s
        //      Run duckdb on dataframe. Time cost: 0.10320639610290527 s
        //      Run with new chDB on dataframe. Time cost: 0.09642386436462402 s
        //      Run with new chDB on dataframe(arrow). Time cost: 0.11595273017883301 s
        // chdb todo: maybe we can use the ArrowExtensionArray directly
        if (py::hasattr(values, "to_numpy"))
        {
            py::array numpy_array = values.attr("to_numpy")();
            insert_from_pyobject<T>(numpy_array, column);
            return column;
        }
    }

    throw Exception(
        ErrorCodes::BAD_TYPE_OF_FIELD,
        "Unsupported type {} for value {}",
        obj.get_type().attr("__name__").cast<std::string>(),
        py::str(obj).cast<std::string>());
}

Chunk PythonSource::generate()
{
    size_t num_rows = 0;
    py::gil_scoped_acquire acquire;
    try
    {
        auto names = description.sample_block.getNames();
        auto data = reader.attr("read")(names, max_block_size).cast<std::vector<py::object>>();

        LOG_DEBUG(logger, "Read {} columns", data.size());
        LOG_DEBUG(logger, "Need {} columns", description.sample_block.columns());
        LOG_DEBUG(logger, "Max block size: {}", max_block_size);

        // if log level is debug, print all the data
        if (logger->debug())
        {
            // print all the data
            for (auto && col : data)
            {
                if (py::isinstance<py::list>(col))
                {
                    py::list list = col.cast<py::list>();
                    for (auto && i : list)
                        LOG_DEBUG(logger, "Data: {}", py::str(i).cast<std::string>());
                }
                else if (py::isinstance<py::array>(col))
                {
                    py::array array = col.cast<py::array>();
                    for (auto && i : array)
                        LOG_DEBUG(logger, "Data: {}", py::str(i).cast<std::string>());
                }
                else
                {
                    LOG_DEBUG(logger, "Data: {}", py::str(col).cast<std::string>());
                }
            }
        }

        Columns columns(description.sample_block.columns());
        // fill in the columns
        for (size_t i = 0; i < data.size(); ++i)
        {
            if (i == 0)
                num_rows = py::len(data[i]);
            const auto & column = data[i];
            const auto & type = description.sample_block.getByPosition(i).type;
            WhichDataType which(type);

            if (which.isUInt8())
                columns[i] = convert_and_insert<UInt8>(column);
            else if (which.isUInt16())
                columns[i] = convert_and_insert<UInt16>(column);
            else if (which.isUInt32())
                columns[i] = convert_and_insert<UInt32>(column);
            else if (which.isUInt64())
                columns[i] = convert_and_insert<UInt64>(column);
            else if (which.isUInt128())
                columns[i] = convert_and_insert<UInt128>(column);
            else if (which.isUInt256())
                columns[i] = convert_and_insert<UInt256>(column);
            else if (which.isInt8())
                columns[i] = convert_and_insert<Int8>(column);
            else if (which.isInt16())
                columns[i] = convert_and_insert<Int16>(column);
            else if (which.isInt32())
                columns[i] = convert_and_insert<Int32>(column);
            else if (which.isInt64())
                columns[i] = convert_and_insert<Int64>(column);
            else if (which.isInt128())
                columns[i] = convert_and_insert<Int128>(column);
            else if (which.isInt256())
                columns[i] = convert_and_insert<Int256>(column);
            else if (which.isFloat32())
                columns[i] = convert_and_insert<Float32>(column);
            else if (which.isFloat64())
                columns[i] = convert_and_insert<Float64>(column);
            else if (which.isDecimal128())
            {
                const auto & dtype = typeid_cast<const DataTypeDecimal<Decimal128> *>(type.get());
                columns[i] = convert_and_insert<Decimal128>(column, dtype->getScale());
            }
            else if (which.isDecimal256())
            {
                const auto & dtype = typeid_cast<const DataTypeDecimal<Decimal256> *>(type.get());
                columns[i] = convert_and_insert<Decimal256>(column, dtype->getScale());
            }
            else if (which.isDateTime())
                columns[i] = convert_and_insert<UInt32>(column);
            else if (which.isDateTime64())
                columns[i] = convert_and_insert<DateTime64>(column);
            else if (which.isString())
                columns[i] = convert_and_insert<String>(column);
            else
                throw Exception(
                    ErrorCodes::BAD_TYPE_OF_FIELD,
                    "Unsupported type {} for column {}",
                    type->getName(),
                    description.sample_block.getByPosition(i).name);
        }

        if (num_rows == 0)
            return {};

        return Chunk(std::move(columns), num_rows);
    }
    catch (const std::exception & e)
    {
        // py::gil_scoped_release release;
        throw Exception(ErrorCodes::LOGICAL_ERROR, e.what());
    }
}


}
