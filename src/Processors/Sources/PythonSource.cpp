#include <Columns/ColumnDecimal.h>
#include <Columns/ColumnString.h>
#include <Columns/IColumn.h>
#include <DataTypes/DataTypeDecimalBase.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesDecimal.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/IDataType.h>
#include <Processors/Sources/PythonSource.h>
#include <Storages/StoragePython.h>
#include <base/Decimal.h>
#include <pybind11/gil.h>
#include <pybind11/pytypes.h>
#include <Common/Exception.h>
#include <Common/logger_useful.h>
#include <base/Decimal_fwd.h>
#include <base/types.h>

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
        // list.dec_ref();
    }
    else if (py::isinstance<py::array>(obj))
    {
        py::array array = obj.cast<py::array>();
        for (auto && item : array)
            column->insert(item.cast<T>());
        // array.dec_ref();
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

        if (values.attr("__class__").attr("__name__").cast<std::string>() == "ArrowExtensionArray")
        {
            py::object array = values.attr("to_pandas")();
            py::array array_values = array.cast<py::array>();
            insert_from_pyobject<T>(array_values, column);
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
        // // Set data vector to empty to avoid trigger py::object destructor without GIL
        // // Note: we have already manually decremented the reference count of the list or array in `convert_and_insert` function
        // for (auto && col : data)
        // {
        //     col.dec_ref();
        //     col.release();
        // }
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
