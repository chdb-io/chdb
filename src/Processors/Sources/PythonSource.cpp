#include "pybind11/numpy.h"
#define PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF

#include <cstddef>
#include <memory>
#include <vector>
#include <Columns/ColumnDecimal.h>
// #include <Columns/ColumnPyObject.h>
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
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <Poco/Logger.h>
#include <Common/COW.h>
#include <Common/Exception.h>
#include <Common/PythonUtils.h>
#include <Common/logger_useful.h>


namespace DB
{

namespace py = pybind11;

namespace ErrorCodes
{
extern const int PY_OBJECT_NOT_FOUND;
extern const int PY_EXCEPTION_OCCURED;
}

PythonSource::PythonSource(
    py::object & data_source_,
    const Block & sample_block_,
    const UInt64 max_block_size_,
    const size_t stream_index,
    const size_t num_streams)
    : ISource(sample_block_.cloneEmpty())
    , data_source(data_source_)
    , max_block_size(max_block_size_)
    , stream_index(stream_index)
    , num_streams(num_streams)
    , cursor(0)
{
    description.init(sample_block_);
}

template <typename T>
void insert_from_list(const py::list & obj, const MutableColumnPtr & column)
{
    py::gil_scoped_acquire acquire;
    for (auto && item : obj)
        column->insert(item.cast<T>());
}

void insert_string_from_array(const py::handle obj, const MutableColumnPtr & column)
{
    auto array = castToPyHandleVector(obj);
    for (auto && item : array)
    {
        size_t str_len;
        const char * ptr = GetPyUtf8StrData(item, str_len);
        column->insertData(ptr, str_len);
    }
}

template <typename T>
void insert_from_ptr(const void * ptr, const MutableColumnPtr & column, const size_t row_count)
{
    column->reserve(row_count);
    // get the raw data from the array and memcpy it into the column
    ColumnVectorHelper * helper = static_cast<ColumnVectorHelper *>(column.get());
    helper->appendRawData<sizeof(T)>(static_cast<const char *>(ptr), row_count);
}


template <typename T>
ColumnPtr convert_and_insert(const py::object & obj, UInt32 scale = 0)
{
    MutableColumnPtr column;
    if constexpr (std::is_same_v<T, DateTime64> || std::is_same_v<T, Decimal128> || std::is_same_v<T, Decimal256>)
        column = ColumnDecimal<T>::create(0, scale);
    else if constexpr (std::is_same_v<T, String>)
        column = ColumnString::create();
    else
        column = ColumnVector<T>::create();

    std::string type_name;
    size_t row_count;
    py::handle py_array;
    const void * data = tryGetPyArray(obj, py_array, type_name, row_count);
    if (!py_array.is_none())
    {
        if constexpr (std::is_same_v<T, String>)
            insert_string_from_array(py_array, column);
        else
            insert_from_ptr<T>(data, column, row_count);
        return column;
    }

    if (type_name == "list")
    {
        //reserve the size of the column
        column->reserve(row_count);
        insert_from_list<T>(obj, column);
        return column;
    }

    throw Exception(ErrorCodes::BAD_TYPE_OF_FIELD, "Unsupported type {} for value {}", getPyType(obj), castToStr(obj));
}

void PythonSource::destory(std::shared_ptr<std::vector<py::object>> & data)
{
    // manually destory std::shared_ptr<std::vector<py::object>> and trigger the py::object dec_ref with GIL holded
    py::gil_scoped_acquire acquire;
    data->clear();
    data.reset();
}

Chunk PythonSource::generate()
{
    size_t num_rows = 0;
    auto names = description.sample_block.getNames();
    if (names.empty())
        return {};

    std::shared_ptr<std::vector<py::object>> data;
    if (isInheritsFromPyReader(data_source))
    {
        py::gil_scoped_acquire acquire;
        data = std::move(castToSharedPtrVector<py::object>(data_source.attr("read")(names, max_block_size)));
    }
    else
    {
        auto total_rows = getLengthOfValueByKey(data_source, names.front());
        auto rows_per_stream = total_rows / num_streams;
        auto start = stream_index * rows_per_stream;
        auto end = (stream_index + 1) * rows_per_stream;
        if (stream_index == num_streams - 1)
            end = total_rows;
        if (cursor == 0)
            cursor = start;
        auto count = std::min(max_block_size, end - cursor);
        if (count == 0)
            return {};
        LOG_DEBUG(logger, "Stream index {} Reading {} rows from {}", stream_index, count, cursor);
        data = PyReader::readData(data_source, names, cursor, count);
    }

    if (data->empty())
        return {};

    // // if log level is debug, print all the data
    // if (logger->debug())
    // {
    //     // print all the data
    //     for (auto && col : data)
    //     {
    //         if (isInstanceOf<py::list>(col))
    //         {
    //             py::list list = col.cast<py::list>();
    //             for (auto && i : list)
    //                 LOG_DEBUG(logger, "Data: {}", py::str(i).cast<std::string>());
    //         }
    //         else if (isInstanceOf<py::array>(col))
    //         {
    //             py::array array = col.cast<py::array>();
    //             for (auto && i : array)
    //                 LOG_DEBUG(logger, "Data: {}", py::str(i).cast<std::string>());
    //         }
    //         else
    //         {
    //             LOG_DEBUG(logger, "Data: {}", py::str(col).cast<std::string>());
    //         }
    //     }
    // }

    Columns columns(description.sample_block.columns());
    for (size_t i = 0; i < data->size(); ++i)
    {
        if (i == 0)
            num_rows = getObjectLength((*data)[i]);
        const auto & column = (*data)[i];
        const auto & type = description.sample_block.getByPosition(i).type;
        WhichDataType which(type);

        try
        {
            // Dispatch to the appropriate conversion function based on data type
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
        catch (const Exception & e)
        {
            destory(data);
            LOG_ERROR(logger, "Error processing column {}: {}", i, e.what());
            throw;
        }
    }

    destory(data);

    if (num_rows == 0)
        return {};

    return Chunk(std::move(columns), num_rows);
}
}
