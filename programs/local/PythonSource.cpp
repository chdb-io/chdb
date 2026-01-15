#include "PythonSource.h"
#include "ColumnVectorHelper.h"
#include "ListScan.h"
#include "PandasScan.h"
#include "StoragePython.h"

#include <algorithm>
#include <exception>
#include <type_traits>
#include <boolobject.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/detail/non_limited_api.h>

#include <Columns/ColumnVector.h>
#include <Poco/Logger.h>
#include <Common/COW.h>
#include <Common/Exception.h>
#include <Common/iota.h>
#include <Common/logger_useful.h>
#include <Columns/ColumnDecimal.h>
#include <Columns/ColumnString.h>
#include <Columns/IColumn.h>
#include <DataTypes/DataTypeDecimalBase.h>
#include <DataTypes/DataTypeObject.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesDecimal.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeDateTime64.h>
#include <DataTypes/DataTypeNullable.h>
#include <Interpreters/ExpressionActions.h>
#include <base/Decimal.h>
#include <base/Decimal_fwd.h>
#include <base/scope_guard.h>
#include <base/types.h>
#include <Common/typeid_cast.h>
#if USE_JEMALLOC
#    include <Common/memory.h>
#endif

using namespace CHDB;

namespace DB
{

namespace py = pybind11;

namespace ErrorCodes
{
    extern const int BAD_TYPE_OF_FIELD;
    extern const int PY_OBJECT_NOT_FOUND;
    extern const int PY_EXCEPTION_OCCURED;
}

PythonSource::PythonSource(
    py::object & data_source_,
    bool isInheritsFromPyReader_,
    bool isPandasDataFrame_,
    const Block & sample_block_,
    PyColumnVecPtr column_cache,
    size_t data_source_row_count,
    size_t max_block_size_,
    size_t stream_index,
    size_t num_streams,
    const FormatSettings & format_settings_,
    ArrowTableReaderPtr arrow_table_reader_)
    : ISource(std::make_shared<Block>(sample_block_.cloneEmpty()))
    , data_source(data_source_)
    , isInheritsFromPyReader(isInheritsFromPyReader_)
    , isPandasDataFrame(isPandasDataFrame_)
    , sample_block(sample_block_)
    , column_cache(column_cache)
    , data_source_row_count(data_source_row_count)
    , max_block_size(max_block_size_)
    , stream_index(stream_index)
    , num_streams(num_streams)
    , cursor(0)
    , format_settings(format_settings_)
    , arrow_table_reader(arrow_table_reader_)
{
}

template <typename T>
void PythonSource::insert_from_list(const py::list & obj, const MutableColumnPtr & column)
{
    py::gil_scoped_acquire acquire;
#if USE_JEMALLOC
    ::Memory::MemoryCheckScope memory_check_scope;
#endif
    for (auto && item : obj)
    {
        if constexpr (std::is_same_v<T, UInt8>)
        {
            if (PyBool_Check(item.ptr()))
            {
                column->insert(static_cast<UInt8>(py::cast<bool>(item) ? 1 : 0));
            }
            else
            {
                column->insert(py::cast<UInt8>(item));
            }
        }
        else if (item.is_none())
        {
            column->insertDefault();
        }
        else
        {
            column->insert(item.cast<T>());
        }
    }
}

void PythonSource::insert_string_from_array(const py::handle obj, const MutableColumnPtr & column)
{
    auto array = castToPyHandleVector(obj);
    auto * string_column = static_cast<ColumnString *>(column.get());
    for (auto && item : array)
    {
        FillColumnString(item.ptr(), string_column);
    }
}

void PythonSource::convert_string_array_to_block(
    PyObject ** buf, const MutableColumnPtr & column, const size_t offset, const size_t row_count, size_t stride)
{
    ColumnString * string_column = typeid_cast<ColumnString *>(column.get());
    if (string_column == nullptr)
        throw Exception(ErrorCodes::BAD_TYPE_OF_FIELD, "Column is not a string column");
    ColumnString::Chars & data = string_column->getChars();
    ColumnString::Offsets & offsets = string_column->getOffsets();
    offsets.reserve(row_count);

    const size_t effective_stride = (stride == 0) ? sizeof(PyObject *) : stride;
    const auto * base_ptr = reinterpret_cast<const char *>(buf);

    for (size_t i = offset; i < offset + row_count; ++i)
    {
        auto * obj = *reinterpret_cast<PyObject * const *>(base_ptr + i * effective_stride);
        if (!PyUnicode_Check(obj))
        {
            insertObjToStringColumn(obj, string_column);
            continue;
        }
        FillColumnString(obj, string_column);
        // Try to help reserve memory for the string column data every 100 rows to avoid frequent reallocations
        // Check the avg size of the string column data and reserve memory accordingly
        if ((i - offset) % 10 == 9)
        {
            size_t data_size = data.size();
            size_t counter = i - offset + 1;
            size_t avg_size = data_size / counter;
            size_t reserve_size = avg_size * row_count;
            if (reserve_size > data.capacity())
            {
                LOG_DEBUG(logger, "Reserving memory for string column data from {} to {}, avg size: {}, count: {}",
                            data_size, reserve_size, avg_size, counter);
                data.reserve(reserve_size);
            }
        }
    }
}

template <typename T>
void PythonSource::insert_from_ptr(const void * ptr, const MutableColumnPtr & column, const size_t offset, const size_t row_count, size_t stride)
{
    column->reserve(row_count);

    if (stride == 0 || stride == sizeof(T))
    {
    ColumnVectorHelper * helper = static_cast<ColumnVectorHelper *>(column.get());
    const char * start = static_cast<const char *>(ptr) + offset * sizeof(T);
    helper->appendRawData<sizeof(T)>(start, row_count);
    }
    else
    {
        const auto * base_ptr = static_cast<const char *>(ptr);
        for (size_t i = offset; i < offset + row_count; ++i)
        {
            T value = *reinterpret_cast<const T *>(base_ptr + i * stride);
            column->insert(value);
        }
    }
}

template <typename T>
ColumnPtr PythonSource::convert_and_insert(const py::object & obj, UInt32 scale, bool is_json)
{
    MutableColumnPtr column;
    if (is_json)
    {
        auto nested_type = std::make_shared<DataTypeObject>(DataTypeObject::SchemaFormat::JSON);
        auto data_type = std::make_shared<DataTypeNullable>(std::move(nested_type));
        column = data_type->createColumn();
    }
    else if constexpr (std::is_same_v<T, DateTime64> || std::is_same_v<T, Decimal128> || std::is_same_v<T, Decimal256>)
        column = ColumnDecimal<T>::create(0, scale);
    else if constexpr (std::is_same_v<T, String>)
        column = ColumnString::create();
    else
        column = ColumnVector<T>::create();

    std::string type_name;
    size_t row_count = 0;
    py::handle py_array;
    py::handle tmp;
    SCOPE_EXIT({
        if (!tmp.is_none())
            tmp.dec_ref();
    });
    const void * data = tryGetPyArray(obj, py_array, tmp, type_name, row_count);
    if (type_name == "list")
    {
        if (is_json)
        {
            CHDB::ListScan::scanObject(0, row_count, format_settings, obj, column);
            return column;
        }

        //reserve the size of the column
        column->reserve(row_count);
        insert_from_list<T>(obj, column);
        return column;
    }

    if (!py_array.is_none() && data != nullptr)
    {
        if (is_json)
            CHDB::PandasScan::scanObject(0, row_count, format_settings, data, column);
        else if constexpr (std::is_same_v<T, String>)
            insert_string_from_array(py_array, column);
        else
            insert_from_ptr<T>(data, column, 0, row_count);
        return column;
    }

    throw Exception(ErrorCodes::BAD_TYPE_OF_FIELD, "Unsupported type {} for value {}", getPyType(obj), castToStr(obj));
}


template <typename T>
ColumnPtr PythonSource::convert_and_insert_array(const ColumnWrapper & col_wrap, size_t & cursor, const size_t count, UInt32 scale)
{
    MutableColumnPtr column;
    if constexpr (std::is_same_v<T, DateTime64> || std::is_same_v<T, Decimal128> || std::is_same_v<T, Decimal256>)
        column = ColumnDecimal<T>::create(0, scale);
    else if constexpr (std::is_same_v<T, String>)
        column = ColumnString::create();
    else
        column = ColumnVector<T>::create();

    if (col_wrap.data.is_none())
        throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED, "Column data is None");

    if (col_wrap.py_type == "list")
    {
        py::gil_scoped_acquire acquire;
        insert_from_list<T>(col_wrap.data.cast<py::list>().attr("__getitem__")(py::slice(cursor, cursor + count, 1)), column);
        return column;
    }
    if constexpr (std::is_same_v<T, String>)
        convert_string_array_to_block(static_cast<PyObject **>(col_wrap.buf), column, cursor, count, col_wrap.stride);
    else
        insert_from_ptr<T>(col_wrap.buf, column, cursor, count, col_wrap.stride);

    return column;
}

void PythonSource::destory(PyObjectVecPtr & data)
{
    // manually destory PyObjectVec and trigger the py::object dec_ref with GIL holded
    py::gil_scoped_acquire acquire;
    data->clear();
    data.reset();
}

Chunk PythonSource::genChunk(size_t & num_rows, PyObjectVecPtr data)
{
    Columns columns(sample_block.columns());
    for (size_t i = 0; i < data->size(); ++i)
    {
        if (i == 0)
            num_rows = getObjectLength((*data)[i]);
        const auto & column = (*data)[i];
        const auto & type = sample_block.getByPosition(i).type;
        auto data_type = removeNullable(type);
        WhichDataType which(data_type);

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
            {
                const auto & dtype = typeid_cast<const DataTypeDateTime64 *>(type.get());
                columns[i] = convert_and_insert<DateTime64>(column, dtype->getScale());
            }
            else if (which.isString())
                columns[i] = convert_and_insert<String>(column);
            else if (which.isObject())
                columns[i] = convert_and_insert<String>(column, 0, true);
            else
                throw Exception(
                    ErrorCodes::BAD_TYPE_OF_FIELD,
                    "Unsupported type {} for column {}",
                    type->getName(),
                    sample_block.getByPosition(i).name);
        }
        catch (Exception & e)
        {
            destory(data);
            LOG_ERROR(logger, "Error processing column \"{}\": {}", sample_block.getByPosition(i).name, e.what());
            throw Exception(
                ErrorCodes::PY_EXCEPTION_OCCURED,
                "Error processing column \"{}\": {}",
                sample_block.getByPosition(i).name,
                e.what());
        }
        catch (std::exception & e)
        {
            destory(data);
            LOG_ERROR(logger, "Error processing column \"{}\": {}", sample_block.getByPosition(i).name, e.what());
            throw Exception(
                ErrorCodes::PY_EXCEPTION_OCCURED,
                "Error processing column \"{}\": {}",
                sample_block.getByPosition(i).name,
                e.what());
        }
        catch (...)
        {
            destory(data);
            LOG_ERROR(logger, "Error processing column \"{}\": unknown exception", sample_block.getByPosition(i).name);
            throw Exception(
                ErrorCodes::PY_EXCEPTION_OCCURED,
                "Error processing column \"{}\": unknown exception",
                sample_block.getByPosition(i).name);
        }
    }

    destory(data);

    if (num_rows == 0)
        return {};

    return Chunk(std::move(columns), num_rows);
}

std::shared_ptr<PyObjectVec>
PythonSource::scanData(const py::object & data, const std::vector<std::string> & col_names, size_t & cursor, size_t count)
{
    py::gil_scoped_acquire acquire;
    auto block = std::make_shared<PyObjectVec>();
    // Access columns directly by name and slice
    for (const auto & col : col_names)
    {
        py::object col_data = data[py::str(col)]; // Use dictionary-style access
        block->push_back(col_data.attr("__getitem__")(py::slice(cursor, cursor + count, 1)));
    }

    if (!block->empty())
        cursor += py::len((*block)[0]); // Update cursor based on the length of the first column slice

    return std::move(block);
}



Chunk PythonSource::scanDataToChunk()
{
    auto names = sample_block.getNames();
    if (names.empty())
        return {};

    auto [offset, count] = calculateOffsetAndCount();
    if (count == 0)
        return {};
    LOG_DEBUG(logger, "Stream index {} Reading {} rows from {}", stream_index, count, cursor);

    Columns columns(sample_block.columns());

    for (size_t i = 0; i < sample_block.columns(); ++i)
    {
        const auto & col = (*column_cache)[i];
        if (col.is_virtual)
        {
            const auto & col_name = sample_block.getByPosition(i).name;
            chassert(col_name == "_row_id");
            auto row_id_column = ColumnVector<UInt64>::create(count);
            auto & row_id_data = row_id_column->getData();
            iota(row_id_data.data(), count, static_cast<UInt64>(cursor));
            columns[i] = std::move(row_id_column);
            continue;
        }

        const auto & type = sample_block.getByPosition(i).type;

        bool is_nullable = type->isNullable();
        auto data_type = removeNullable(type);
        WhichDataType which(data_type);

        try
        {
            if (isPandasDataFrame && is_nullable)
            {
                columns[i] = PandasScan::scanColumn(col, cursor, count, format_settings);
                continue;
            }

            // Dispatch to the appropriate conversion function based on data type
            if (which.isUInt8())
                columns[i] = convert_and_insert_array<UInt8>(col, cursor, count);
            else if (which.isUInt16())
                columns[i] = convert_and_insert_array<UInt16>(col, cursor, count);
            else if (which.isUInt32())
                columns[i] = convert_and_insert_array<UInt32>(col, cursor, count);
            else if (which.isUInt64())
                columns[i] = convert_and_insert_array<UInt64>(col, cursor, count);
            else if (which.isUInt128())
                columns[i] = convert_and_insert_array<UInt128>(col, cursor, count);
            else if (which.isUInt256())
                columns[i] = convert_and_insert_array<UInt256>(col, cursor, count);
            else if (which.isInt8())
                columns[i] = convert_and_insert_array<Int8>(col, cursor, count);
            else if (which.isInt16())
                columns[i] = convert_and_insert_array<Int16>(col, cursor, count);
            else if (which.isInt32())
                columns[i] = convert_and_insert_array<Int32>(col, cursor, count);
            else if (which.isInt64())
                columns[i] = convert_and_insert_array<Int64>(col, cursor, count);
            else if (which.isInt128())
                columns[i] = convert_and_insert_array<Int128>(col, cursor, count);
            else if (which.isInt256())
                columns[i] = convert_and_insert_array<Int256>(col, cursor, count);
            else if (which.isFloat32())
                columns[i] = convert_and_insert_array<Float32>(col, cursor, count);
            else if (which.isFloat64())
                columns[i] = convert_and_insert_array<Float64>(col, cursor, count);
            else if (which.isDecimal128())
            {
                const auto & dtype = typeid_cast<const DataTypeDecimal<Decimal128> *>(type.get());
                columns[i] = convert_and_insert_array<Decimal128>(col, cursor, count, dtype->getScale());
            }
            else if (which.isDecimal256())
            {
                const auto & dtype = typeid_cast<const DataTypeDecimal<Decimal256> *>(type.get());
                columns[i] = convert_and_insert_array<Decimal256>(col, cursor, count, dtype->getScale());
            }
            else if (which.isDateTime())
                columns[i] = convert_and_insert_array<UInt32>(col, cursor, count);
            else if (which.isDateTime64())
            {
                const auto & dtype = typeid_cast<const DataTypeDateTime64 *>(type.get());
                columns[i] = convert_and_insert_array<DateTime64>(col, cursor, count, dtype->getScale());
            }
            else if (which.isDate32())
                columns[i] = convert_and_insert_array<Int32>(col, cursor, count);
            else if (which.isDate())
                columns[i] = convert_and_insert_array<UInt16>(col, cursor, count);
            else if (which.isString())
                columns[i] = convert_and_insert_array<String>(col, cursor, count);
            else if (which.isNullable())
                columns[i] = convert_and_insert_array<String>(col, cursor, count);
            else if (which.isObject())
            {
                if (col.py_type == "list")
                {
                    columns[i] = CHDB::ListScan::scanObject(col, cursor, count, format_settings);
                }
                else
                {
                    chassert(!isPandasDataFrame);
                    columns[i] = CHDB::PandasScan::scanObject(col, cursor, count, format_settings);
                }
            }
            else
                throw Exception(ErrorCodes::BAD_TYPE_OF_FIELD, "Unsupported type {} for column {}", type->getName(), col.name);
        }
        catch (std::exception & e)
        {
            LOG_ERROR(logger, "Error processing column \"{}\": {}", col.name, e.what());
            throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED, "Error processing column \"{}\": {}", col.name, e.what());
        }
        catch (...)
        {
            LOG_ERROR(logger, "Error processing column \"{}\": unknown exception", col.name);
            throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED, "Error processing column \"{}\": unknown exception", col.name);
        }
    }

    cursor += count;

    return Chunk(std::move(columns), count);
}

Chunk PythonSource::generate()
{
    size_t num_rows = 0;
    auto names = sample_block.getNames();
    if (names.empty())
        return {};

    try
    {
        if (arrow_table_reader)
        {
            auto chunk = arrow_table_reader->readNextChunk(stream_index);
            return chunk;
        }

        if (isInheritsFromPyReader)
        {
            PyObjectVecPtr data;
            py::gil_scoped_acquire acquire;
            data = std::move(castToSharedPtrVector<py::object>(data_source.attr("read")(names, max_block_size)));
            if (data->empty())
                return {};

            return std::move(genChunk(num_rows, data));
        }

        return std::move(scanDataToChunk());
    }
    catch (const std::exception & e)
    {
        throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED, "Python data handling {}", e.what());
    }
    catch (...)
    {
        throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED, "Python data handling unknown exception");
    }
}

std::pair<size_t, size_t> PythonSource::calculateOffsetAndCount()
{
    auto rows_per_stream = data_source_row_count / num_streams;
    auto start = stream_index * rows_per_stream;
    auto end = (stream_index + 1) * rows_per_stream;
    if (stream_index == num_streams - 1)
        end = data_source_row_count;
    if (cursor == 0)
        cursor = start;
    auto count = std::min(max_block_size, end - cursor);

    return std::make_pair(cursor, count);
}

}