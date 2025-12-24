#include "PandasScan.h"
#include "PythonConversion.h"
#include "PythonImporter.h"
#include "ColumnVectorHelper.h"

#include <Columns/ColumnNullable.h>
#include <Columns/ColumnObject.h>
#include <Columns/ColumnVector.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/IDataType.h>
#include <DataTypes/Serializations/SerializationJSON.h>
#include <IO/WriteHelpers.h>
#include <base/defines.h>
#include <Common/assert_cast.h>
#if USE_JEMALLOC
#    include <Common/memory.h>
#endif

namespace DB
{

namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
    extern const int PY_EXCEPTION_OCCURED;
}

}

using namespace DB;

namespace CHDB
{

namespace
{

template <typename T>
void insertIntegerValue(py::handle h, typename ColumnVector<T>::Container & container, PythonObjectType object_type)
{
    if (object_type != PythonObjectType::Integer)
    {
        container.push_back(T{});
        return;
    }

    auto * ptr = h.ptr();

    if constexpr (std::is_signed_v<T>)
    {
        int overflow;
        int64_t value = PyLong_AsLongLongAndOverflow(ptr, &overflow);
        if (overflow)
        {
            container.push_back(T{});
            return;
        }

        if constexpr (sizeof(T) < sizeof(int64_t))
        {
            if (value > std::numeric_limits<T>::max() || value < std::numeric_limits<T>::min())
            {
                container.push_back(T{});
                return;
            }
        }
        container.push_back(static_cast<T>(value));
    }
    else
    {
        uint64_t value = PyLong_AsUnsignedLongLong(ptr);
        if (PyErr_Occurred())
        {
            PyErr_Clear();
            container.push_back(T{});
            return;
        }

        if constexpr (sizeof(T) < sizeof(uint64_t))
        {
            if (value > std::numeric_limits<T>::max())
            {
                container.push_back(T{});
                return;
            }
        }
        container.push_back(static_cast<T>(value));
    }
}

template <typename T>
void scanIntegerColumn(py::handle handle, MutableColumnPtr & column)
{
    transformPythonObject(handle, column, [](py::handle h, MutableColumnPtr & col, PythonObjectType object_type)
    {
        auto & container = assert_cast<ColumnVector<T> &>(*col).getData();
        insertIntegerValue<T>(h, container, object_type);
    });
}

} // anonymous namespace

ColumnPtr PandasScan::scanColumn(
    const DB::ColumnWrapper & col_wrap,
    const size_t cursor,
    const size_t count,
    const DB::FormatSettings & format_settings)
{
    innerCheck(col_wrap);

    const auto & data_type = col_wrap.dest_type;
    chassert(data_type->isNullable());
    auto column = data_type->createColumn();
    column->reserve(count);

    auto real_type = removeNullable(data_type);

    WhichDataType which(real_type);


    if (col_wrap.is_object_type)
    {
        SerializationPtr serialization;
        if (which.idx == TypeIndex::Object)
            serialization = real_type->getDefaultSerialization();

        auto * object_array = static_cast<PyObject **>(col_wrap.buf);
        innerScanObject(cursor, count, format_settings, serialization, object_array, column, which);
        return column;
    }


    switch (which.idx)
	{
    case TypeIndex::Float64:
        {
            const auto * float64_array = static_cast<const Float64 *>(col_wrap.buf);
            innerScanFloat64(cursor, count, float64_array, column);
            break;
        }
    case TypeIndex::DateTime64:
        {
            const auto * int64_array = static_cast<const Int64 *>(col_wrap.buf);
            innerScanDateTime64(cursor, count, int64_array, column);
            break;
        }
    default:
        throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Unsupported target type: {}", which.idx);
	}

    return column;
}

ColumnPtr PandasScan::scanObject(
    const ColumnWrapper & col_wrap,
    const size_t cursor,
    const size_t count,
    const FormatSettings & format_settings)
{
    innerCheck(col_wrap);

    const auto & data_type = col_wrap.dest_type;
    auto column = data_type->createColumn();
    auto ** object_array = static_cast<PyObject **>(col_wrap.buf);
    auto serialization = data_type->getDefaultSerialization();

    innerScanObject(cursor, count, format_settings, serialization, object_array, column);

    return column;
}

void PandasScan::scanObject(
    const size_t cursor,
    const size_t count,
    const FormatSettings & format_settings,
    const void * buf,
    MutableColumnPtr & column)
{
    auto * object_array = static_cast<PyObject **>(const_cast<void *>(buf));
    auto data_type = std::make_shared<DataTypeObject>(DataTypeObject::SchemaFormat::JSON);
    SerializationPtr serialization = data_type->getDefaultSerialization();

    innerScanObject(cursor, count, format_settings, serialization, object_array, column);
}

void PandasScan::innerScanObject(
    const size_t cursor,
    const size_t count,
    const FormatSettings & format_settings,
    SerializationPtr & serialization,
    PyObject ** objects,
    MutableColumnPtr & column,
    WhichDataType which)
{
    py::gil_scoped_acquire acquire;
#if USE_JEMALLOC
    ::Memory::MemoryCheckScope memory_check_scope;
#endif

    ColumnString::Chars * string_chars_ptr = nullptr;
    if (which.idx == TypeIndex::String)
    {
        auto & nullable_col = assert_cast<ColumnNullable &>(*column);
        auto * column_string = assert_cast<ColumnString *>(nullable_col.getNestedColumn().assumeMutable().get());
        string_chars_ptr = &column_string->getChars();
    }

    for (size_t i = cursor; i < cursor + count; ++i)
    {
        auto * obj = objects[i];
        auto handle = py::handle(obj);

        switch (which.idx)
	    {
        case TypeIndex::Object:
            {
                transformPythonObject(handle, column, [&](py::handle h, MutableColumnPtr & col, PythonObjectType /* object_type */)
                {
                    if (!tryInsertJsonResult(h, format_settings, col, serialization))
                        col->insertDefault();
                });
                break;
            }
        case TypeIndex::String:
            {
                size_t local_idx = i - cursor;
                if (local_idx % 10 == 9)
                {
                    size_t data_size = string_chars_ptr->size();
                    size_t counter = local_idx + 1;
                    size_t avg_size = data_size / counter;
                    size_t reserve_size = avg_size * count;
                    if (reserve_size > string_chars_ptr->capacity())
                        string_chars_ptr->reserve(reserve_size);
                }

                transformPythonObject(handle, column, [](py::handle h, MutableColumnPtr & col, PythonObjectType /* object_type */)
                {
                    auto * obj = h.ptr();
                    auto * col_string = assert_cast<ColumnString *>(col.get());

                    if (!PyUnicode_Check(obj))
                        insertObjToStringColumn(obj, col_string);
                    else
                        FillColumnString(obj, col_string);
                });
                break;
            }
        case TypeIndex::Float64:
            {
                transformPythonObject(handle, column, [&](py::handle h, MutableColumnPtr & col, PythonObjectType object_type)
                {
                    auto & container = assert_cast<ColumnVector<Float64> &>(*col).getData();
                    switch (object_type)
                    {
                    case PythonObjectType::Float:
                        {
                            container.push_back(h.cast<double>());
                            break;
                        }
                    case PythonObjectType::Integer:
                        {
                            double number = PyLong_AsDouble(handle.ptr());
                            if (number == -1.0 && PyErr_Occurred())
                            {
                                number = 0.0;
                                PyErr_Clear();
                            }
                            container.push_back(number);
                            break;
                        }
                    default:
                        {
                            container.push_back(0.0);
                            break;
                        }
                    }
                });
                break;
            }
        case TypeIndex::Int64:
            scanIntegerColumn<Int64>(handle, column);
            break;
        case TypeIndex::Int32:
            scanIntegerColumn<Int32>(handle, column);
            break;
        case TypeIndex::UInt64:
            scanIntegerColumn<UInt64>(handle, column);
            break;
        default:
            throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Unsupported nullable target type: {}", which.idx);
        }
    }
}

void PandasScan::innerScanFloat64(
    const size_t cursor,
    const size_t count,
    const Float64 * ptr,
    DB::MutableColumnPtr & column)
{
    py::gil_scoped_acquire acquire;
#if USE_JEMALLOC
    ::Memory::MemoryCheckScope memory_check_scope;
#endif

    auto & nullable_column = typeid_cast<ColumnNullable &>(*column);
    auto data_column = nullable_column.getNestedColumnPtr()->assumeMutable();
    auto & null_map = nullable_column.getNullMapData();

    ColumnVectorHelper * helper = static_cast<ColumnVectorHelper *>(data_column.get());
    const Float64 * start = ptr + cursor;
    helper->appendRawData<sizeof(Float64)>(reinterpret_cast<const char *>(start), count);

    for (size_t i = 0; i < count; ++i)
    {
        bool is_nan = std::isnan(start[i]);
        null_map.push_back(is_nan ? 1 : 0);
    }
}

void PandasScan::innerScanDateTime64(
    const size_t cursor,
    const size_t count,
    const Int64 * ptr,
    DB::MutableColumnPtr & column)
{
    py::gil_scoped_acquire acquire;
#if USE_JEMALLOC
    ::Memory::MemoryCheckScope memory_check_scope;
#endif

    auto & nullable_column = typeid_cast<ColumnNullable &>(*column);
    auto data_column = nullable_column.getNestedColumnPtr()->assumeMutable();
    auto & null_map = nullable_column.getNullMapData();

    ColumnVectorHelper * helper = static_cast<ColumnVectorHelper *>(data_column.get());
    const Int64 * start = ptr + cursor;
    helper->appendRawData<sizeof(Int64)>(reinterpret_cast<const char *>(start), count);

    for (size_t i = 0; i < count; ++i)
    {
        bool is_nat = start[i] <= std::numeric_limits<Int64>::min();
        null_map.push_back(is_nat ? 1 : 0);
    }
}

void PandasScan::innerCheck(const ColumnWrapper & col_wrap)
{
    if (col_wrap.data.is_none())
        throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED, "Column data is None");

    if (!col_wrap.buf)
        throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED, "Column buffer is null");
}

} // namespace CHDB
