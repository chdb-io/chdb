#include "PandasScan.h"
#include "PythonConversion.h"
#include "PythonImporter.h"
#include "ColumnVectorHelper.h"

#include <Columns/ColumnDecimal.h>
#include <Columns/ColumnLowCardinality.h>
#include <Columns/ColumnNullable.h>
#include <Columns/ColumnObject.h>
#include <Columns/ColumnsNumber.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnVector.h>
#include <DataTypes/DataTypeLowCardinality.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypeString.h>
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

static inline const bool * getMaskPtr(const ColumnWrapper & col_wrap)
{
    return col_wrap.registered_array
        ? static_cast<const bool *>(col_wrap.registered_array->numpy_array.data())
        : nullptr;
}

template <typename T>
static void insertIntegerValue(py::handle h, typename ColumnVector<T>::Container & container)
{
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
static void scanIntegerColumn(py::handle handle, MutableColumnPtr & column)
{
    auto & nullable_column = typeid_cast<ColumnNullable &>(*column);
    auto data_column = nullable_column.getNestedColumnPtr()->assumeMutable();
    auto & null_map = nullable_column.getNullMapData();

    if (!py::isinstance<py::int_>(handle))
    {
        null_map.push_back(1);
        data_column->insertDefault();
        return;
    }

    null_map.push_back(0);
    auto & container = assert_cast<ColumnVector<T> &>(*data_column).getData();
    insertIntegerValue<T>(handle, container);
}

ColumnPtr PandasScan::scanColumn(
    const DB::ColumnWrapper & col_wrap,
    const size_t cursor,
    const size_t count,
    const DB::FormatSettings & format_settings)
{
    innerCheck(col_wrap);

    const auto & data_type = col_wrap.dest_type;
    auto column = data_type->createColumn();
    column->reserve(count);

    if (col_wrap.is_category)
    {
        chassert(data_type->lowCardinality());
        const auto & codes_type = col_wrap.category_codes_type;
        if (codes_type == "int8")
            innerScanCategory<Int8, UInt8>(cursor, count, static_cast<const Int8 *>(col_wrap.buf), col_wrap.category_unique, column, col_wrap.stride);
        else if (codes_type == "int16")
            innerScanCategory<Int16, UInt16>(cursor, count, static_cast<const Int16 *>(col_wrap.buf), col_wrap.category_unique, column, col_wrap.stride);
        else if (codes_type == "int32")
            innerScanCategory<Int32, UInt32>(cursor, count, static_cast<const Int32 *>(col_wrap.buf), col_wrap.category_unique, column, col_wrap.stride);
        else if (codes_type == "int64")
            innerScanCategory<Int64, UInt64>(cursor, count, static_cast<const Int64 *>(col_wrap.buf), col_wrap.category_unique, column, col_wrap.stride);
        else
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Unsupported category codes type: {}", codes_type);
        return column;
    }

    chassert(data_type->isNullable());
    auto real_type = removeNullable(data_type);

    WhichDataType which(real_type);

    if (col_wrap.is_object_type)
    {
        SerializationPtr serialization;
        if (which.idx == TypeIndex::Object)
            serialization = real_type->getDefaultSerialization();

        auto * object_array = static_cast<PyObject **>(col_wrap.buf);
        innerScanObject(cursor, count, format_settings, serialization, object_array, column, which, col_wrap.stride);
        return column;
    }

    switch (which.idx)
	{
    case TypeIndex::Float32:
        innerScanFloat<Float32>(cursor, count, static_cast<const Float32 *>(col_wrap.buf), column, col_wrap.stride);
        break;
    case TypeIndex::Float64:
        innerScanFloat<Float64>(cursor, count, static_cast<const Float64 *>(col_wrap.buf), column, col_wrap.stride);
        break;
    case TypeIndex::Int8:
        innerScanNumeric<Int8>(cursor, count, static_cast<const Int8 *>(col_wrap.buf), getMaskPtr(col_wrap), column, col_wrap.stride, col_wrap.mask_stride);
        break;
    case TypeIndex::Int16:
        innerScanNumeric<Int16>(cursor, count, static_cast<const Int16 *>(col_wrap.buf), getMaskPtr(col_wrap), column, col_wrap.stride, col_wrap.mask_stride);
        break;
    case TypeIndex::Int32:
        innerScanNumeric<Int32>(cursor, count, static_cast<const Int32 *>(col_wrap.buf), getMaskPtr(col_wrap), column, col_wrap.stride, col_wrap.mask_stride);
        break;
    case TypeIndex::Int64:
        innerScanNumeric<Int64>(cursor, count, static_cast<const Int64 *>(col_wrap.buf), getMaskPtr(col_wrap), column, col_wrap.stride, col_wrap.mask_stride);
        break;
    case TypeIndex::UInt8:
        innerScanNumeric<UInt8>(cursor, count, static_cast<const UInt8 *>(col_wrap.buf), getMaskPtr(col_wrap), column, col_wrap.stride, col_wrap.mask_stride);
        break;
    case TypeIndex::UInt16:
        innerScanNumeric<UInt16>(cursor, count, static_cast<const UInt16 *>(col_wrap.buf), getMaskPtr(col_wrap), column, col_wrap.stride, col_wrap.mask_stride);
        break;
    case TypeIndex::UInt32:
        innerScanNumeric<UInt32>(cursor, count, static_cast<const UInt32 *>(col_wrap.buf), getMaskPtr(col_wrap), column, col_wrap.stride, col_wrap.mask_stride);
        break;
    case TypeIndex::UInt64:
        innerScanNumeric<UInt64>(cursor, count, static_cast<const UInt64 *>(col_wrap.buf), getMaskPtr(col_wrap), column, col_wrap.stride, col_wrap.mask_stride);
        break;
    case TypeIndex::DateTime64:
        innerScanDateTime64(cursor, count, static_cast<const Int64 *>(col_wrap.buf), column, col_wrap.stride);
        break;
    case TypeIndex::Interval:
        // Interval uses ColumnVector<Int64> storage (different from DateTime64 which uses ColumnDecimal)
        // pandas timedelta64[ns] is also Int64 (nanoseconds)
        innerScanInterval(cursor, count, static_cast<const Int64 *>(col_wrap.buf), column, col_wrap.stride);
        break;
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
    WhichDataType which,
    size_t stride)
{
    py::gil_scoped_acquire acquire;
#if USE_JEMALLOC
    ::Memory::MemoryCheckScope memory_check_scope;
#endif

    const size_t effective_stride = (stride == 0) ? sizeof(PyObject *) : stride;
    const auto * base_ptr = reinterpret_cast<const char *>(objects);

    switch (which.idx)
    {
    case TypeIndex::Object:
        {
            auto & nullable_column = typeid_cast<ColumnNullable &>(*column);
            auto data_column = nullable_column.getNestedColumnPtr()->assumeMutable();
            auto & null_map = nullable_column.getNullMapData();

            for (size_t i = cursor; i < cursor + count; ++i)
            {
                auto * obj_ptr = *reinterpret_cast<PyObject * const *>(base_ptr + i * effective_stride);
                auto handle = py::handle(obj_ptr);
                if (!py::isinstance<py::dict>(handle))
                {
                    null_map.push_back(1);
                    data_column->insertDefault();
                    continue;
                }

                null_map.push_back(0);
                tryInsertJsonResult(handle, format_settings, data_column, serialization);
            }
            break;
        }
    case TypeIndex::String:
        {
            auto & nullable_col = assert_cast<ColumnNullable &>(*column);
            auto data_column = nullable_col.getNestedColumnPtr()->assumeMutable();
            auto & null_map = nullable_col.getNullMapData();
            auto * column_string = assert_cast<ColumnString *>(data_column.get());
            auto * string_chars_ptr = &column_string->getChars();

            for (size_t i = cursor; i < cursor + count; ++i)
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

                auto * obj_ptr = *reinterpret_cast<PyObject * const *>(base_ptr + i * effective_stride);
                auto handle = py::handle(obj_ptr);

                bool is_null = false;
                bool is_str = py::isinstance<py::str>(handle);
                if (!is_str)
                {
                    if (isNone(handle) || (isFloat(handle) && std::isnan(PyFloat_AsDouble(handle.ptr()))))
                        is_null = true;
                }

                if (is_null)
                {
                    null_map.push_back(1);
                    data_column->insertDefault();
                    continue;
                }

                null_map.push_back(0);
                auto * obj = handle.ptr();
                if (!is_str)
                    insertObjToStringColumn(obj, column_string);
                else
                    FillColumnString(obj, column_string);
            }
            break;
        }
    case TypeIndex::Float64:
        {
            auto & nullable_column = typeid_cast<ColumnNullable &>(*column);
            auto data_column = nullable_column.getNestedColumnPtr()->assumeMutable();
            auto & null_map = nullable_column.getNullMapData();
            auto & container = assert_cast<ColumnVector<Float64> &>(*data_column).getData();

            for (size_t i = cursor; i < cursor + count; ++i)
            {
                auto * obj_ptr = *reinterpret_cast<PyObject * const *>(base_ptr + i * effective_stride);
                auto handle = py::handle(obj_ptr);

                if (!py::isinstance<py::int_>(handle) && !py::isinstance<py::float_>(handle))
                {
                    null_map.push_back(1);
                    data_column->insertDefault();
                    continue;
                }

                if (py::isinstance<py::int_>(handle))
                {
                    double number = PyLong_AsDouble(handle.ptr());
                    if (number == -1.0 && PyErr_Occurred())
                    {
                        number = 0.0;
                        PyErr_Clear();
                    }
                    null_map.push_back(0);
                    container.push_back(number);
                }
                else
                {
                    double value = handle.cast<double>();
                    if (std::isnan(value))
                    {
                        null_map.push_back(1);
                        data_column->insertDefault();
                    }
                    else
                    {
                        null_map.push_back(0);
                        container.push_back(value);
                    }
                }
            }
            break;
        }
    case TypeIndex::Int64:
        for (size_t i = cursor; i < cursor + count; ++i)
        {
            auto * obj_ptr = *reinterpret_cast<PyObject * const *>(base_ptr + i * effective_stride);
            scanIntegerColumn<Int64>(py::handle(obj_ptr), column);
        }
        break;
    case TypeIndex::Int32:
        for (size_t i = cursor; i < cursor + count; ++i)
        {
            auto * obj_ptr = *reinterpret_cast<PyObject * const *>(base_ptr + i * effective_stride);
            scanIntegerColumn<Int32>(py::handle(obj_ptr), column);
        }
        break;
    case TypeIndex::UInt64:
        for (size_t i = cursor; i < cursor + count; ++i)
        {
            auto * obj_ptr = *reinterpret_cast<PyObject * const *>(base_ptr + i * effective_stride);
            scanIntegerColumn<UInt64>(py::handle(obj_ptr), column);
        }
        break;
    default:
        throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Unsupported nullable target type: {}", which.idx);
    }
}

template <typename T>
void PandasScan::innerScanFloat(
    const size_t cursor,
    const size_t count,
    const T * ptr,
    DB::MutableColumnPtr & column,
    size_t stride)
{
    auto & nullable_column = typeid_cast<ColumnNullable &>(*column);
    auto data_column = nullable_column.getNestedColumnPtr()->assumeMutable();
    auto & null_map = nullable_column.getNullMapData();

    if (stride == 0 || stride == sizeof(T))
    {
        ColumnVectorHelper * helper = static_cast<ColumnVectorHelper *>(data_column.get());
        const T * start = ptr + cursor;
        helper->appendRawData<sizeof(T)>(reinterpret_cast<const char *>(start), count);

        for (size_t i = 0; i < count; ++i)
        {
            bool is_nan = std::isnan(start[i]);
            null_map.push_back(is_nan ? 1 : 0);
        }
    }
    else
    {
        auto & container = assert_cast<ColumnVector<T> &>(*data_column).getData();
        const auto * base_ptr = reinterpret_cast<const char *>(ptr);
        for (size_t i = cursor; i < cursor + count; ++i)
        {
            T value = *reinterpret_cast<const T *>(base_ptr + i * stride);
            container.push_back(value);
            null_map.push_back(std::isnan(value) ? 1 : 0);
        }
    }
}

template void PandasScan::innerScanFloat<Float32>(const size_t, const size_t, const Float32 *, DB::MutableColumnPtr &, size_t);
template void PandasScan::innerScanFloat<Float64>(const size_t, const size_t, const Float64 *, DB::MutableColumnPtr &, size_t);

template <typename T>
void PandasScan::innerScanNumeric(
    const size_t cursor,
    const size_t count,
    const T * data_ptr,
    const bool * mask_ptr,
    DB::MutableColumnPtr & column,
    size_t stride,
    size_t mask_stride)
{
    auto & nullable_column = typeid_cast<ColumnNullable &>(*column);
    auto data_column = nullable_column.getNestedColumnPtr()->assumeMutable();
    auto & null_map = nullable_column.getNullMapData();

    const bool data_contiguous = (stride == 0 || stride == sizeof(T));
    const bool mask_contiguous = (mask_stride == 0 || mask_stride == sizeof(bool));

    if (data_contiguous && mask_contiguous)
    {
        ColumnVectorHelper * helper = static_cast<ColumnVectorHelper *>(data_column.get());
        const T * start = data_ptr + cursor;
        helper->appendRawData<sizeof(T)>(reinterpret_cast<const char *>(start), count);

        if (mask_ptr != nullptr)
        {
            const bool * mask_start = mask_ptr + cursor;
            null_map.insert(reinterpret_cast<const UInt8 *>(mask_start), reinterpret_cast<const UInt8 *>(mask_start + count));
        }
        else
        {
            null_map.resize_fill(null_map.size() + count, 0);
        }
    }
    else
    {
        // Slow path: non-contiguous data or mask
        auto & container = assert_cast<ColumnVector<T> &>(*data_column).getData();
        const size_t effective_stride = (stride == 0) ? sizeof(T) : stride;
        const size_t effective_mask_stride = (mask_stride == 0) ? sizeof(bool) : mask_stride;
        const auto * data_base = reinterpret_cast<const char *>(data_ptr);
        const auto * mask_base = reinterpret_cast<const char *>(mask_ptr);

        for (size_t i = cursor; i < cursor + count; ++i)
        {
            T value = *reinterpret_cast<const T *>(data_base + i * effective_stride);
            container.push_back(value);

            if (mask_ptr != nullptr)
            {
                bool is_null = *reinterpret_cast<const bool *>(mask_base + i * effective_mask_stride);
                null_map.push_back(is_null ? 1 : 0);
            }
            else
            {
                null_map.push_back(0);
            }
        }
    }
}

template void PandasScan::innerScanNumeric<Int8>(const size_t, const size_t, const Int8 *, const bool *, DB::MutableColumnPtr &, size_t, size_t);
template void PandasScan::innerScanNumeric<Int16>(const size_t, const size_t, const Int16 *, const bool *, DB::MutableColumnPtr &, size_t, size_t);
template void PandasScan::innerScanNumeric<Int32>(const size_t, const size_t, const Int32 *, const bool *, DB::MutableColumnPtr &, size_t, size_t);
template void PandasScan::innerScanNumeric<Int64>(const size_t, const size_t, const Int64 *, const bool *, DB::MutableColumnPtr &, size_t, size_t);
template void PandasScan::innerScanNumeric<UInt8>(const size_t, const size_t, const UInt8 *, const bool *, DB::MutableColumnPtr &, size_t, size_t);
template void PandasScan::innerScanNumeric<UInt16>(const size_t, const size_t, const UInt16 *, const bool *, DB::MutableColumnPtr &, size_t, size_t);
template void PandasScan::innerScanNumeric<UInt32>(const size_t, const size_t, const UInt32 *, const bool *, DB::MutableColumnPtr &, size_t, size_t);
template void PandasScan::innerScanNumeric<UInt64>(const size_t, const size_t, const UInt64 *, const bool *, DB::MutableColumnPtr &, size_t, size_t);

void PandasScan::innerScanDateTime64(
    const size_t cursor,
    const size_t count,
    const Int64 * ptr,
    DB::MutableColumnPtr & column,
    size_t stride)
{
    auto & nullable_column = typeid_cast<ColumnNullable &>(*column);
    auto data_column = nullable_column.getNestedColumnPtr()->assumeMutable();
    auto & null_map = nullable_column.getNullMapData();

    if (stride == 0 || stride == sizeof(Int64))
    {
        ColumnVectorHelper * helper = static_cast<ColumnVectorHelper *>(data_column.get());
        const Int64 * start = ptr + cursor;
        helper->appendRawData<sizeof(Int64)>(reinterpret_cast<const char *>(start), count);

        for (size_t i = 0; i < count; ++i)
        {
            bool is_nat = start[i] <= std::numeric_limits<Int64>::min();
            null_map.push_back(is_nat ? 1 : 0);
        }
    }
    else
    {
        // DateTime64 uses ColumnDecimal<DateTime64>, which has the same memory layout as Int64
        auto & container = assert_cast<ColumnDecimal<DateTime64> &>(*data_column).getData();
        const auto * base_ptr = reinterpret_cast<const char *>(ptr);
        for (size_t i = cursor; i < cursor + count; ++i)
        {
            Int64 value = *reinterpret_cast<const Int64 *>(base_ptr + i * stride);
            container.push_back(DateTime64(value));
            bool is_nat = value <= std::numeric_limits<Int64>::min();
            null_map.push_back(is_nat ? 1 : 0);
        }
    }
}

void PandasScan::innerScanInterval(
    const size_t cursor,
    const size_t count,
    const Int64 * ptr,
    DB::MutableColumnPtr & column,
    size_t stride)
{
    auto & nullable_column = typeid_cast<ColumnNullable &>(*column);
    auto data_column = nullable_column.getNestedColumnPtr()->assumeMutable();
    auto & null_map = nullable_column.getNullMapData();

    if (stride == 0 || stride == sizeof(Int64))
    {
        ColumnVectorHelper * helper = static_cast<ColumnVectorHelper *>(data_column.get());
        const Int64 * start = ptr + cursor;
        helper->appendRawData<sizeof(Int64)>(reinterpret_cast<const char *>(start), count);

        for (size_t i = 0; i < count; ++i)
        {
            bool is_nat = start[i] <= std::numeric_limits<Int64>::min();
            null_map.push_back(is_nat ? 1 : 0);
        }
    }
    else
    {
        // Interval uses ColumnVector<Int64>
        auto & container = assert_cast<ColumnVector<Int64> &>(*data_column).getData();
        const auto * base_ptr = reinterpret_cast<const char *>(ptr);
        for (size_t i = cursor; i < cursor + count; ++i)
        {
            Int64 value = *reinterpret_cast<const Int64 *>(base_ptr + i * stride);
            container.push_back(value);
            bool is_nat = value <= std::numeric_limits<Int64>::min();
            null_map.push_back(is_nat ? 1 : 0);
        }
    }
}

void PandasScan::innerCheck(const ColumnWrapper & col_wrap)
{
    if (col_wrap.data.is_none())
        throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED, "Column data is None");

    if (!col_wrap.buf)
        throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED, "Column buffer is null");
}

template <typename T, typename IndexType>
void PandasScan::innerScanCategory(
    const size_t cursor,
    const size_t count,
    const T * codes_ptr,
    const ColumnUniquePtr & category_unique,
    MutableColumnPtr & column,
    size_t stride)
{
    const size_t effective_stride = (stride == 0) ? sizeof(T) : stride;
    const auto * base_ptr = reinterpret_cast<const char *>(codes_ptr);

    auto indexes_column = ColumnVector<IndexType>::create();
    auto & indexes_data = indexes_column->getData();
    indexes_data.reserve(count);

    for (size_t i = cursor; i < cursor + count; ++i)
    {
        T code = *reinterpret_cast<const T *>(base_ptr + i * effective_stride);
        indexes_data.push_back(code < 0 ? 0 : static_cast<IndexType>(code + 1));
    }

    column = ColumnLowCardinality::create(category_unique->assumeMutable(), std::move(indexes_column), true);
}

template void PandasScan::innerScanCategory<Int8, UInt8>(size_t, size_t, const Int8 *, const ColumnUniquePtr &, MutableColumnPtr &, size_t);
template void PandasScan::innerScanCategory<Int16, UInt16>(size_t, size_t, const Int16 *, const ColumnUniquePtr &, MutableColumnPtr &, size_t);
template void PandasScan::innerScanCategory<Int32, UInt32>(size_t, size_t, const Int32 *, const ColumnUniquePtr &, MutableColumnPtr &, size_t);
template void PandasScan::innerScanCategory<Int64, UInt64>(size_t, size_t, const Int64 *, const ColumnUniquePtr &, MutableColumnPtr &, size_t);

} // namespace CHDB
