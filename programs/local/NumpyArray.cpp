#include "NumpyArray.h"
#include "NumpyType.h"
#include "NumpyNestedTypes.h"
#include "PythonImporter.h"
#include "FieldToPython.h"

#include <Processors/Chunk.h>
#include <base/defines.h>
#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnLowCardinality.h>
#include <Columns/ColumnNullable.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnVector.h>
#include <Columns/ColumnsNumber.h>
#include <Common/logger_useful.h>
#include <Core/UUID.h>
#include <DataTypes/DataTypeDateTime.h>
#include <DataTypes/DataTypeDateTime64.h>
#include <DataTypes/DataTypeEnum.h>
#include <DataTypes/DataTypeLowCardinality.h>
#include <DataTypes/DataTypeFactory.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesDecimal.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeTime.h>
#include <DataTypes/DataTypeTime64.h>
#include <Interpreters/castColumn.h>
#include <base/Decimal.h>
#include <base/types.h>
#include <Common/formatIPv6.h>
#include <pybind11/pytypes.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int NOT_IMPLEMENTED;
	extern const int LOGICAL_ERROR;
}

}

using namespace DB;

namespace CHDB
{

struct RegularConvert
{
	template <class CHTYPE, class NUMPYTYPE>
	static NUMPYTYPE convertValue(CHTYPE val, NumpyAppendData & append_data)
	{
		(void)append_data;
		return (NUMPYTYPE)val;
	}

	template <class NUMPYTYPE>
	static NUMPYTYPE nullValue(bool & set_mask)
	{
		set_mask = true;
		return 0;
	}
};

struct TimeConvert
{
	template <class CHTYPE, class NUMPYTYPE>
	static NUMPYTYPE convertValue(CHTYPE val, NumpyAppendData & append_data)
	{
		chassert(append_data.type);

		Field field(static_cast<Int64>(val));
		auto time_object = convertTimeFieldToPython(field);
		return time_object.release().ptr();
	}

	template <class NUMPYTYPE>
	static NUMPYTYPE nullValue(bool & set_mask)
	{
		set_mask = true;
		return nullptr;
	}
};

struct Time64Convert
{
	template <class CHTYPE, class NUMPYTYPE>
	static NUMPYTYPE convertValue(CHTYPE val, NumpyAppendData & append_data)
	{
		chassert(append_data.type);

		const auto & time64_type = typeid_cast<const DataTypeTime64 &>(*append_data.type);
		UInt32 scale = time64_type.getScale();
		DecimalField<Decimal64> decimal_field(static_cast<Decimal64::NativeType>(val), scale);
		Field field(decimal_field);

		auto time64_object = convertTime64FieldToPython(field);
		return time64_object.release().ptr();
	}

	template <class NUMPYTYPE>
	static NUMPYTYPE nullValue(bool & set_mask)
	{
		set_mask = true;
		return nullptr;
	}
};

struct Enum8Convert
{
	template <class CHTYPE, class NUMPYTYPE>
	static NUMPYTYPE convertValue(CHTYPE val, NumpyAppendData & append_data)
	{
		const auto & enum_type = typeid_cast<const DataTypeEnum8 &>(*append_data.type);

		try
		{
			auto it = enum_type.findByValue(static_cast<Int8>(val));
			String enum_name(it->second.data, it->second.size);
			return py::str(enum_name).release().ptr();
		}
		catch (...)
		{
			return py::str(toString(static_cast<Int8>(val))).release().ptr();
		}
	}

	template <class NUMPYTYPE>
	static NUMPYTYPE nullValue(bool & set_mask)
	{
		set_mask = true;
		return nullptr;
	}
};

struct Enum16Convert
{
	template <class CHTYPE, class NUMPYTYPE>
	static NUMPYTYPE convertValue(CHTYPE val, NumpyAppendData & append_data)
	{
		const auto & enum_type = typeid_cast<const DataTypeEnum16 &>(*append_data.type);
		try
		{
			auto it = enum_type.findByValue(static_cast<Int16>(val));
			String enum_name(it->second.data, it->second.size);
			return py::str(enum_name).release().ptr();
		}
		catch (...)
		{
			return py::str(toString(static_cast<Int16>(val))).release().ptr();
		}
	}

	template <class NUMPYTYPE>
	static NUMPYTYPE nullValue(bool & set_mask)
	{
		set_mask = true;
		return nullptr;
	}
};

template <class CHTYPE, class NUMPYTYPE, class CONVERT>
static bool TransformColumn(NumpyAppendData & append_data)
{
	bool has_null = false;
	const IColumn * data_column = &append_data.column;
	const ColumnNullable * nullable_column = nullptr;

	/// Check if column is nullable
	if (const auto * nullable = typeid_cast<const ColumnNullable *>(&append_data.column))
	{
		nullable_column = nullable;
		data_column = &nullable->getNestedColumn();
	}

	const auto * tmp_ptr = static_cast<const ColumnFixedSizeHelper *>(data_column)->getRawDataBegin<sizeof(CHTYPE)>();
	const auto * src_ptr = reinterpret_cast<const CHTYPE *>(tmp_ptr);
	auto * dest_ptr = reinterpret_cast<NUMPYTYPE *>(append_data.target_data);
	auto * mask_ptr = append_data.target_mask;

	for (size_t i = 0; i < append_data.src_count; i++)
	{
		size_t src_index = append_data.src_offset + i;
		size_t dest_index = append_data.dest_offset + i;
		if (nullable_column && nullable_column->isNullAt(src_index))
		{
			dest_ptr[dest_index] = CONVERT::template nullValue<NUMPYTYPE>(mask_ptr[dest_index]);
			has_null = has_null || mask_ptr[dest_index];
		}
		else
		{
			dest_ptr[dest_index] = CONVERT::template convertValue<CHTYPE, NUMPYTYPE>(src_ptr[src_index], append_data);
			mask_ptr[dest_index] = false;
		}
	}

	return has_null;
}

template <class T>
static bool CHColumnToNumpyArray(NumpyAppendData & append_data)
{
	return TransformColumn<T, T, RegularConvert>(append_data);
}

template <typename DecimalType>
static bool CHColumnDecimalToNumpyArray(NumpyAppendData & append_data, const DataTypePtr & data_type)
{
	bool has_null = false;
	const IColumn * data_column = &append_data.column;
	const ColumnNullable * nullable_column = nullptr;

	/// Check if column is nullable
	if (const auto * nullable = typeid_cast<const ColumnNullable *>(&append_data.column))
	{
		nullable_column = nullable;
		data_column = &nullable->getNestedColumn();
	}

	const auto * decimal_column = typeid_cast<const ColumnDecimal<DecimalType> *>(data_column);
	if (!decimal_column)
		throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected ColumnDecimal");

	/// Get scale from data type to convert integer to actual decimal value
	const auto * decimal_type = typeid_cast<const DataTypeDecimal<DecimalType> *>(data_type.get());
	if (!decimal_type)
		throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected DataTypeDecimal");

	UInt32 scale = decimal_type->getScale();

	auto * dest_ptr = reinterpret_cast<double *>(append_data.target_data);
	auto * mask_ptr = append_data.target_mask;

	for (size_t i = 0; i < append_data.src_count; i++)
	{
		size_t src_index = append_data.src_offset + i;
		size_t dest_index = append_data.dest_offset + i;
		if (nullable_column && nullable_column->isNullAt(src_index))
		{
			/// Set to 0.0 for null values
			dest_ptr[dest_index] = 0.0;
			mask_ptr[dest_index] = true;
			has_null = true;
		}
		else
		{
			auto decimal_value = decimal_column->getElement(src_index);
			dest_ptr[dest_index] = DecimalUtils::convertTo<double>(decimal_value, scale);
			mask_ptr[dest_index] = false;
		}
	}

	return has_null;
}

static bool CHColumnDateTime64ToNumpyArray(NumpyAppendData & append_data)
{
	bool has_null = false;
	const IColumn * data_column = &append_data.column;
	const ColumnNullable * nullable_column = nullptr;

	if (const auto * nullable = typeid_cast<const ColumnNullable *>(&append_data.column))
	{
		nullable_column = nullable;
		data_column = &nullable->getNestedColumn();
	}

	const auto * decimal_column = typeid_cast<const ColumnDecimal<DateTime64> *>(data_column);
	if (!decimal_column)
		throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected ColumnDecimal<DateTime64>");

	auto * dest_ptr = reinterpret_cast<Int64 *>(append_data.target_data);
	auto * mask_ptr = append_data.target_mask;

	for (size_t i = 0; i < append_data.src_count; i++)
	{
		size_t src_index = append_data.src_offset + i;
		size_t dest_index = append_data.dest_offset + i;
		if (nullable_column && nullable_column->isNullAt(src_index))
		{
			dest_ptr[dest_index] = 0;
			mask_ptr[dest_index] = true;
			has_null = true;
		}
		else
		{
			/// Get the DateTime64 value and convert to nanoseconds
			Int64 raw_value = decimal_column->getInt(src_index);
			auto scale = decimal_column->getScale();

			Int64 ns_value;
			chassert(scale <= 9);
			Int64 multiplier = common::exp10_i32(9 - scale);
			ns_value = raw_value * multiplier;

			dest_ptr[dest_index] = ns_value;
			mask_ptr[dest_index] = false;
		}
	}

	return has_null;
}

static bool CHColumnIntervalToNumpyArray(NumpyAppendData & append_data)
{
	bool has_null = false;
	const IColumn * data_column = &append_data.column;
	const ColumnNullable * nullable_column = nullptr;

	/// Check if column is nullable
	if (const auto * nullable = typeid_cast<const ColumnNullable *>(&append_data.column))
	{
		nullable_column = nullable;
		data_column = &nullable->getNestedColumn();
	}

	const auto * int64_column = typeid_cast<const ColumnVector<Int64> *>(data_column);
	if (!int64_column)
		throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected ColumnVector<Int64> for Interval");

	auto * dest_ptr = reinterpret_cast<Int64 *>(append_data.target_data);
	auto * mask_ptr = append_data.target_mask;

	for (size_t i = 0; i < append_data.src_count; i++)
	{
		size_t src_index = append_data.src_offset + i;
		size_t dest_index = append_data.dest_offset + i;
		if (nullable_column && nullable_column->isNullAt(src_index))
		{
			dest_ptr[dest_index] = 0;
			mask_ptr[dest_index] = true;
			has_null = true;
		}
		else
		{
			Int64 interval_value = int64_column->getElement(src_index);

			/// Convert quarter to month by multiplying by 3
			/// This function is only called for Quarter intervals
			interval_value *= 3;

			dest_ptr[dest_index] = interval_value;
			mask_ptr[dest_index] = false;
		}
	}

	return has_null;
}

static bool CHColumnUUIDToNumpyArray(NumpyAppendData & append_data)
{
	bool has_null = false;
	const IColumn * data_column = &append_data.column;
	const ColumnNullable * nullable_column = nullptr;

	/// Check if column is nullable
	if (const auto * nullable = typeid_cast<const ColumnNullable *>(&append_data.column))
	{
		nullable_column = nullable;
		data_column = &nullable->getNestedColumn();
	}

	const auto * uuid_column = typeid_cast<const ColumnVector<UUID> *>(data_column);
	if (!uuid_column)
		throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected ColumnVector<UUID>");

	auto * dest_ptr = reinterpret_cast<PyObject **>(append_data.target_data);
	auto * mask_ptr = append_data.target_mask;

	for (size_t i = 0; i < append_data.src_count; i++)
	{
		size_t src_index = append_data.src_offset + i;
		size_t dest_index = append_data.dest_offset + i;
		if (nullable_column && nullable_column->isNullAt(src_index))
		{
			dest_ptr[dest_index] = nullptr;
			has_null = true;
			mask_ptr[dest_index] = true;
		}
		else
		{
			/// Convert UUID to Python uuid.UUID object
			UUID uuid_value = uuid_column->getElement(src_index);
			const auto formatted_uuid = formatUUID(uuid_value);
			const char * uuid_str = formatted_uuid.data();
			const size_t uuid_str_len = formatted_uuid.size();

			/// Create Python uuid.UUID object
			auto & import_cache = PythonImporter::ImportCache();
			py::handle uuid_handle = import_cache.uuid.UUID()(String(uuid_str, uuid_str_len)).release();
			dest_ptr[dest_index] = uuid_handle.ptr();
			mask_ptr[dest_index] = false;
		}
	}

	return has_null;
}

static bool CHColumnIPv4ToNumpyArray(NumpyAppendData & append_data)
{
	bool has_null = false;
	const IColumn * data_column = &append_data.column;
	const ColumnNullable * nullable_column = nullptr;

	/// Check if column is nullable
	if (const auto * nullable = typeid_cast<const ColumnNullable *>(&append_data.column))
	{
		nullable_column = nullable;
		data_column = &nullable->getNestedColumn();
	}

	const auto * ipv4_column = typeid_cast<const ColumnVector<IPv4> *>(data_column);
	if (!ipv4_column)
		throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected ColumnVector<IPv4>");

	auto * dest_ptr = reinterpret_cast<PyObject **>(append_data.target_data);
	auto * mask_ptr = append_data.target_mask;

	for (size_t i = 0; i < append_data.src_count; i++)
	{
		size_t src_index = append_data.src_offset + i;
		size_t dest_index = append_data.dest_offset + i;
		if (nullable_column && nullable_column->isNullAt(src_index))
		{
			dest_ptr[dest_index] = nullptr;
			has_null = true;
			mask_ptr[dest_index] = true;
		}
		else
		{
			/// Convert IPv4 to Python ipaddress.IPv4Address object
			IPv4 ipv4_value = ipv4_column->getElement(src_index);

			char ipv4_str[IPV4_MAX_TEXT_LENGTH];
			char * ptr = ipv4_str;
			formatIPv4(reinterpret_cast<const unsigned char*>(&ipv4_value), ptr);
			const size_t ipv4_str_len = ptr - ipv4_str;

			/// Create Python ipaddress.IPv4Address object
			auto & import_cache = PythonImporter::ImportCache();
			py::handle ipv4_handle = import_cache.ipaddress.ipv4_address()(String(ipv4_str, ipv4_str_len)).release();
			dest_ptr[dest_index] = ipv4_handle.ptr();
			mask_ptr[dest_index] = false;
		}
	}

	return has_null;
}

static bool CHColumnIPv6ToNumpyArray(NumpyAppendData & append_data)
{
	bool has_null = false;
	const IColumn * data_column = &append_data.column;
	const ColumnNullable * nullable_column = nullptr;

	/// Check if column is nullable
	if (const auto * nullable = typeid_cast<const ColumnNullable *>(&append_data.column))
	{
		nullable_column = nullable;
		data_column = &nullable->getNestedColumn();
	}

	const auto * ipv6_column = typeid_cast<const ColumnVector<IPv6> *>(data_column);
	if (!ipv6_column)
		throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected ColumnVector<IPv6>");

	auto * dest_ptr = reinterpret_cast<PyObject **>(append_data.target_data);
	auto * mask_ptr = append_data.target_mask;

	for (size_t i = 0; i < append_data.src_count; i++)
	{
		size_t src_index = append_data.src_offset + i;
		size_t dest_index = append_data.dest_offset + i;
		if (nullable_column && nullable_column->isNullAt(src_index))
		{
			dest_ptr[dest_index] = nullptr;
			has_null = true;
			mask_ptr[dest_index] = true;
		}
		else
		{
			/// Convert IPv6 to Python ipaddress.IPv6Address object
			IPv6 ipv6_value = ipv6_column->getElement(src_index);

			/// Use ClickHouse's built-in IPv6 formatting function
			char ipv6_str[IPV6_MAX_TEXT_LENGTH];
			char * ptr = ipv6_str;
			formatIPv6(reinterpret_cast<const unsigned char*>(&ipv6_value), ptr);
			const size_t ipv6_str_len = ptr - ipv6_str;

			/// Create Python ipaddress.IPv6Address object
			auto & import_cache = PythonImporter::ImportCache();
			py::handle ipv6_handle = import_cache.ipaddress.ipv6_address()(String(ipv6_str, ipv6_str_len)).release();
			dest_ptr[dest_index] = ipv6_handle.ptr();
			mask_ptr[dest_index] = false;
		}
	}

	return has_null;
}

template <typename StringColumnType>
static bool CHColumnStringToNumpyArray(NumpyAppendData & append_data)
{
	bool has_null = false;
	const IColumn * data_column = &append_data.column;
	const ColumnNullable * nullable_column = nullptr;

	/// Check if column is nullable
	if (const auto * nullable = typeid_cast<const ColumnNullable *>(&append_data.column))
	{
		nullable_column = nullable;
		data_column = &nullable->getNestedColumn();
	}

	const auto * string_column = typeid_cast<const StringColumnType *>(data_column);
	if (!string_column)
		throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected String ColumnType");

	auto * dest_ptr = reinterpret_cast<PyObject **>(append_data.target_data);

	for (size_t i = 0; i < append_data.src_count; i++)
	{
		size_t src_index = append_data.src_offset + i;
		size_t dest_index = append_data.dest_offset + i;
		if (nullable_column && nullable_column->isNullAt(src_index))
		{
			Py_INCREF(Py_None);
			dest_ptr[dest_index] = Py_None;
		}
		else
		{
			StringRef str_ref = string_column->getDataAt(src_index);
			auto * str_ptr = const_cast<char *>(str_ref.data);
			auto str_size = str_ref.size;
			dest_ptr[dest_index] = PyUnicode_FromStringAndSize(str_ptr, str_size);
		}
	}

	return has_null;
}

NumpyAppendData::NumpyAppendData(
	const DB::IColumn & column_,
	const DB::DataTypePtr & type_)
	: column(column_)
	, type(type_)
	, src_offset(0)
	, src_count(0)
	, dest_offset(0)
	, target_data(nullptr)
	, target_mask(nullptr)
{
}

InternalNumpyArray::InternalNumpyArray(const DataTypePtr & type_)
	: data(nullptr)
	, type(type_)
	, count(0)
{
}

void InternalNumpyArray::init(size_t capacity)
{
	String type_str = DataTypeToNumpyTypeStr(type);

	array = py::array(py::dtype(type_str), capacity);
	data = reinterpret_cast<UInt8 *>(array.mutable_data());
}

void InternalNumpyArray::resize(size_t capacity)
{
	std::vector<py::ssize_t> new_shape {py::ssize_t(capacity)};

	array.resize(new_shape, false);
	data = reinterpret_cast<UInt8 *>(array.mutable_data());
}

NumpyArray::NumpyArray(const DataTypePtr & type_)
	: hava_null(false)
{
	data_array = std::make_unique<InternalNumpyArray>(type_);
}

void NumpyArray::init(size_t capacity, bool may_have_null)
{
	data_array->init(capacity);

	if (may_have_null)
	{
		if (!mask_array)
			mask_array = std::make_unique<InternalNumpyArray>(DataTypeFactory::instance().get("Bool"));

		mask_array->init(capacity);
	}
}

void NumpyArray::resize(size_t capacity, bool may_have_null)
{
	data_array->resize(capacity);

	if (may_have_null)
	{
		if (!mask_array)
			mask_array = std::make_unique<InternalNumpyArray>(DataTypeFactory::instance().get("Bool"));

		mask_array->resize(capacity);
	}
}

static bool CHColumnNothingToNumpyArray(NumpyAppendData & append_data)
{
	/// Nothing type represents columns with no actual values, so we fill all positions with None
	bool has_null = true;
	auto * dest_ptr = reinterpret_cast<PyObject **>(append_data.target_data);
	auto * mask_ptr = append_data.target_mask;

	for (size_t i = 0; i < append_data.src_count; i++)
	{
		size_t dest_index = append_data.dest_offset + i;

		dest_ptr[dest_index] = nullptr;
		mask_ptr[dest_index] = true;
	}

	return has_null;
}

void NumpyArray::append(const ColumnPtr & column)
{
	append(column, 0, column->size());
}

void NumpyArray::append(
	const ColumnPtr & column,
	size_t offset,
	size_t count)
{
	auto actual_column = column->convertToFullColumnIfLowCardinality();
	DataTypePtr actual_type = removeLowCardinalityAndNullable(data_array->type);

	chassert(data_array);
	chassert(mask_array);

	auto * data_ptr = data_array->data;
	auto * mask_ptr = reinterpret_cast<bool *>(mask_array->data);
	chassert(data_ptr);
	chassert(mask_ptr);
	chassert(actual_column->isNullable() || actual_column->getDataType() == actual_type->getColumnType());

	data_array->count += count;
	mask_array->count += count;
	bool may_have_null = false;

	NumpyAppendData append_data(*actual_column, actual_type);
	append_data.src_offset = offset;
	append_data.src_count = count;
	append_data.target_data = data_ptr;
	append_data.target_mask = mask_ptr;
	append_data.dest_offset = data_array->count - count;

	switch (actual_type->getTypeId())
	{
	case TypeIndex::Nothing:
		may_have_null = CHColumnNothingToNumpyArray(append_data);
		break;

	case TypeIndex::Int8:
		may_have_null = CHColumnToNumpyArray<Int8>(append_data);
		break;

	case TypeIndex::UInt8:
		{
			auto is_bool = isBool(actual_type);
			if (is_bool)
				may_have_null = CHColumnToNumpyArray<bool>(append_data);
			else
				may_have_null = CHColumnToNumpyArray<UInt8>(append_data);
		}
		break;

	case TypeIndex::Int16:
		may_have_null = CHColumnToNumpyArray<Int16>(append_data);
		break;

	case TypeIndex::UInt16:
		may_have_null = CHColumnToNumpyArray<UInt16>(append_data);
		break;

	case TypeIndex::Int32:
		may_have_null = CHColumnToNumpyArray<Int32>(append_data);
		break;

	case TypeIndex::UInt32:
		may_have_null = CHColumnToNumpyArray<UInt32>(append_data);
		break;

	case TypeIndex::Int64:
		may_have_null = CHColumnToNumpyArray<Int64>(append_data);
		break;

	case TypeIndex::UInt64:
		may_have_null = CHColumnToNumpyArray<UInt64>(append_data);
		break;

	case TypeIndex::Float32:
		may_have_null = CHColumnToNumpyArray<Float32>(append_data);
		break;

	case TypeIndex::Float64:
		may_have_null = CHColumnToNumpyArray<Float64>(append_data);
		break;

	case TypeIndex::Int128:
		may_have_null = TransformColumn<Int128, Float64, RegularConvert>(append_data);
		break;

	case TypeIndex::Int256:
		may_have_null = TransformColumn<Int256, Float64, RegularConvert>(append_data);
		break;

	case TypeIndex::UInt128:
		may_have_null = TransformColumn<UInt128, Float64, RegularConvert>(append_data);
		break;

	case TypeIndex::UInt256:
		may_have_null = TransformColumn<UInt256, Float64, RegularConvert>(append_data);
		break;

	case TypeIndex::BFloat16:
		may_have_null = TransformColumn<BFloat16, Float32, RegularConvert>(append_data);
		break;

	case TypeIndex::Date:
		may_have_null = TransformColumn<UInt16, Int64, RegularConvert>(append_data);
		break;

	case TypeIndex::Date32:
		may_have_null = TransformColumn<Int32, Int64, RegularConvert>(append_data);
		break;

	case TypeIndex::DateTime:
		may_have_null = TransformColumn<UInt32, Int64, RegularConvert>(append_data);
		break;

	case TypeIndex::DateTime64:
		may_have_null = CHColumnDateTime64ToNumpyArray(append_data);
		break;

	case TypeIndex::Time:
		may_have_null = TransformColumn<Int32, PyObject *, TimeConvert>(append_data);
		break;

	case TypeIndex::Time64:
		may_have_null = TransformColumn<Decimal64, PyObject *, Time64Convert>(append_data);
		break;

	case TypeIndex::String:
		may_have_null = CHColumnStringToNumpyArray<ColumnString>(append_data);
		break;

	case TypeIndex::FixedString:
		may_have_null = CHColumnStringToNumpyArray<ColumnFixedString>(append_data);
		break;

	case TypeIndex::Enum8:
		may_have_null = TransformColumn<Int8, PyObject *, Enum8Convert>(append_data);
		break;

	case TypeIndex::Enum16:
		may_have_null = TransformColumn<Int16, PyObject *, Enum16Convert>(append_data);
		break;

	case TypeIndex::Decimal32:
		may_have_null = CHColumnDecimalToNumpyArray<Decimal32>(append_data, actual_type);
		break;

	case TypeIndex::Decimal64:
		may_have_null = CHColumnDecimalToNumpyArray<Decimal64>(append_data, actual_type);
		break;

	case TypeIndex::Decimal128:
		may_have_null = CHColumnDecimalToNumpyArray<Decimal128>(append_data, actual_type);
		break;

	case TypeIndex::Decimal256:
		may_have_null = CHColumnDecimalToNumpyArray<Decimal256>(append_data, actual_type);
		break;

	case TypeIndex::UUID:
		may_have_null = CHColumnUUIDToNumpyArray(append_data);
		break;

	case TypeIndex::Array:
		may_have_null = CHColumnArrayToNumpyArray(append_data, actual_type);
		break;

	case TypeIndex::Tuple:
		may_have_null = CHColumnTupleToNumpyArray(append_data, actual_type);
		break;

	case TypeIndex::Interval:
		{
			const auto * interval_type = typeid_cast<const DataTypeInterval *>(actual_type.get());
			if (interval_type && interval_type->getKind() == IntervalKind::Kind::Quarter)
			{
				may_have_null = CHColumnIntervalToNumpyArray(append_data);
			}
			else
			{
				may_have_null = CHColumnToNumpyArray<Int64>(append_data);
			}
		}
		break;

	case TypeIndex::Map:
		may_have_null = CHColumnMapToNumpyArray(append_data, actual_type);
		break;

    case TypeIndex::Object:
		may_have_null = CHColumnObjectToNumpyArray(append_data, actual_type);
		break;

	case TypeIndex::IPv4:
		may_have_null = CHColumnIPv4ToNumpyArray(append_data);
		break;

    case TypeIndex::IPv6:
		may_have_null = CHColumnIPv6ToNumpyArray(append_data);
		break;

	case TypeIndex::Variant:
		may_have_null = CHColumnVariantToNumpyArray(append_data, actual_type);
		break;

	case TypeIndex::Dynamic:
		may_have_null = CHColumnDynamicToNumpyArray(append_data, actual_type);
		break;

	/// Set types are used only in WHERE clauses for IN operations, not in actual data storage
	case TypeIndex::Set:
	/// JSONPaths is an internal type used only for JSON schema inference,
	case TypeIndex::JSONPaths:
	/// Deprecated type, should not appear in normal data processing
	case TypeIndex::ObjectDeprecated:
	/// Function types are not actual data types, should not appear here
	case TypeIndex::Function:
	/// Aggregate function types are not actual data types, should not appear here
	case TypeIndex::AggregateFunction:
	/// LowCardinality should be unwrapped before reaching this point
	case TypeIndex::LowCardinality:
	/// Nullable cannot contain another Nullable type, so this should not appear in nested conversion
	case TypeIndex::Nullable:
	/// QBit type is supported in newer versions of ClickHouse
	/// case TypeIndex::QBit:
	default:
		throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Unsupported type {}", data_array->type->getName());
	}

	if (may_have_null)
	{
		hava_null = true;
	}
}

void NumpyArray::append(
	const DB::IColumn & column,
	const DB::DataTypePtr & type,
	size_t index)
{
	chassert(data_array);
	chassert(!mask_array);

	auto * data_ptr = data_array->data;
	chassert(data_ptr);

	auto * dest_ptr = reinterpret_cast<py::object *>(data_ptr) + data_array->count;

	*dest_ptr = convertFieldToPython(column, type, index);

	data_array->count += 1;
}

py::object NumpyArray::toArray() const
{
	chassert(data_array);

	data_array->resize(data_array->count);
	if (!hava_null)
	{
		return std::move(data_array->array);
	}

	chassert(mask_array);

	mask_array->resize(mask_array->count);
	auto data_values = std::move(data_array->array);
	auto null_values = std::move(mask_array->array);

	auto masked_array = py::module::import("numpy.ma").attr("masked_array")(data_values, null_values);
	return masked_array;
}

} // namespace CHDB
