#include "NumpyArray.h"
#include "NumpyType.h"
#include "PythonImporter.h"

#include <Processors/Chunk.h>
#include <base/defines.h>
#include <Columns/ColumnFixedSizeHelper.h>
#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnNullable.h>
#include <Columns/ColumnString.h>
#include <Columns/IColumn.h>
#include <DataTypes/DataTypeFactory.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypesDecimal.h>
#include <base/types.h>
#include <base/UUID.h>
#include <base/IPv4andIPv6.h>
#include <IO/WriteHelpers.h>
#include <Common/formatIPv6.h>
#include <pybind11/pytypes.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int NOT_IMPLEMENTED;
	extern const int LOGICAL_ERROR
}

}

using namespace DB;

namespace CHDB
{

struct NumpyAppendData
{
public:
	explicit NumpyAppendData(const IColumn & column)
		: column(column)
	{
	}

	const IColumn & column;

	size_t count;
	size_t dest_offset;
	UInt8 * target_data;
	bool * target_mask;
};

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

	for (size_t i = 0; i < append_data.count; i++)
	{
		size_t offset = append_data.dest_offset + i;
		if (nullable_column && nullable_column->isNullAt(i))
		{
			dest_ptr[offset] = CONVERT::template nullValue<NUMPYTYPE>(mask_ptr[offset]);
			has_null = has_null || mask_ptr[offset];
		}
		else
		{
			dest_ptr[offset] = CONVERT::template convertValue<CHTYPE, NUMPYTYPE>(src_ptr[i], append_data);
			mask_ptr[offset] = false;
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

	auto scale_multiplier = decimal_type->getScaleMultiplier();
	double scale_multiplier_double = static_cast<double>(scale_multiplier.value);

	auto * dest_ptr = reinterpret_cast<double *>(append_data.target_data);
	auto * mask_ptr = append_data.target_mask;

	for (size_t i = 0; i < append_data.count; i++)
	{
		size_t offset = append_data.dest_offset + i;
		if (nullable_column && nullable_column->isNullAt(i))
		{
			/// Set to 0.0 for null values
			dest_ptr[offset] = 0.0;
			mask_ptr[offset] = true;
			has_null = true;
		}
		else
		{
			/// Convert decimal integer value to actual decimal by dividing by scale multiplier
			auto decimal_value = decimal_column->getElement(i);
			dest_ptr[offset] = static_cast<double>(decimal_value.value) / scale_multiplier_double;
			mask_ptr[offset] = false;
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

	for (size_t i = 0; i < append_data.count; i++)
	{
		size_t offset = append_data.dest_offset + i;
		if (nullable_column && nullable_column->isNullAt(i))
		{
			Py_INCREF(Py_None);
			dest_ptr[offset] = Py_None;
			has_null = true;
			mask_ptr[offset] = true;
		}
		else
		{
			/// Convert UUID to Python uuid.UUID object
			UUID uuid_value = uuid_column->getElement(i);
			const auto formatted_uuid = formatUUID(uuid_value);
			const char * uuid_str = formatted_uuid.data();
			const size_t uuid_str_len = formatted_uuid.size();

			/// Create Python uuid.UUID object
			auto & import_cache = PythonImporter::ImportCache();
			py::handle uuid_handle = import_cache.uuid.UUID()(String(uuid_str, uuid_str_len)).release();
			dest_ptr[offset] = uuid_handle.ptr();
			mask_ptr[offset] = false;
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

	for (size_t i = 0; i < append_data.count; i++)
	{
		size_t offset = append_data.dest_offset + i;
		if (nullable_column && nullable_column->isNullAt(i))
		{
			Py_INCREF(Py_None);
			dest_ptr[offset] = Py_None;
			has_null = true;
		}
		else
		{
			/// Convert IPv4 to Python ipaddress.IPv4Address object
			IPv4 ipv4_value = ipv4_column->getElement(i);

			char ipv4_str[IPV4_MAX_TEXT_LENGTH];
			char * ptr = ipv4_str;
			formatIPv4(reinterpret_cast<const unsigned char*>(&ipv4_value), ptr);
			const size_t ipv4_str_len = ptr - ipv4_str;

			/// Create Python ipaddress.IPv4Address object
			auto & import_cache = PythonImporter::ImportCache();
			py::handle ipv4_handle = import_cache.ipaddress.ipv4_address()(String(ipv4_str, ipv4_str_len)).release();
			dest_ptr[offset] = ipv4_handle.ptr();
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

	for (size_t i = 0; i < append_data.count; i++)
	{
		size_t offset = append_data.dest_offset + i;
		if (nullable_column && nullable_column->isNullAt(i))
		{
			Py_INCREF(Py_None);
			dest_ptr[offset] = Py_None;
			has_null = true;
		}
		else
		{
			/// Convert IPv6 to Python ipaddress.IPv6Address object
			IPv6 ipv6_value = ipv6_column->getElement(i);

			/// Use ClickHouse's built-in IPv6 formatting function
			char ipv6_str[IPV6_MAX_TEXT_LENGTH];
			char * ptr = ipv6_str;
			formatIPv6(reinterpret_cast<const unsigned char*>(&ipv6_value), ptr);
			const size_t ipv6_str_len = ptr - ipv6_str;

			/// Create Python ipaddress.IPv6Address object
			auto & import_cache = PythonImporter::ImportCache();
			py::handle ipv6_handle = import_cache.ipaddress.ipv6_address()(String(ipv6_str, ipv6_str_len)).release();
			dest_ptr[offset] = ipv6_handle.ptr();
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

	for (size_t i = 0; i < append_data.count; i++)
	{
		size_t offset = append_data.dest_offset + i;
		if (nullable_column && nullable_column->isNullAt(i))
		{
			Py_INCREF(Py_None);
			dest_ptr[offset] = Py_None;
		}
		else
		{
			StringRef str_ref = string_column->getDataAt(i);
			auto * str_ptr = const_cast<char *>(str_ref.data);
			auto str_size = str_ref.size;
			dest_ptr[offset] = PyUnicode_FromStringAndSize(str_ptr, str_size);
		}
	}

	return has_null;
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
	mask_array = std::make_unique<InternalNumpyArray>(DataTypeFactory::instance().get("Bool"));
}

void NumpyArray::init(size_t capacity)
{
	data_array->init(capacity);
	mask_array->init(capacity);
}

void NumpyArray::resize(size_t capacity)
{
	data_array->resize(capacity);
	mask_array->resize(capacity);
}

void NumpyArray::append(const ColumnPtr & column)
{
	chassert(data_array);
	chassert(mask_array);

	auto * data_ptr = data_array->data;
	auto * mask_ptr = reinterpret_cast<bool *>(mask_array->data);
	chassert(data_ptr);
	chassert(mask_ptr);
	chassert(column->getDataType() == data_array->type->getColumnType());

	size_t size = column->size();
	data_array->count += size;
	mask_array->count += size;
	bool may_have_null = false;

	NumpyAppendData append_data(*column);
	append_data.count = size;
	append_data.target_data = data_ptr;
	append_data.target_mask = mask_ptr;
	append_data.dest_offset = data_array->count - size;

	/// For nullable types, we need to get the nested type
	DataTypePtr actual_type = data_array->type;
	if (const auto * nullable_type = typeid_cast<const DataTypeNullable *>(data_array->type.get()))
	{
		actual_type = nullable_type->getNestedType();
	}

	switch (actual_type->getTypeId())
	{
	case TypeIndex::Int8:
		may_have_null = CHColumnToNumpyArray<Int8>(append_data);
		break;
	case TypeIndex::UInt8:
		{
			auto is_bool = isBool(data_array->type);
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
		may_have_null = CHColumnToNumpyArray<Int64>(append_data);
		break;
	case TypeIndex::Time:
		may_have_null = TransformColumn<Int32, Int64, RegularConvert>(append_data);
		break;
	case TypeIndex::Time64:
		may_have_null = CHColumnToNumpyArray<Int64>(append_data);
		break;
	case TypeIndex::String:
		may_have_null = CHColumnStringToNumpyArray<ColumnString>(append_data);
		break;
	case TypeIndex::FixedString:
		may_have_null = CHColumnStringToNumpyArray<ColumnFixedString>(append_data);
		break;
	case TypeIndex::Enum8:
		may_have_null = CHColumnToNumpyArray<Int8>(append_data);
		break;
	case TypeIndex::Enum16:
		may_have_null = CHColumnToNumpyArray<Int16>(append_data);
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
	case TypeIndex::Tuple:
	case TypeIndex::Set:
	case TypeIndex::Interval:
		may_have_null = CHColumnToNumpyArray<Int64>(append_data);
		break;
	case TypeIndex::Map:
    case TypeIndex::Object:
    case TypeIndex::IPv4:
		may_have_null = CHColumnIPv4ToNumpyArray(append_data);
		break;
    case TypeIndex::IPv6:
		may_have_null = CHColumnIPv6ToNumpyArray(append_data);
		break;
    case TypeIndex::JSONPaths:
    case TypeIndex::Variant:
    case TypeIndex::Dynamic:
		/// TODO
		break;

	/// Deprecated type, should not appear in normal data processing
	case TypeIndex::ObjectDeprecated:
	/// Function types are not data types, should not appear here
	case TypeIndex::Function:
	/// Aggregate function types are not data types, should not appear here
	case TypeIndex::AggregateFunction:
	/// LowCardinality should be unwrapped before reaching this point
	case TypeIndex::LowCardinality:
	default:
		throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Unsupported type {}", data_array->type->getName());
	}

	if (may_have_null)
	{
		hava_null = true;
	}
}

py::object NumpyArray::toArray() const
{
	chassert(data_array && mask_array);

	data_array->resize(data_array->count);
	if (!hava_null)
	{
		return std::move(data_array->array);
	}

	mask_array->resize(mask_array->count);
	auto data_values = std::move(data_array->array);
	auto null_values = std::move(mask_array->array);

	auto masked_array = py::module::import("numpy.ma").attr("masked_array")(data_values, null_values);
	return masked_array;
}

} // namespace CHDB
