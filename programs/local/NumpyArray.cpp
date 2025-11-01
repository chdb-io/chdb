#include "NumpyArray.h"
#include "NumpyType.h"

#include <Processors/Chunk.h>
#include <base/defines.h>
#include <Columns/ColumnFixedSizeHelper.h>
#include <Columns/ColumnNullable.h>
#include <Columns/IColumn.h>
#include <DataTypes/DataTypeFactory.h>
#include <base/types.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int NOT_IMPLEMENTED;
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

	for (size_t i = 0; i < append_data.count; i++) {
		size_t offset = append_data.dest_offset + i;
		if (nullable_column && nullable_column->isNullAt(i)) {
			dest_ptr[offset] = CONVERT::template nullValue<NUMPYTYPE>(mask_ptr[offset]);
			has_null = has_null || mask_ptr[offset];
		} else {
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

	switch (data_array->type->getTypeId())
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
	case TypeIndex::Time64:
	case TypeIndex::String:
	case TypeIndex::FixedString:
	case TypeIndex::Enum8:
	case TypeIndex::Enum16:
	case TypeIndex::Decimal32:
	case TypeIndex::Decimal64:
	case TypeIndex::Decimal128:
	case TypeIndex::Decimal256:
	case TypeIndex::UUID:
	case TypeIndex::Array:
	case TypeIndex::Tuple:
	case TypeIndex::Set:
	case TypeIndex::Interval:
	case TypeIndex::Map:
    case TypeIndex::Object:
    case TypeIndex::IPv4:
    case TypeIndex::IPv6:
    case TypeIndex::JSONPaths:
    case TypeIndex::Variant:
    case TypeIndex::Dynamic:
		/// TODO
		break;

	case TypeIndex::ObjectDeprecated:  /// Deprecated type, should not appear in normal data processing
	case TypeIndex::Function:          /// Function types are not data types, should not appear here
	case TypeIndex::AggregateFunction: /// Aggregate function types are not data types, should not appear here
	case TypeIndex::LowCardinality:    /// LowCardinality should be unwrapped before reaching this point
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
