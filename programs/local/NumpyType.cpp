#include "NumpyType.h"
#include "PythonImporter.h"

#include <DataTypes/DataTypeDateTime64.h>
#include <DataTypes/DataTypeInterval.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypeObject.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypeTime64.h>

using namespace DB;

namespace DB
{

namespace ErrorCodes
{
	extern const int LOGICAL_ERROR;
    extern const int NOT_IMPLEMENTED;
}

}

namespace CHDB
{

static bool IsDateTime(NumpyNullableType type)
{
	switch (type) {
	case NumpyNullableType::DATETIME_NS:
	case NumpyNullableType::DATETIME_S:
	case NumpyNullableType::DATETIME_MS:
	case NumpyNullableType::DATETIME_US:
		return true;
	default:
		return false;
	};
}

String NumpyType::toString() const
{
	std::string type_str;
    switch (type) {
    case NumpyNullableType::BOOL:
		type_str = "BOOL";
		break;
    case NumpyNullableType::INT_8:
		type_str = "INT8";
		break;
    case NumpyNullableType::UINT_8:
		type_str = "UINT8";
		break;
	case NumpyNullableType::INT_16:
        type_str = "INT16";
        break;
    case NumpyNullableType::UINT_16:
        type_str = "UINT16";
        break;
    case NumpyNullableType::INT_32:
        type_str = "INT32";
        break;
    case NumpyNullableType::UINT_32:
        type_str = "UINT32";
        break;
    case NumpyNullableType::INT_64:
        type_str = "INT64";
        break;
    case NumpyNullableType::UINT_64:
        type_str = "UINT64";
        break;
    case NumpyNullableType::FLOAT_16:
        type_str = "FLOAT16";
        break;
    case NumpyNullableType::FLOAT_32:
        type_str = "FLOAT32";
        break;
    case NumpyNullableType::FLOAT_64:
		type_str = "FLOAT64";
		break;
    case NumpyNullableType::OBJECT:
	  	type_str = "OBJECT";
		break;
    case NumpyNullableType::STRING:
		type_str = "STRING";
		break;
    case NumpyNullableType::DATETIME_NS:
		type_str = "DATETIME_NS";
		break;
    case NumpyNullableType::DATETIME_US:
		type_str = "DATETIME_US)";
		break;
    case NumpyNullableType::DATETIME_MS:
		type_str = "DATETIME_MS";
		break;
    case NumpyNullableType::DATETIME_S:
		type_str = "DATETIME_S";
		break;
    case NumpyNullableType::TIMEDELTA:
		type_str = "TIMEDELTA";
		break;
    case NumpyNullableType::CATEGORY:
		type_str = "CATEGORY";
		break;
    }

    if (has_timezone && IsDateTime(type)) {
        type_str += " WITH TIMEZONE";
    }
    return type_str;
}

static NumpyNullableType ConvertNumpyTypeInternal(const String & col_type_str)
{
	static const std::map<String, NumpyNullableType> type_map =
	{
		{"bool", NumpyNullableType::BOOL},
        {"boolean", NumpyNullableType::BOOL},
        {"uint8", NumpyNullableType::UINT_8},
        {"UInt8", NumpyNullableType::UINT_8},
        {"uint16", NumpyNullableType::UINT_16},
        {"UInt16", NumpyNullableType::UINT_16},
        {"uint32", NumpyNullableType::UINT_32},
        {"UInt32", NumpyNullableType::UINT_32},
        {"uint64", NumpyNullableType::UINT_64},
        {"UInt64", NumpyNullableType::UINT_64},
        {"int8", NumpyNullableType::INT_8},
        {"Int8", NumpyNullableType::INT_8},
        {"int16", NumpyNullableType::INT_16},
        {"Int16", NumpyNullableType::INT_16},
        {"int32", NumpyNullableType::INT_32},
        {"Int32", NumpyNullableType::INT_32},
        {"int64", NumpyNullableType::INT_64},
        {"Int64", NumpyNullableType::INT_64},
        {"float16", NumpyNullableType::FLOAT_16},
        {"Float16", NumpyNullableType::FLOAT_16},
        {"float32", NumpyNullableType::FLOAT_32},
        {"Float32", NumpyNullableType::FLOAT_32},
        {"float64", NumpyNullableType::FLOAT_64},
        {"Float64", NumpyNullableType::FLOAT_64},
        {"string", NumpyNullableType::STRING},
        {"object", NumpyNullableType::OBJECT},
        {"timedelta64[ns]", NumpyNullableType::TIMEDELTA},
        {"category", NumpyNullableType::CATEGORY},
    };

	auto it = type_map.find(col_type_str);
    if (it != type_map.end())
        return it->second;

	if (startsWith(col_type_str, "datetime64[ns"))
		return NumpyNullableType::DATETIME_NS;
	if (startsWith(col_type_str, "datetime64[us"))
		return NumpyNullableType::DATETIME_US;
	if (startsWith(col_type_str, "datetime64[ms"))
		return NumpyNullableType::DATETIME_MS;
	if (startsWith(col_type_str, "datetime64[s"))
		return NumpyNullableType::DATETIME_S;

	/// Legacy datetime type indicators
	if (startsWith(col_type_str, "<M8[ns"))
		return NumpyNullableType::DATETIME_NS;
	if (startsWith(col_type_str, "<M8[s"))
		return NumpyNullableType::DATETIME_S;
	if (startsWith(col_type_str, "<M8[us"))
		return NumpyNullableType::DATETIME_US;
	if (startsWith(col_type_str, "<M8[ms"))
		return NumpyNullableType::DATETIME_MS;

	throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Unsupported data type: {}", col_type_str);
}

NumpyType ConvertNumpyType(const py::handle & col_type)
{
	auto col_type_str = String(py::str(col_type));
	NumpyType numpy_type;

	numpy_type.type = ConvertNumpyTypeInternal(col_type_str);
	if (IsDateTime(numpy_type.type)) {
		if (hasattr(col_type, "tz")) {
			/// The datetime has timezone information.
			numpy_type.has_timezone = true;
		}
	}
	return numpy_type;
}

std::shared_ptr<IDataType> NumpyToDataType(const NumpyType & col_type)
{
	switch (col_type.type)
	{
	case NumpyNullableType::BOOL:
		return std::make_shared<DataTypeUInt8>();
	case NumpyNullableType::INT_8:
		return std::make_shared<DataTypeInt8>();
	case NumpyNullableType::UINT_8:
		return std::make_shared<DataTypeUInt8>();
	case NumpyNullableType::INT_16:
		return std::make_shared<DataTypeInt16>();
	case NumpyNullableType::UINT_16:
		return std::make_shared<DataTypeUInt16>();
	case NumpyNullableType::INT_32:
		return std::make_shared<DataTypeInt32>();
	case NumpyNullableType::UINT_32:
		return std::make_shared<DataTypeUInt32>();
	case NumpyNullableType::INT_64:
		return std::make_shared<DataTypeInt64>();
	case NumpyNullableType::UINT_64:
		return std::make_shared<DataTypeUInt64>();
	case NumpyNullableType::FLOAT_16:
		return std::make_shared<DataTypeFloat32>();
	case NumpyNullableType::FLOAT_32:
		return std::make_shared<DataTypeFloat32>();
	case NumpyNullableType::FLOAT_64:
		return std::make_shared<DataTypeFloat64>();
	case NumpyNullableType::STRING:
		return std::make_shared<DataTypeString>();
	case NumpyNullableType::OBJECT:
		return std::make_shared<DataTypeObject>(DataTypeObject::SchemaFormat::JSON);
	case NumpyNullableType::DATETIME_MS:
		return std::make_shared<DataTypeDateTime64>(3);
	case NumpyNullableType::DATETIME_NS:
		return std::make_shared<DataTypeDateTime64>(9);
	case NumpyNullableType::DATETIME_S:
		return std::make_shared<DataTypeDateTime64>(0);
	case NumpyNullableType::DATETIME_US:
		std::make_shared<DataTypeDateTime64>(6);
	case NumpyNullableType::TIMEDELTA:
		/// return std::make_shared<DataTypeInterval>();
	case NumpyNullableType::CATEGORY:
	default:
		throw Exception(ErrorCodes::LOGICAL_ERROR, "Unkonow numpy column type: {}", col_type.toString());
	}
}

String DataTypeToNumpyTypeStr(const std::shared_ptr<const IDataType> & data_type)
{
    if (!data_type)
        return "object";

    TypeIndex type_id = data_type->getTypeId();
    switch (type_id)
    {
    case TypeIndex::Int8:
        return "int8";
    case TypeIndex::UInt8:
        /// Special case: UInt8 could be Bool type, need to check getName()
        {
            auto is_bool = isBool(data_type);
            return is_bool ? "bool" : "uint8";
        }
    case TypeIndex::Int16:
        return "int16";
    case TypeIndex::UInt16:
        return "uint16";
    case TypeIndex::Int32:
        return "int32";
    case TypeIndex::UInt32:
        return "uint32";
    case TypeIndex::Int64:
        return "int64";
    case TypeIndex::UInt64:
        return "uint64";
    case TypeIndex::BFloat16:
    case TypeIndex::Float32:
        return "float32";
    case TypeIndex::Int256:
    case TypeIndex::UInt256:
    case TypeIndex::Int128:
    case TypeIndex::UInt128:
    case TypeIndex::Float64:
        return "float64";
    case TypeIndex::String:
    case TypeIndex::FixedString:
        return "object";
    case TypeIndex::DateTime:
        return "datetime64[s]";
    case TypeIndex::DateTime64:
        {
            if (const auto * dt64 = typeid_cast<const DataTypeDateTime64 *>(data_type.get()))
            {
                UInt32 scale = dt64->getScale();
                if (scale == 0)
                    return "datetime64[s]";
                else if (scale == 3)
                    return "datetime64[ms]";
                else if (scale == 6)
                    return "datetime64[us]";
                else if (scale == 9)
                    return "datetime64[ns]";
                else
                    return "datetime64[ns]";
            }
            return "datetime64[ns]";
        }
    case TypeIndex::Date:
    case TypeIndex::Date32:
        return "datetime64[D]";
    case TypeIndex::Time:
        return "timedelta64[s]";
    case TypeIndex::Time64:
        {
            if (const auto * time64 = typeid_cast<const DataTypeTime64 *>(data_type.get()))
            {
                UInt32 scale = time64->getScale();
                if (scale == 0)
                    return "timedelta64[s]";
                else if (scale == 3)
                    return "timedelta64[ms]";
                else if (scale == 6)
                    return "timedelta64[us]";
                else if (scale == 9)
                    return "timedelta64[ns]";
                else
                    return "timedelta64[ns]";
            }
            return "timedelta64[ns]";
        }
    case TypeIndex::Interval:
        {
            if (const auto * interval = typeid_cast<const DataTypeInterval *>(data_type.get()))
            {
                IntervalKind kind = interval->getKind();
                switch (kind.kind)
                {
                    case IntervalKind::Kind::Nanosecond:
                        return "timedelta64[ns]";
                    case IntervalKind::Kind::Microsecond:
                        return "timedelta64[us]";
                    case IntervalKind::Kind::Millisecond:
                        return "timedelta64[ms]";
                    case IntervalKind::Kind::Second:
                        return "timedelta64[s]";
                    case IntervalKind::Kind::Minute:
                        return "timedelta64[m]";
                    case IntervalKind::Kind::Hour:
                        return "timedelta64[h]";
                    case IntervalKind::Kind::Day:
                        return "timedelta64[D]";
                    case IntervalKind::Kind::Week:
                        return "timedelta64[W]";
                    case IntervalKind::Kind::Month:
                        return "timedelta64[M]";
                    case IntervalKind::Kind::Quarter:
                        /// numpy doesn't have quarter type, use int64
                        return "int64";
                    case IntervalKind::Kind::Year:
                        return "timedelta64[Y]";
                    default:
                        return "timedelta64[s]";
                }
            }
            return "timedelta64[s]";
        }

    case TypeIndex::UUID:
    case TypeIndex::IPv4:
    case TypeIndex::IPv6:
        return "object";
    case TypeIndex::Decimal32:
    case TypeIndex::Decimal64:
    case TypeIndex::Decimal128:
    case TypeIndex::Decimal256:
        return "float64";
    case TypeIndex::Array:
    case TypeIndex::Tuple:
    case TypeIndex::Map:
    case TypeIndex::Set:
    case TypeIndex::Dynamic:
    case TypeIndex::Variant:
    case TypeIndex::Object:
        return "object";
    case TypeIndex::Enum8:
        return "int8";
    case TypeIndex::Enum16:
        return "int16";
    case TypeIndex::Nullable:
        {
            if (const auto * nullable = typeid_cast<const DataTypeNullable *>(data_type.get()))
            {
                return DataTypeToNumpyTypeStr(nullable->getNestedType());
            }
            return "object";
        }
    default:
        throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Unsupported type {}", data_type->getName());
    }
}

py::object ConvertNumpyDtype(const py::handle & numpy_array)
{
    chassert(py::gil_check());

	auto & import_cache = PythonImporter::ImportCache();

	auto dtype = numpy_array.attr("dtype");
	if (!py::isinstance(numpy_array, import_cache.numpy.ma.masked_array()))
    {
		return dtype;
	}

	auto numpy_type = ConvertNumpyType(dtype);
	switch (numpy_type.type)
    {
    case NumpyNullableType::BOOL:
        return import_cache.pandas.BooleanDtype()();
    case NumpyNullableType::UINT_8:
        return import_cache.pandas.UInt8Dtype()();
    case NumpyNullableType::UINT_16:
        return import_cache.pandas.UInt16Dtype()();
    case NumpyNullableType::UINT_32:
        return import_cache.pandas.UInt32Dtype()();
    case NumpyNullableType::UINT_64:
        return import_cache.pandas.UInt64Dtype()();
    case NumpyNullableType::INT_8:
        return import_cache.pandas.Int8Dtype()();
    case NumpyNullableType::INT_16:
        return import_cache.pandas.Int16Dtype()();
    case NumpyNullableType::INT_32:
        return import_cache.pandas.Int32Dtype()();
    case NumpyNullableType::INT_64:
        return import_cache.pandas.Int64Dtype()();
    case NumpyNullableType::FLOAT_32:
        return import_cache.pandas.Float32Dtype()();
    case NumpyNullableType::FLOAT_64:
        return import_cache.pandas.Float64Dtype()();
    case NumpyNullableType::FLOAT_16:
    default:
        return dtype;
	}
}

} // namespace CHDB
