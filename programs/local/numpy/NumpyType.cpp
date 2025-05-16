#include "NumpyType.h"

#include <Common/StringUtils.h>
#include "DataTypes/DataTypeInterval.h"
#include "DataTypes/DataTypeObject.h"
#include "DataTypes/DataTypesNumber.h"
#include "DataTypes/DataTypeString.h"

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

DataTypePtr NumpyToDataType(const NumpyType & col_type)
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
		return std::make_shared<DataTypeObject>();
	case NumpyNullableType::TIMEDELTA:
		return std::make_shared<DataTypeInterval>();
	case NumpyNullableType::DATETIME_MS:
		return std::make_shared<DataTypeDateTime64>(3);
	case NumpyNullableType::DATETIME_NS:
		return std::make_shared<DataTypeDateTime64>(9);
	case NumpyNullableType::DATETIME_S:
		return std::make_shared<DataTypeDateTime64>(0)
	case NumpyNullableType::DATETIME_US:
		std::make_shared<DataTypeDateTime64>(6);
	default:
		throw Exception(ErrorCodes::LOGICAL_ERROR, "Unkonow numpy column type: {}", col_type);
	}
}

} // namespace CHDB
