#include "FieldToPython.h"
#include "PythonImporter.h"

#include <Core/TypeId.h>
#include <Core/UUID.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypeDateTime.h>
#include <DataTypes/DataTypeDateTime64.h>
#include <base/IPv4andIPv6.h>
#include <Common/Exception.h>
#include <Common/LocalDate.h>
#include <Common/LocalDateTime.h>
#include <Common/DateLUTImpl.h>
#include <Common/formatIPv6.h>
#include <Core/DecimalFunctions.h>
#include <base/types.h>

namespace CHDB
{

using namespace DB;
namespace py = pybind11;

namespace ErrorCodes
{
    extern const int NOT_IMPLEMENTED;
}

static py::object convertLocalDateToPython(const LocalDate & local_date, auto & import_cache, const Field & field)
{
    auto year = local_date.year();
    auto month = local_date.month();
    auto day = local_date.day();

    try
    {
        return import_cache.datetime.date()(year, month, day);
    }
    catch (py::error_already_set &)
    {
        return py::str(toString(field));
    }
}

py::object convertFieldToPython(
    const Field & field,
    const DB::DataTypePtr & type)
{
    chassert(type);

    auto filed_type = field.getType();
    if (filed_type == Field::Types::Null)
    {
        return py::none();
    }

    DataTypePtr actual_type = type;
    if (const auto * nullable_type = typeid_cast<const DataTypeNullable *>(type.get()))
    {
        actual_type = nullable_type->getNestedType();
    }

    auto & import_cache = PythonImporter::ImportCache();

	switch (actual_type->getTypeId())
	{
	case TypeIndex::Nothing:
		return py::none();

	case TypeIndex::Int8:
		return py::cast(field.safeGet<Int64>());

	case TypeIndex::UInt8:
        if (filed_type == Field::Types::Bool)
            return py::cast(field.safeGet<bool>());

		return py::cast(field.safeGet<UInt64>());

	case TypeIndex::Int16:
		return py::cast(field.safeGet<Int64>());

	case TypeIndex::UInt16:
		return py::cast(field.safeGet<UInt64>());

	case TypeIndex::Int32:
		return py::cast(field.safeGet<Int64>());

	case TypeIndex::UInt32:
		return py::cast(field.safeGet<UInt64>());

	case TypeIndex::Int64:
		return py::cast(field.safeGet<Int64>());

	case TypeIndex::UInt64:
		return py::cast(field.safeGet<UInt64>());

	case TypeIndex::Float32:
		return py::cast(field.safeGet<Float64>());

	case TypeIndex::Float64:
		return py::cast(field.safeGet<Float64>());

	case TypeIndex::Int128:
		return py::cast((double)field.safeGet<Int128>());

	case TypeIndex::Int256:
		return py::cast((double)field.safeGet<Int256>());

	case TypeIndex::UInt128:
		return py::cast((double)field.safeGet<UInt128>());

	case TypeIndex::UInt256:
		return py::cast((double)field.safeGet<UInt256>());

	case TypeIndex::BFloat16:
		return py::cast((double)field.safeGet<Float64>());

	case TypeIndex::Date:
        {
            auto days = field.safeGet<UInt64>();
            LocalDate local_date(static_cast<UInt16>(days));
            return convertLocalDateToPython(local_date, import_cache, field);
        }

    case TypeIndex::Date32:
        {
            auto days = field.safeGet<Int64>();
            LocalDate local_date(static_cast<Int32>(days));
            return convertLocalDateToPython(local_date, import_cache, field);
        }

    case TypeIndex::DateTime:
        {
            auto seconds = field.safeGet<UInt64>();

            const auto * datetime_type = typeid_cast<const DataTypeDateTime *>(type.get());
            const auto & time_zone = datetime_type ? datetime_type->getTimeZone() : DateLUT::instance("UTC");

            time_t timestamp = static_cast<time_t>(seconds);
            LocalDateTime local_dt(timestamp, time_zone);

            int year = local_dt.year();
            int month = local_dt.month();
            int day = local_dt.day();
            int hour = local_dt.hour();
            int minute = local_dt.minute();
            int second = local_dt.second();
            int microsecond = 0;

            try
            {
                py::object timestamp_object = import_cache.datetime.datetime()(
                    year, month, day, hour, minute, second, microsecond
                );

                const String & tz_name = time_zone.getTimeZone();
                auto tz_obj = import_cache.pytz.timezone()(tz_name);
                return tz_obj.attr("localize")(timestamp_object);
            }
            catch (py::error_already_set &)
            {
                return py::str(toString(field));
            }
        }

    case TypeIndex::DateTime64:
        {
            auto datetime64_field = field.safeGet<DecimalField<DateTime64>>();
            auto datetime64_value = datetime64_field.getValue();
            Int64 datetime64_ticks = datetime64_value.value;

            const auto * datetime64_type = typeid_cast<const DataTypeDateTime64 *>(type.get());
            const auto & time_zone = datetime64_type ? datetime64_type->getTimeZone() : DateLUT::instance("UTC");

            UInt32 scale = datetime64_field.getScale();
            Int64 scale_multiplier = DecimalUtils::scaleMultiplier<DateTime64::NativeType>(scale);

            auto seconds = static_cast<time_t>(datetime64_ticks / scale_multiplier);
            auto fractional = datetime64_ticks % scale_multiplier;

            LocalDateTime local_dt(seconds, time_zone);

            int year = local_dt.year();
            int month = local_dt.month();
            int day = local_dt.day();
            int hour = local_dt.hour();
            int minute = local_dt.minute();
            int second = local_dt.second();
            int microsecond = static_cast<int>((fractional * 1000000) / scale_multiplier);

            try
            {
                py::object timestamp_object = import_cache.datetime.datetime()(
                    year, month, day, hour, minute, second, microsecond
                );

                const String & tz_name = time_zone.getTimeZone();
                auto tz_obj = import_cache.pytz.timezone()(tz_name);
                return tz_obj.attr("localize")(timestamp_object);
            }
            catch (py::error_already_set &)
            {
                return py::str(toString(field));
            }
        }

    case TypeIndex::Time:
        {
            auto time_seconds = field.safeGet<Int64>();

            if (time_seconds < 0)
            {
                return py::str(toString(field));
            }

            /// Handle time overflow (should be within 24 hours)
            /// ClickHouse Time range is [-999:59:59, 999:59:59]
            time_seconds = time_seconds % 86400;

            int hour = static_cast<int>(time_seconds / 3600);
            int minute = static_cast<int>((time_seconds % 3600) / 60);
            int second = static_cast<int>(time_seconds % 60);
            int microsecond = 0;

            try
            {
                return import_cache.datetime.time()(hour, minute, second, microsecond);
            }
            catch (py::error_already_set &)
            {
                return py::str(toString(field));
            }
        }

    case TypeIndex::Time64:
        {
            auto time64_field = field.safeGet<DecimalField<Decimal64>>();
            auto time64_value = time64_field.getValue();
            Int64 time64_ticks = time64_value.value;

            if (time64_ticks < 0)
            {
                return py::str(toString(field));
            }

            UInt32 scale = time64_field.getScale();
            Int64 scale_multiplier = DecimalUtils::scaleMultiplier<Decimal64::NativeType>(scale);

            /// Convert to seconds and fractional part within a day
            Int64 total_seconds = time64_ticks / scale_multiplier;
            Int64 fractional = time64_ticks % scale_multiplier;

            /// Handle time overflow (should be within 24 hours)
            /// ClickHouse Time range is [-999:59:59, 999:59:59]
            total_seconds = total_seconds % 86400;

            int hour = static_cast<int>(total_seconds / 3600);
            int minute = static_cast<int>((total_seconds % 3600) / 60);
            int second = static_cast<int>(total_seconds % 60);
            int microsecond = static_cast<int>((fractional * 1000000) / scale_multiplier);

            try
            {
                return import_cache.datetime.time()(hour, minute, second, microsecond);
            }
            catch (py::error_already_set &)
            {
                return py::str(toString(field));
            }
        }

    case TypeIndex::String:
    case TypeIndex::FixedString:
        return py::cast(field.safeGet<String>());

    case TypeIndex::Enum8:
    case TypeIndex::Enum16:
        return py::cast(field.safeGet<Int64>());

    case TypeIndex::Decimal32:
        {
            auto decimal_field = field.safeGet<DecimalField<Decimal32>>();
            auto decimal_value = decimal_field.getValue();
            UInt32 scale = decimal_field.getScale();
            double result = DecimalUtils::convertTo<double>(decimal_value, scale);
            return py::cast(result);
        }

    case TypeIndex::Decimal64:
        {
            auto decimal_field = field.safeGet<DecimalField<Decimal64>>();
            auto decimal_value = decimal_field.getValue();
            UInt32 scale = decimal_field.getScale();
            double result = DecimalUtils::convertTo<double>(decimal_value, scale);
            return py::cast(result);
        }

    case TypeIndex::Decimal128:
        {
            auto decimal_field = field.safeGet<DecimalField<Decimal128>>();
            auto decimal_value = decimal_field.getValue();
            UInt32 scale = decimal_field.getScale();
            double result = DecimalUtils::convertTo<double>(decimal_value, scale);
            return py::cast(result);
        }

    case TypeIndex::Decimal256:
        {
            auto decimal_field = field.safeGet<DecimalField<Decimal256>>();
            auto decimal_value = decimal_field.getValue();
            UInt32 scale = decimal_field.getScale();
            double result = DecimalUtils::convertTo<double>(decimal_value, scale);
            return py::cast(result);
        }

    case TypeIndex::UUID:
		break;

	// case TypeIndex::Array:
	// 	may_have_null = CHColumnArrayToNumpyArray(append_data, actual_type);
	// 	break;

	// case TypeIndex::Tuple:
	// 	may_have_null = CHColumnTupleToNumpyArray(append_data, actual_type);
	// 	break;

	// case TypeIndex::Interval:
	// 	{
	// 		const auto * interval_type = typeid_cast<const DataTypeInterval *>(actual_type.get());
	// 		if (interval_type && interval_type->getKind() == IntervalKind::Kind::Quarter)
	// 		{
	// 			may_have_null = CHColumnIntervalToNumpyArray(append_data);
	// 		}
	// 		else
	// 		{
	// 			may_have_null = CHColumnToNumpyArray<Int64>(append_data);
	// 		}
	// 	}
	// 	break;

	// case TypeIndex::Map:
	// 	may_have_null = CHColumnMapToNumpyArray(append_data, actual_type);
	// 	break;

    // case TypeIndex::Object:
	// 	may_have_null = CHColumnObjectToNumpyArray(append_data, actual_type);
	// 	break;

	// case TypeIndex::IPv4:
	// 	may_have_null = CHColumnIPv4ToNumpyArray(append_data);
	// 	break;

    // case TypeIndex::IPv6:
	// 	may_have_null = CHColumnIPv6ToNumpyArray(append_data);
	// 	break;

	// case TypeIndex::Variant:
	// 	may_have_null = CHColumnVariantToNumpyArray(append_data, actual_type);
	// 	break;

	// case TypeIndex::Dynamic:
	// 	may_have_null = CHColumnDynamicToNumpyArray(append_data, actual_type);
	// 	break;

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
		throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Unsupported type {}", type->getName());
	}
}

} // namespace CHDB
