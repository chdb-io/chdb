#include "FieldToPython.h"
#include "PythonImporter.h"
#include "ObjectToPython.h"

#include <Core/DecimalComparison.h>
#include <Columns/ColumnArray.h>
#include <Columns/ColumnTuple.h>
#include <Columns/ColumnMap.h>
#include <Columns/ColumnDynamic.h>
#include <Columns/ColumnObject.h>
#include <DataTypes/DataTypeLowCardinality.h>
#include <IO/ReadBufferFromMemory.h>
#include <DataTypes/Serializations/SerializationInfo.h>
#include <DataTypes/DataTypesBinaryEncoding.h>
#include <Formats/FormatSettings.h>
#include <Core/UUID.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypeDateTime.h>
#include <DataTypes/DataTypeDateTime64.h>
#include <DataTypes/DataTypeEnum.h>
#include <DataTypes/DataTypeInterval.h>
#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypeTuple.h>
#include <DataTypes/DataTypeMap.h>
#include <DataTypes/DataTypeVariant.h>
#include <DataTypes/DataTypeDynamic.h>
#include <DataTypes/DataTypeObject.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypeDate.h>
#include <DataTypes/DataTypesNumber.h>
#include <base/IPv4andIPv6.h>
#include <Common/Exception.h>
#include <Common/LocalDate.h>
#include <Common/LocalDateTime.h>
#include <Common/DateLUTImpl.h>
#include <Common/formatIPv6.h>
#include <Core/DecimalFunctions.h>
#include <IO/WriteHelpers.h>
#include <base/types.h>

namespace DB
{

namespace ErrorCodes
{
extern const int NOT_IMPLEMENTED;
extern const int LOGICAL_ERROR;
}

}

namespace CHDB
{

using namespace DB;

py::object convertTimeFieldToPython(const Field & field)
{
    auto & import_cache = PythonImporter::ImportCache();
    auto time_seconds = field.safeGet<Int64>();

    if (time_seconds < 0)
    {
        return py::str(std::to_string(time_seconds));
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
        return py::str(field.dump());
    }
}

py::object convertTime64FieldToPython(const Field & field)
{
    auto & import_cache = PythonImporter::ImportCache();
    auto time64_field = field.safeGet<DecimalField<Decimal64>>();
    auto time64_value = time64_field.getValue();
    Int64 time64_ticks = time64_value.value;
    UInt32 scale = time64_field.getScale();

    if (time64_ticks < 0)
    {
        Int64 scale_multiplier = DecimalUtils::scaleMultiplier<Decimal64::NativeType>(scale);
        Int64 abs_ticks = -time64_ticks;
        Int64 integer_part = abs_ticks / scale_multiplier;
        Int64 fractional_part = abs_ticks % scale_multiplier;

        std::string result = "-" + std::to_string(integer_part);
        if (fractional_part > 0)
        {
            std::string frac_str = std::to_string(fractional_part);
            while (frac_str.length() < scale)
                frac_str = "0" + frac_str;
            while (!frac_str.empty() && frac_str.back() == '0')
                frac_str.pop_back();
            if (!frac_str.empty())
                result += "." + frac_str;
        }
        return py::str(result);
    }
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
        return py::str(field.dump());
    }
}

static bool canTypeBeUsedAsDictKey(const DataTypePtr & type)
{
    DataTypePtr actual_type = removeLowCardinalityAndNullable(type);

    switch (actual_type->getTypeId())
	{
	case TypeIndex::Nothing:
	case TypeIndex::Int8:
	case TypeIndex::UInt8:
	case TypeIndex::Int16:
	case TypeIndex::UInt16:
	case TypeIndex::Int32:
	case TypeIndex::UInt32:
	case TypeIndex::Int64:
    case TypeIndex::UInt64:
	case TypeIndex::Float32:
	case TypeIndex::Float64:
	case TypeIndex::Int128:
	case TypeIndex::Int256:
	case TypeIndex::UInt128:
	case TypeIndex::UInt256:
	case TypeIndex::BFloat16:
	case TypeIndex::Date:
	case TypeIndex::Date32:
	case TypeIndex::DateTime:
	case TypeIndex::DateTime64:
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
    case TypeIndex::Interval:
    case TypeIndex::IPv4:
	case TypeIndex::IPv6:
        return true;

	case TypeIndex::Array:
	case TypeIndex::Tuple:
	case TypeIndex::Map:
	case TypeIndex::Object:
	case TypeIndex::Dynamic:
        return false;

	case TypeIndex::Variant:
		{
			const auto * variant_type = typeid_cast<const DataTypeVariant *>(type.get());
            chassert(variant_type);

			const auto & variants = variant_type->getVariants();
			for (const auto & variant : variants)
			{
				if (!canTypeBeUsedAsDictKey(variant))
					return false;
			}
			return true;
		}

	case TypeIndex::Set:
	case TypeIndex::JSONPaths:
	case TypeIndex::Function:
	case TypeIndex::AggregateFunction:
	case TypeIndex::LowCardinality:
	case TypeIndex::Nullable:
	default:
		return false;
	}
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
        return py::str(field.dump());
    }
}

py::object convertFieldToPython(
    const IColumn & column,
    const DataTypePtr & type,
    size_t index)
{
    if (column.isNullAt(index))
    {
        return py::none();
    }

    DataTypePtr actual_type = removeLowCardinalityAndNullable(type);

    auto & import_cache = PythonImporter::ImportCache();

	switch (actual_type->getTypeId())
	{
	case TypeIndex::Nothing:
		return py::none();

	case TypeIndex::Int8:
        {
            auto field = column[index];
            return py::cast(field.safeGet<Int64>());
        }

	case TypeIndex::UInt8:
        {
            auto field = column[index];
            auto is_bool = isBool(actual_type);
            if (is_bool)
            {
                bool val = field.safeGet<bool>();
                return py::cast(val);
            }

            return py::cast(field.safeGet<UInt64>());
        }

	case TypeIndex::Int16:
        {
            auto field = column[index];
            return py::cast(field.safeGet<Int64>());
        }

	case TypeIndex::UInt16:
        {
            auto field = column[index];
            return py::cast(field.safeGet<UInt64>());
        }

	case TypeIndex::Int32:
        {
            auto field = column[index];
            return py::cast(field.safeGet<Int64>());
        }

	case TypeIndex::UInt32:
        {
            auto field = column[index];
            return py::cast(field.safeGet<UInt64>());
        }

	case TypeIndex::Int64:
        {
            auto field = column[index];
            return py::cast(field.safeGet<Int64>());
        }

	case TypeIndex::UInt64:
        {
            auto field = column[index];
            return py::cast(field.safeGet<UInt64>());
        }

	case TypeIndex::Float32:
        {
            auto field = column[index];
            return py::cast(field.safeGet<Float64>());
        }

	case TypeIndex::Float64:
        {
            auto field = column[index];
            return py::cast(field.safeGet<Float64>());
        }

	case TypeIndex::Int128:
        {
            auto field = column[index];
            return py::cast((double)field.safeGet<Int128>());
        }

	case TypeIndex::Int256:
        {
            auto field = column[index];
            return py::cast((double)field.safeGet<Int256>());
        }

	case TypeIndex::UInt128:
        {
            auto field = column[index];
            return py::cast((double)field.safeGet<UInt128>());
        }

	case TypeIndex::UInt256:
        {
            auto field = column[index];
            return py::cast((double)field.safeGet<UInt256>());
        }

	case TypeIndex::BFloat16:
        {
            auto field = column[index];
            return py::cast((double)field.safeGet<Float64>());
        }

	case TypeIndex::Date:
        {
            auto field = column[index];
            auto days = field.safeGet<UInt64>();
            LocalDate local_date(DayNum(static_cast<UInt16>(days)));
            return convertLocalDateToPython(local_date, import_cache, field);
        }

    case TypeIndex::Date32:
        {
            auto field = column[index];
            auto days = field.safeGet<Int64>();
            LocalDate local_date(ExtendedDayNum(static_cast<Int32>(days)));
            return convertLocalDateToPython(local_date, import_cache, field);
        }

    case TypeIndex::DateTime:
        {
            auto field = column[index];
            auto seconds = field.safeGet<UInt64>();

            const auto * datetime_type = typeid_cast<const DataTypeDateTime *>(actual_type.get());
            const auto & utc_time_zone = DateLUT::instance("UTC");
            const auto & time_zone = datetime_type ? datetime_type->getTimeZone() : utc_time_zone;

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
                return py::str(field.dump());
            }
        }

    case TypeIndex::DateTime64:
        {
            auto field = column[index];
            auto datetime64_field = field.safeGet<DecimalField<DateTime64>>();
            auto datetime64_value = datetime64_field.getValue();
            Int64 datetime64_ticks = datetime64_value.value;

            const auto * datetime64_type = typeid_cast<const DataTypeDateTime64 *>(actual_type.get());
            const auto & utc_time_zone = DateLUT::instance("UTC");
            const auto & time_zone = datetime64_type ? datetime64_type->getTimeZone() : utc_time_zone;

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
                return py::str(field.dump());
            }
        }

    case TypeIndex::Time:
        {
            auto field = column[index];
            return convertTimeFieldToPython(field);
        }

    case TypeIndex::Time64:
        {
            auto field = column[index];
            return convertTime64FieldToPython(field);
        }

    case TypeIndex::String:
    case TypeIndex::FixedString:
        {
            auto field = column[index];
            return py::cast(field.safeGet<String>());
        }

    case TypeIndex::Enum8:
        {
            auto field = column[index];
            try
            {
                const auto & enum_type = typeid_cast<const DataTypeEnum8 &>(*type);
                auto it = enum_type.findByValue(static_cast<Int8>(field.safeGet<Int64>()));
                String enum_name(it->second.data(), it->second.size());
                return py::cast(enum_name);
            }
            catch (...)
            {
                return py::cast(field.dump());
            }
        }

    case TypeIndex::Enum16:
        {
            auto field = column[index];
            try
            {
                const auto & enum_type = typeid_cast<const DataTypeEnum16 &>(*type);
                auto it = enum_type.findByValue(static_cast<Int16>(field.safeGet<Int64>()));
                String enum_name(it->second.data(), it->second.size());
                return py::cast(enum_name);
            }
            catch (...)
            {
                return py::cast(field.dump());
            }
        }

    case TypeIndex::Decimal32:
        {
            auto field = column[index];
            auto decimal_field = field.safeGet<DecimalField<Decimal32>>();
            auto decimal_value = decimal_field.getValue();
            UInt32 scale = decimal_field.getScale();
            double result = DecimalUtils::convertTo<double>(decimal_value, scale);
            return py::cast(result);
        }

    case TypeIndex::Decimal64:
        {
            auto field = column[index];
            auto decimal_field = field.safeGet<DecimalField<Decimal64>>();
            auto decimal_value = decimal_field.getValue();
            UInt32 scale = decimal_field.getScale();
            double result = DecimalUtils::convertTo<double>(decimal_value, scale);
            return py::cast(result);
        }

    case TypeIndex::Decimal128:
        {
            auto field = column[index];
            auto decimal_field = field.safeGet<DecimalField<Decimal128>>();
            auto decimal_value = decimal_field.getValue();
            UInt32 scale = decimal_field.getScale();
            double result = DecimalUtils::convertTo<double>(decimal_value, scale);
            return py::cast(result);
        }

    case TypeIndex::Decimal256:
        {
            auto field = column[index];
            auto decimal_field = field.safeGet<DecimalField<Decimal256>>();
            auto decimal_value = decimal_field.getValue();
            UInt32 scale = decimal_field.getScale();
            double result = DecimalUtils::convertTo<double>(decimal_value, scale);
            return py::cast(result);
        }

    case TypeIndex::UUID:
        {
            auto field = column[index];
            auto uuid_value = field.safeGet<UUID>();
            const auto formatted_uuid = formatUUID(uuid_value);
            return import_cache.uuid.UUID()(String(formatted_uuid.data(), formatted_uuid.size()));
        }

	case TypeIndex::Array:
		{
			const auto & array_column = typeid_cast<const ColumnArray &>(column);

			const auto * array_type = typeid_cast<const DataTypeArray *>(actual_type.get());
			chassert(array_type);

			const auto & element_type = array_type->getNestedType();
			const auto & offsets = array_column.getOffsets();
			const auto & nested_column = array_column.getDataPtr();

			size_t start_offset = (index == 0) ? 0 : offsets[index - 1];
			size_t end_offset = offsets[index];

			py::list python_list;
			for (size_t i = start_offset; i < end_offset; ++i)
			{
				auto python_element = convertFieldToPython(*nested_column, element_type, i);
				python_list.append(python_element);
			}

			return python_list;
		}

	case TypeIndex::Tuple:
		{
			const auto & tuple_column = typeid_cast<const ColumnTuple &>(column);

			const auto * tuple_type = typeid_cast<const DataTypeTuple *>(actual_type.get());
			chassert(tuple_type);

			const auto & element_types = tuple_type->getElements();
			const auto & tuple_columns = tuple_column.getColumns();

			py::tuple python_tuple(tuple_columns.size());
			for (size_t i = 0; i < tuple_columns.size(); ++i)
			{
				auto python_element = convertFieldToPython(*(tuple_columns[i]), element_types[i], index);
				python_tuple[i] = python_element;
			}

			return python_tuple;
		}

	case TypeIndex::Interval:
        {
            auto field = column[index];
            auto interval_value = field.safeGet<Int64>();
            const auto * interval_type = typeid_cast<const DataTypeInterval *>(actual_type.get());
            chassert(interval_type);
            IntervalKind::Kind interval_kind = interval_type->getKind();

            switch (interval_kind)
            {
                case IntervalKind::Kind::Nanosecond:
                    return import_cache.datetime.timedelta()(py::arg("microseconds") = interval_value / 1000);
                case IntervalKind::Kind::Microsecond:
                    return import_cache.datetime.timedelta()(py::arg("microseconds") = interval_value);
                case IntervalKind::Kind::Millisecond:
                    return import_cache.datetime.timedelta()(py::arg("milliseconds") = interval_value);
                case IntervalKind::Kind::Second:
                    return import_cache.datetime.timedelta()(py::arg("seconds") = interval_value);
                case IntervalKind::Kind::Minute:
                    return import_cache.datetime.timedelta()(py::arg("minutes") = interval_value);
                case IntervalKind::Kind::Hour:
                    return import_cache.datetime.timedelta()(py::arg("hours") = interval_value);
                case IntervalKind::Kind::Day:
                    return import_cache.datetime.timedelta()(py::arg("days") = interval_value);
                case IntervalKind::Kind::Week:
                    return import_cache.datetime.timedelta()(py::arg("weeks") = interval_value);
                case IntervalKind::Kind::Month:
                    /// Approximate: 1 month = 30 days
                    return import_cache.datetime.timedelta()(py::arg("days") = interval_value * 30);
                case IntervalKind::Kind::Quarter:
                    /// 1 quarter = 3 months = 90 days
                    return import_cache.datetime.timedelta()(py::arg("days") = interval_value * 90);
                case IntervalKind::Kind::Year:
                    /// 1 year = 365 days
                    return import_cache.datetime.timedelta()(py::arg("days") = interval_value * 365);
                default:
                    throw Exception(ErrorCodes::LOGICAL_ERROR, "Unsupported interval kind");
            }
        }

	case TypeIndex::Map:
        {
            const auto & map_column = typeid_cast<const ColumnMap &>(column);

            const auto * map_type = typeid_cast<const DataTypeMap *>(actual_type.get());
            chassert(map_type);

            const auto & key_type = map_type->getKeyType();
            const auto & value_type = map_type->getValueType();

            /// Get the nested array column containing tuples
            const auto & nested_array = map_column.getNestedColumn();
            const auto & array_column = typeid_cast<const ColumnArray &>(nested_array);

            const auto & offsets = array_column.getOffsets();
            const auto & tuple_column_ptr = array_column.getDataPtr();
            const auto & tuple_column = typeid_cast<const ColumnTuple &>(*tuple_column_ptr);

            size_t start_offset = (index == 0) ? 0 : offsets[index - 1];
            size_t end_offset = offsets[index];

            const auto & key_column = tuple_column.getColumn(0);
            const auto & value_column = tuple_column.getColumn(1);

            bool use_dict = canTypeBeUsedAsDictKey(key_type);

            if (use_dict)
            {
                py::dict python_dict;
                for (size_t i = start_offset; i < end_offset; ++i)
                {
                    auto python_key = convertFieldToPython(key_column, key_type, i);
                    auto python_value = convertFieldToPython(value_column, value_type, i);

                    python_dict[std::move(python_key)] = std::move(python_value);
                }

                return python_dict;
            }
            else
            {
                py::list keys_list;
                py::list values_list;
                for (size_t i = start_offset; i < end_offset; ++i)
                {
                    auto python_key = convertFieldToPython(key_column, key_type, i);
                    auto python_value = convertFieldToPython(value_column, value_type, i);

                    keys_list.append(std::move(python_key));
                    values_list.append(std::move(python_value));
                }

                py::dict python_dict;
                python_dict["keys"] = std::move(keys_list);
                python_dict["values"] = std::move(values_list);

                return python_dict;
            }
        }

	case TypeIndex::Variant:
        {
            const auto & variant_column = typeid_cast<const ColumnVariant &>(column);
            auto discriminator = variant_column.globalDiscriminatorAt(index);
            if (discriminator == ColumnVariant::NULL_DISCRIMINATOR)
            {
                return py::none();
            }

            const auto & variant_type = typeid_cast<const DataTypeVariant &>(*actual_type);
            const auto & variants = variant_type.getVariants();
            const auto & variant_data_type = variants[discriminator];

            auto offset = variant_column.offsetAt(index);
            const auto & variant_inner_column = variant_column.getVariantByGlobalDiscriminator(discriminator);

            return convertFieldToPython(variant_inner_column, variant_data_type, offset);
        }


    case TypeIndex::Dynamic:
        {
            const auto & dynamic_column = typeid_cast<const ColumnDynamic &>(column);
            const auto & variant_column = dynamic_column.getVariantColumn();

            /// Check if this row has value in shared variant
            if (variant_column.globalDiscriminatorAt(index) == dynamic_column.getSharedVariantDiscriminator())
            {
                /// Get data from shared variant and deserialize it
                auto value = dynamic_column.getSharedVariant().getDataAt(variant_column.offsetAt(index));
                ReadBufferFromMemory buf(value.data(), value.size());
                auto variant_type = decodeDataType(buf);
                auto tmp_variant_column = variant_type->createColumn();
                auto variant_serialization = variant_type->getDefaultSerialization();
                variant_serialization->deserializeBinary(*tmp_variant_column, buf, FormatSettings{});

                /// Convert the deserialized value
                return convertFieldToPython(*tmp_variant_column, variant_type, 0);
            }
            else
            {
                /// Use variant conversion logic directly
                return convertFieldToPython(variant_column, dynamic_column.getVariantInfo().variant_type, index);
            }
        }

    case TypeIndex::Object:
        {
            return convertObjectToPython(column, actual_type, index);
        }

	case TypeIndex::IPv4:
		{
            auto field = column[index];
			auto ipv4_value = field.safeGet<IPv4>();

			char ipv4_str[IPV4_MAX_TEXT_LENGTH];
			char * ptr = ipv4_str;
			formatIPv4(reinterpret_cast<const unsigned char*>(&ipv4_value), ptr);
			const size_t ipv4_str_len = ptr - ipv4_str;

			return import_cache.ipaddress.ipv4_address()(String(ipv4_str, ipv4_str_len));
		}

    case TypeIndex::IPv6:
		{
            auto field = column[index];
			auto ipv6_value = field.safeGet<IPv6>();

			char ipv6_str[IPV6_MAX_TEXT_LENGTH];
			char * ptr = ipv6_str;
			formatIPv6(reinterpret_cast<const unsigned char*>(&ipv6_value), ptr);
			const size_t ipv6_str_len = ptr - ipv6_str;

			return import_cache.ipaddress.ipv6_address()(String(ipv6_str, ipv6_str_len));
		}

	/// Set types are used only in WHERE clauses for IN operations, not in actual data storage
	case TypeIndex::Set:
	/// JSONPaths is an internal type used only for JSON schema inference,
	case TypeIndex::JSONPaths:
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
