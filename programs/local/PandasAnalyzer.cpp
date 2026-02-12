#include "PandasAnalyzer.h"
#include "PythonConversion.h"
#include "PythonImporter.h"

#include <Common/Exception.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypeObject.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeFactory.h>
#if USE_JEMALLOC
#include <Common/memory.h>
#endif

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

namespace Setting
{
	extern const SettingsUInt64 pandas_analyze_sample;
}

}

using namespace DB;

namespace CHDB
{

static TypeIndex getTypeIndex(const DataTypePtr & type)
{
    return type ? type->getTypeId() : TypeIndex::Nothing;
}

static bool mergeAnalyzedTypes(DataTypePtr & current, const DataTypePtr & next)
{
    auto current_idx = getTypeIndex(current);
    auto next_idx = getTypeIndex(next);

    if (current_idx == TypeIndex::Nothing)
    {
        current = next;
        return true;
    }
    if (next_idx == TypeIndex::Nothing)
        return true;

	chassert(current && next);

    bool current_is_bool = isBool(current);
    bool next_is_bool = isBool(next);
    if (current_is_bool != next_is_bool)
        return false;
    if (current_is_bool && next_is_bool)
        return true;

    if (current_idx == next_idx)
        return true;

    if (current_idx == TypeIndex::Object || next_idx == TypeIndex::Object)
        return false;

    auto is_signed_int = [](TypeIndex idx) { return idx == TypeIndex::Int32 || idx == TypeIndex::Int64; };
    if (is_signed_int(current_idx) && is_signed_int(next_idx))
    {
        current = std::make_shared<DataTypeInt64>();
        return true;
    }

    auto is_expected_numeric = [](TypeIndex idx) {
        return idx == TypeIndex::Int32 || idx == TypeIndex::Int64
            || idx == TypeIndex::UInt64 || idx == TypeIndex::Float64;
    };
    if (!is_expected_numeric(current_idx) || !is_expected_numeric(next_idx))
        return false;

    current = std::make_shared<DataTypeFloat64>();
    return true;
}

static DataTypePtr getBestIntegerType(PyObject * obj)
{
	chassert(obj);

    int overflow = 0;
    int64_t val = PyLong_AsLongLongAndOverflow(obj, &overflow);

	if (overflow == -1)
	{
		return std::make_shared<DataTypeFloat64>();
	}
	else if (overflow == 1)
	{
		uint64_t unsigned_value = PyLong_AsUnsignedLongLong(obj);
		if (PyErr_Occurred())
		{
			PyErr_Clear();
			return std::make_shared<DataTypeFloat64>();
		}
		else
		{
			return std::make_shared<DataTypeUInt64>();
		}
	}
	else if (val == -1 && PyErr_Occurred())
	{
		PyErr_Clear();
		return {};
	}

	if (val < static_cast<Int64>(std::numeric_limits<int32_t>::min()) || val > static_cast<Int64>(std::numeric_limits<int32_t>::max()))
		return std::make_shared<DataTypeInt64>();
	else
		return std::make_shared<DataTypeInt32>();
}

PandasAnalyzer::PandasAnalyzer(const DB::Settings & settings)
{
	analyzed_type = {};

	sample_size = settings[DB::Setting::pandas_analyze_sample];
}

bool PandasAnalyzer::Analyze(py::object column) {
#if USE_JEMALLOC
	::Memory::MemoryCheckScope memory_check_scope;
#endif
	if (sample_size == 0)
		return false;

	if (sample_size < 0)
	{
		analyzed_type = std::make_shared<DataTypeNullable>(std::make_shared<DataTypeObject>(DataTypeObject::SchemaFormat::JSON));
		return true;
	}

	auto & import_cache = PythonImporter::ImportCache();
	auto pandas = import_cache.pandas();
	if (!pandas)
		return false;

	bool can_convert = true;
	auto increment = getSampleIncrement(py::len(column));
	auto type = innerAnalyze(column, can_convert, increment);

	if (can_convert && !type && increment > 1) {
		auto first_valid_index = column.attr("first_valid_index")();
		if (GetPythonObjectType(first_valid_index) != PythonObjectType::None)
		{
			auto row = column.attr("__getitem__");
			auto obj = row(first_valid_index);
			type = getItemType(obj, can_convert);
		}
	}

	if (can_convert)
	{
		if (!type)
			type = std::make_shared<DataTypeString>();
		analyzed_type = std::make_shared<DataTypeNullable>(type);
	}

	return can_convert;
}

size_t PandasAnalyzer::getSampleIncrement(size_t rows)
{
	auto sample = static_cast<uint64_t>(sample_size);
	if (sample > rows)
		sample = rows;

	if (sample == 0)
		return rows;

	return rows / sample;
}

DataTypePtr PandasAnalyzer::getItemType(py::object obj, bool & can_convert)
{
	auto object_type = GetPythonObjectType(obj);

	switch (object_type)
	{
	case PythonObjectType::Dict:
		return std::make_shared<DataTypeObject>(DataTypeObject::SchemaFormat::JSON);
	case PythonObjectType::Integer:
		{
			auto best_type = getBestIntegerType(obj.ptr());
			if (!best_type)
			{
				can_convert = false;
				return std::make_shared<DataTypeString>();
			}
			return best_type;
		}
	case PythonObjectType::Float:
		{
			if (std::isnan(PyFloat_AsDouble(obj.ptr())))
				return {};
			return std::make_shared<DataTypeFloat64>();
		}
	case PythonObjectType::Bool:
		return std::const_pointer_cast<IDataType>(DataTypeFactory::instance().get("Bool"));
	case PythonObjectType::None:
		return {};
	case PythonObjectType::Tuple:
	case PythonObjectType::List:
	case PythonObjectType::Decimal:
	case PythonObjectType::Datetime:
	case PythonObjectType::Time:
	case PythonObjectType::Date:
	case PythonObjectType::Timedelta:
	case PythonObjectType::String:
	case PythonObjectType::Uuid:
	case PythonObjectType::ByteArray:
	case PythonObjectType::MemoryView:
	case PythonObjectType::Bytes:
	case PythonObjectType::NdDatetime:
	case PythonObjectType::NdArray:
	case PythonObjectType::Other:
		{
			can_convert = false;
			return std::make_shared<DataTypeString>();
		}
	default:
		throw DB::Exception(DB::ErrorCodes::LOGICAL_ERROR,
							"Unknown python object type {}", object_type);
	}
}

DataTypePtr PandasAnalyzer::innerAnalyze(py::object column, bool & can_convert, size_t increment) {
	size_t rows = py::len(column);
	can_convert = true;

	if (rows == 0)
		return {};

	auto & import_cache = PythonImporter::ImportCache();
	auto pandas_series = import_cache.pandas.Series();

	if (pandas_series && py::isinstance(column, pandas_series))
		column = column.attr("__array__")();

	auto row = column.attr("__getitem__");

	DataTypePtr current_type = {};
	for (size_t i = 0; i < rows; i += increment)
	{
		auto obj = row(i);
		DataTypePtr next_type = getItemType(obj, can_convert);

		if (!can_convert)
			return next_type;

		if (!mergeAnalyzedTypes(current_type, next_type))
		{
			can_convert = false;
			return next_type;
		}
	}

	return current_type;
}

} // namespace CHDB
