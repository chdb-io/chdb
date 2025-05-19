#include "PandasAnalyzer.h"
#include "DataTypes/DataTypeObject.h"
#include "DataTypes/DataTypeString.h"
#include "PythonConversion.h"
#include "PythonImporter.h"

#include <Common/Exception.h>
#include <DataTypes/DataTypesNumber.h>
#include <cstddef>
#include <memory>

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

}

using namespace DB;

namespace CHDB {

bool PandasAnalyzer::Analyze(py::object column) {
	if (sample_size == 0)
		return false;

	auto & import_cache = PythonImporter::ImportCache();
	auto pandas = import_cache.pandas();
	if (!pandas)
		return false;

	bool can_convert = true;
	auto increment = getSampleIncrement(py::len(column));
	auto type = innerAnalyze(column, can_convert, increment);

	if (can_convert)
		analyzed_type = type;

	return can_convert;
}

size_t PandasAnalyzer::getSampleIncrement(size_t rows)
{
	auto sample = sample_size;
	if (sample > rows)
		sample = rows;

	if (sample == 0)
		return rows;

	return rows / sample;
}

DataTypePtr PandasAnalyzer::getItemType(py::object obj, bool & can_convert)
{
	auto object_type = GetPythonObjectType(obj);

	switch (object_type) {
	case PythonObjectType::Dict:
	{
		PyDictionary dict = PyDictionary(py::reinterpret_borrow<py::object>(obj));

		return std::make_shared<DataTypeObject>(DataTypeObject::SchemaFormat::JSON);
	}
	case PythonObjectType::None:
	case PythonObjectType::Bool:
	case PythonObjectType::Integer:
	case PythonObjectType::Float:
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
	case PythonObjectType::Tuple:
	case PythonObjectType::List:
	case PythonObjectType::NdDatetime:
	case PythonObjectType::NdArray:
	case PythonObjectType::Other:
		can_convert = false;
		return std::make_shared<DataTypeString>();
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

	DataTypePtr item_type = {};
	for (size_t i = 0; i < rows; i += increment)
	{
		auto obj = row(i);
		item_type = getItemType(obj, can_convert);

		/// TODO: support more types such as list, tuple.

		if (!can_convert)
			return item_type;
	}

	return item_type;
}

} // namespace CHDB
