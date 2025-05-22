#include "PythonConversion.h"
#include "PythonImporter.h"

namespace CHDB
{

PythonObjectType GetPythonObjectType(py::handle & obj)
{
	auto & import_cache = PythonImporter::ImportCache();

	/// TODO: support decimal, uuid, numpy.

	if (obj.is_none())
		return PythonObjectType::None;

	if (obj.is(import_cache.pandas.NaT()))
		return PythonObjectType::None;

	if (obj.is(import_cache.pandas.NA()))
		return PythonObjectType::None;

	if (py::isinstance<py::bool_>(obj))
		return PythonObjectType::Bool;

	if (py::isinstance<py::int_>(obj))
		return PythonObjectType::Integer;

	if (py::isinstance<py::float_>(obj))
		return PythonObjectType::Float;

	if (py::isinstance(obj, import_cache.datetime.datetime()))
		return PythonObjectType::Datetime;

	if (py::isinstance(obj, import_cache.datetime.time()))
		return PythonObjectType::Time;

	if (py::isinstance(obj, import_cache.datetime.date()))
		return PythonObjectType::Date;

	if (py::isinstance(obj, import_cache.datetime.timedelta()))
		return PythonObjectType::Timedelta;

	if (py::isinstance<py::str>(obj))
		return PythonObjectType::String;

	if (py::isinstance<py::bytearray>(obj))
		return PythonObjectType::ByteArray;

	if (py::isinstance<py::bytes>(obj))
		return PythonObjectType::Bytes;

	if (py::isinstance<py::memoryview>(obj))
		return PythonObjectType::MemoryView;

	if (py::isinstance<py::list>(obj))
		return PythonObjectType::List;

	if (py::isinstance<py::tuple>(obj))
		return PythonObjectType::Tuple;

	if (py::isinstance<py::dict>(obj))
		return PythonObjectType::Dict;

	return PythonObjectType::Other;
}

} // namespace CHDB
