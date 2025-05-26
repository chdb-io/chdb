#include "PythonConversion.h"
#include "PythonImporter.h"

#include <Common/Exception.h>
#include <IO/ReadBuffer.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
}

}

using namespace DB;

namespace CHDB
{

PythonObjectType GetPythonObjectType(const py::handle & obj)
{
	auto & import_cache = PythonImporter::ImportCache();

	/// TODO: support uuid, numpy.

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

	if (py::isinstance(obj, import_cache.decimal.Decimal()))
		return PythonObjectType::Decimal;

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

bool isInteger(const py::handle & obj)
{
	return GetPythonObjectType(obj) == PythonObjectType::Integer;
}

void writeInteger(const py::handle & obj, rapidjson::Value & json_value)
{
	auto ptr = obj.ptr();
	int overflow = 0;
	int64_t value = PyLong_AsLongLongAndOverflow(ptr, &overflow);

	if (overflow != 0)
	{
		PyErr_Clear();

		if (overflow == 1)
		{
			uint64_t unsigned_value = PyLong_AsUnsignedLongLong(ptr);
			if (!PyErr_Occurred())
			{
				json_value.SetUint64(value);
				return;
			}

			PyErr_Clear();
		}

		double float_value = PyLong_AsDouble(ptr);
		if (float_value == -1.0 && PyErr_Occurred())
		{
			PyErr_Clear();
			throw DB::Exception(ErrorCodes::BAD_ARGUMENTS, "Unexpected transform integer into double in python dict");
		}

		json_value.SetDouble(float_value);
	}
	else if (value == -1 && PyErr_Occurred())
	{
		PyErr_Clear();
		throw DB::Exception(ErrorCodes::BAD_ARGUMENTS, "Unexpected integer in python dict");
	}
	else
	{
		json_value.SetInt64(value);
	}
}

bool isNone(const py::handle & obj)
{
	return GetPythonObjectType(obj) == PythonObjectType::None;
}

void writeNone(const py::handle & obj, rapidjson::Value & json_value)
{
	json_value.SetNull();
}

bool isFloat(const py::handle & obj)
{
	return GetPythonObjectType(obj) == PythonObjectType::Float;
}

void writeFloat(const py::handle & obj, rapidjson::Value & json_value)
{
	auto ptr = obj.ptr();
	if (std::isnan(PyFloat_AsDouble(ptr)))
	{
		json_value.SetNull();
		return;
	}

	double value = obj.cast<double>();
	json_value.SetDouble(value);
}

bool isBoolean(const py::handle & obj)
{
	return GetPythonObjectType(obj) == PythonObjectType::Bool;
}

void writeBoolean(const py::handle & obj, rapidjson::Value & json_value)
{
	json_value.SetBool(py::cast<bool>(obj));
}

bool isDecimal(const py::handle & obj)
{
	return GetPythonObjectType(obj) == PythonObjectType::Decimal;
}

void writeDecimal(const py::handle & obj, rapidjson::Value & json_value, rapidjson::Document::AllocatorType & allocator)
{
	const auto & str = obj.cast<std::string>();
	json_value.SetString(str.data(), str.size(), allocator);
}

bool isString(const py::handle & obj)
{
	return GetPythonObjectType(obj) == PythonObjectType::String
		|| GetPythonObjectType(obj) == PythonObjectType::Bytes;
}

void writeString(const py::handle & obj, rapidjson::Value & json_value, rapidjson::Document::AllocatorType & allocator)
{
	const auto & str = obj.cast<std::string>();
	json_value.SetString(str.data(), str.size(), allocator);
}

void writeOthers(const py::handle & obj, rapidjson::Value & json_value, rapidjson::Document::AllocatorType & allocator)
{
	String str = py::str(obj);
	json_value.SetString(str.data(), str.size(), allocator);
}

void convert_to_json_str(const py::handle & obj, String & ret)
{
    rapidjson::Document d;
    d.SetObject();
    rapidjson::Document::AllocatorType & allocator = d.GetAllocator();

    std::function<void(const py::handle &, rapidjson::Value &)> convert;
    convert = [&](const py::handle & obj, rapidjson::Value & json_value) {
        if (py::isinstance<py::dict>(obj))
        {
            json_value.SetObject();
            for (auto& item : py::cast<py::dict>(obj))
            {
                rapidjson::Value key;
                auto ket_str = py::str(item.first).cast<std::string>();
                key.SetString(ket_str.data(), ket_str.size(), allocator);

                rapidjson::Value val;
                convert(item.second, val);

                json_value.AddMember(key, val, allocator);
            }
        }
        else if (py::isinstance<py::list>(obj))
        {
            json_value.SetArray();
            auto tmp_list = py::cast<py::list>(obj);
            for (auto & item : tmp_list)
            {
                rapidjson::Value element;
                convert(item, element);
                json_value.PushBack(element, allocator);
            }
        }
        else if (py::isinstance<py::tuple>(obj))
        {
            json_value.SetArray();
            auto tmp_tuple = py::cast<py::tuple>(obj);
            for (auto & item : tmp_tuple)
            {
                rapidjson::Value element;
                convert(item, element);
                json_value.PushBack(element, allocator);
            }
        }
        else if (py::isinstance<py::array>(obj))
        {
            auto arr = py::cast<py::array>(obj);
            json_value.SetArray();
            auto my_list = arr.attr("tolist")();
            auto arr_size = arr.size();

            for (size_t i = 0; i < arr_size; i++)
            {
                rapidjson::Value element;
                auto item = my_list.attr("__getitem__")(i);
                convert(item, element);
                json_value.PushBack(element, allocator);
		    }
        }
        else if (isInteger(obj))
        {
            writeInteger(obj, json_value);
        }
        else if (isNone(obj))
        {
            writeNone(obj, json_value);
        }
        else if (isFloat(obj))
        {
            writeFloat(obj, json_value);
        }
        else if (isBoolean(obj))
        {
            writeBoolean(obj, json_value);
        }
        else if (isDecimal(obj))
        {
            writeDecimal(obj, json_value, allocator);
        }
        else if (isString(obj))
        {
            writeString(obj, json_value, allocator);
        }
        else
        {
            writeOthers(obj, json_value, allocator);
        }
    };

    rapidjson::Value root;
    convert(obj, root);

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    root.Accept(writer);

    ret = buffer.GetString();
}

bool tryInsertJsonResult(
    const py::handle & handle,
    const FormatSettings & format_settings,
    MutableColumnPtr & column,
    SerializationPtr & serialization)
{
    if (py::isinstance<py::dict>(handle))
    {
        String json_str;
        convert_to_json_str(handle, json_str);

        ReadBuffer read_buffer(json_str.data(), json_str.size(), 0);
        serialization->deserializeTextJSON(*column, read_buffer, format_settings);

        return true;
    }

    return false;
}

} // namespace CHDB
