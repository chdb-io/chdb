#pragma once

#include "PybindWrapper.h"
#include "PythonUtils.h"

#include <base/types.h>
#include <rapidjson/document.h>

namespace CHDB {

enum class PythonObjectType {
	Other,
	None,
	Integer,
	Float,
	Bool,
	Decimal,
	Uuid,
	Datetime,
	Date,
	Time,
	Timedelta,
	String,
	ByteArray,
	MemoryView,
	Bytes,
	List,
	Tuple,
	Dict,
	NdArray,
	NdDatetime,
	Value
};

PythonObjectType GetPythonObjectType(const py::handle & obj);

bool isInteger(const py::handle & obj);

void writeInteger(const py::handle & obj, rapidjson::Value & json_value);

bool isNone(const py::handle & obj);

void writeNone(const py::handle & obj, rapidjson::Value & json_value);

bool isFloat(const py::handle & obj);

void writeFloat(const py::handle & obj, rapidjson::Value & json_value);

bool isBoolean(const py::handle & obj);

void writeBoolean(const py::handle & obj, rapidjson::Value & json_value);

bool isDecimal(const py::handle & obj);

void writeDecimal(const py::handle & obj, rapidjson::Value & json_value, rapidjson::Document::AllocatorType & allocator);

bool isString(const py::handle & obj);

void writeString(const py::handle & obj, rapidjson::Value & json_value, rapidjson::Document::AllocatorType & allocator);

void writeOthers(const py::handle & obj, rapidjson::Value & json_value, rapidjson::Document::AllocatorType & allocator);

void convert_to_json_str(const py::handle & obj, String & ret);

bool tryInsertJsonResult(
    const py::handle & handle,
    const DB::FormatSettings & format_settings,
    DB::MutableColumnPtr & column,
    DB::SerializationPtr & serialization);

} // namespace CHDB
