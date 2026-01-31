#pragma once

#include "PybindWrapper.h"

#include <cstdint>
#include <DataTypes/IDataType.h>

namespace CHDB {

enum class NumpyNullableType : uint8_t {
	BOOL,
	INT_8,
	UINT_8,
	INT_16,
	UINT_16,
	INT_32,
	UINT_32,
	INT_64,
	UINT_64,
	FLOAT_16,
	FLOAT_32,
	FLOAT_64,
	OBJECT,
	UNICODE,
	DATETIME_S,
	DATETIME_MS,
	DATETIME_NS,
	DATETIME_US,
	TIMEDELTA_NS,
	TIMEDELTA_US,
	TIMEDELTA_MS,
	TIMEDELTA_S,
	TIMEDELTA_D,  // Day precision

	CATEGORY,
	STRING,
};

struct NumpyType {
	NumpyNullableType type;
	String timezone;

	String toString() const;
};

enum class NumpyObjectType : uint8_t {
	INVALID,
	NDARRAY1D,
	NDARRAY2D,
	LIST,
	DICT,
};

NumpyType ConvertNumpyType(const py::handle & col_type);

std::shared_ptr<DB::IDataType> NumpyToDataType(const NumpyType & col_type);

String DataTypeToNumpyTypeStr(const std::shared_ptr<const DB::IDataType> & data_type);

py::object ConvertNumpyDtype(const py::handle & numpy_array);

} // namespace CHDB
