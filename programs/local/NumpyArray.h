#pragma once

#include "PybindWrapper.h"

#include <Columns/IColumn_fwd.h>
#include <Processors/Formats/IRowOutputFormat.h>
#include <base/types.h>

namespace CHDB
{

class InternalNumpyArray
{
public:
	explicit InternalNumpyArray(const DB::DataTypePtr & type);

	void init(size_t capacity);

	void resize(size_t capacity);

	py::array array;
	UInt8 * data;
	DB::DataTypePtr type;
	size_t count;
};

class NumpyArray {
public:
	explicit NumpyArray(const DB::DataTypePtr & type_);

	void init(size_t capacity);

	void resize(size_t capacity);

	void append(const DB::ColumnPtr & column);

	py::object toArray() const;

private:
	bool hava_null;
	std::unique_ptr<InternalNumpyArray> data_array;
	std::unique_ptr<InternalNumpyArray> mask_array;
};

} // namespace CHDB
