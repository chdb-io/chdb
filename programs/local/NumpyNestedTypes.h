#pragma once

#include "NumpyArray.h"

namespace CHDB
{

bool CHColumnArrayToNumpyArray(NumpyAppendData & append_data, const DB::DataTypePtr & data_type);

bool CHColumnTupleToNumpyArray(NumpyAppendData & append_data, const DB::DataTypePtr & data_type);

bool CHColumnMapToNumpyArray(NumpyAppendData & append_data, const DB::DataTypePtr & data_type);

bool CHColumnObjectToNumpyArray(NumpyAppendData & append_data, const DB::DataTypePtr & data_type);

bool CHColumnVariantToNumpyArray(NumpyAppendData & append_data, const DB::DataTypePtr & data_type);

bool CHColumnDynamicToNumpyArray(NumpyAppendData & append_data, const DB::DataTypePtr & data_type);

} // namespace CHDB
