#pragma once

#include <DataTypes/IDataType.h>
#include <Columns/IColumn.h>
#include <pybind11/pybind11.h>

namespace CHDB
{

pybind11::object convertObjectToPython(
    const DB::IColumn & column,
    const DB::DataTypePtr & type,
    size_t index);

} // namespace CHDB
