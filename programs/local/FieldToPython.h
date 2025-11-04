#pragma once

#include <Core/Field.h>
#include <DataTypes/IDataType.h>
#include <Columns/IColumn.h>
#include <pybind11/pybind11.h>

namespace CHDB
{

pybind11::object convertFieldToPython(
    const DB::ColumnPtr & column,
    const DB::DataTypePtr & type,
    size_t index);

} // namespace CHDB
