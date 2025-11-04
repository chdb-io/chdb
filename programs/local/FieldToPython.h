#pragma once

#include <Core/Field.h>
#include <DataTypes/IDataType.h>
#include <Columns/IColumn.h>
#include <pybind11/pybind11.h>

namespace CHDB
{

pybind11::object convertTimeFieldToPython(const DB::Field & field);

pybind11::object convertTime64FieldToPython(const DB::Field & field);

pybind11::object convertFieldToPython(
    const DB::IColumn & column,
    const DB::DataTypePtr & type,
    size_t index);

} // namespace CHDB
