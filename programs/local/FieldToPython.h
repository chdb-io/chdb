#pragma once

#include <Core/Field.h>
#include <DataTypes/IDataType.h>
#include <pybind11/pybind11.h>

namespace CHDB
{

pybind11::object convertFieldToPython(
    const DB::Field & field,
    const DB::DataTypePtr & type);

} // namespace CHDB
