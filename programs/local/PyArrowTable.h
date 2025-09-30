#pragma once

#include "PybindWrapper.h"

#include <Storages/ColumnsDescription.h>
#include <DataTypes/IDataType.h>

namespace CHDB
{

enum class PyArrowObjectType
{
    Invalid,
    Table
};

class PyArrowTable
{
public:
    static DB::ColumnsDescription getActualTableStructure(const py::object & object, DB::ContextPtr & context);

    static bool isPyArrowTable(const py::object & object);

    static PyArrowObjectType getArrowType(const py::object & object);
};

} // namespace CHDB
