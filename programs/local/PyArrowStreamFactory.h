#pragma once

#include "ArrowStreamWrapper.h"

#include <pybind11/pybind11.h>
#include <Core/Names.h>

namespace CHDB
{

/// Factory class for creating ArrowArrayStream from Python objects
class PyArrowStreamFactory
{
public:
    static std::unique_ptr<ArrowArrayStreamWrapper> createFromPyObject(
        pybind11::object & py_obj,
        const DB::Names & column_names);

private:
    static std::unique_ptr<ArrowArrayStreamWrapper> createFromTable(
        pybind11::object & table,
        const DB::Names & column_names);
};

} // namespace CHDB
