#pragma once

#include "PybindWrapper.h"

#include <Storages/ColumnsDescription.h>

namespace CHDB {

class PythonReader {
public:
    static DB::ColumnsDescription getActualTableStructure(const py::object & object, DB::ContextPtr & context);

    static bool isPythonReader(const py::object & object);
};

} // namespace CHDB
