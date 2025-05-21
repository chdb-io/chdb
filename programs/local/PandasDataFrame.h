#pragma once

#include "PybindWrapper.h"

#include "Storages/ColumnsDescription.h"

namespace CHDB {

class PandasDataFrame {
public:
    static DB::ColumnsDescription getActualTableStructure(const py::object & object, DB::ContextPtr & context);

    static bool isPandasDataframe(const py::object & object);

    static bool IsPyArrowBacked(const py::handle & object);
};

} // namespace CHDB
