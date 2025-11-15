#pragma once

#include "PybindWrapper.h"

#include <unordered_map>
#include <base/types.h>

namespace CHDB {

class PythonTableCache {
public:
    void findQueryableObjFromQuery(const String & query_str);

    py::handle getQueryableObj(const String & table_name);

    void clear();

private:
    std::unordered_map<String, py::handle> py_table_cache;
};

} // namespace CHDB
