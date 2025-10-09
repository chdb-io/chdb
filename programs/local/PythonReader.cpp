#include "PythonReader.h"
#include "StoragePython.h"
#include <Common/memory.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
}

}

using namespace DB;

namespace CHDB {

ColumnsDescription PythonReader::getActualTableStructure(const py::object & object, ContextPtr & context)
{
#if USE_JEMALLOC
    Memory::MemoryCheckScope memory_check_scope;  // Enable memory checking for Python calls
#endif
    std::vector<std::pair<std::string, std::string>> schema;

    schema = object.attr("get_schema")().cast<std::vector<std::pair<std::string, std::string>>>();

    return StoragePython::getTableStructureFromData(schema);
}

bool PythonReader::isPythonReader(const py::object & object)
{
    return isInheritsFromPyReader(object);
}

} // namespace CHDB
