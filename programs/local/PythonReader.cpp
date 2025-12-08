#include "PythonReader.h"
#include "StoragePython.h"
#if USE_JEMALLOC
#    include <Common/memory.h>
#endif

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

    return StoragePython::getTableStructureFromData(schema, context);
}

bool PythonReader::isPythonReader(const py::object & object)
{
    return isInheritsFromPyReader(object);
}

} // namespace CHDB
