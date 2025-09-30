#include "PyArrowTable.h"
#include "ArrowSchema.h"
#include "PyArrowCacheItem.h"
#include "PythonImporter.h"

#include <Interpreters/Context.h>

using namespace DB;

namespace CHDB
{

PyArrowObjectType PyArrowTable::getArrowType(const py::object & obj)
{
    chassert(py::gil_check());

	if (ModuleIsLoaded<PyarrowCacheItem>())
    {
		auto & import_cache = PythonImporter::ImportCache();
		auto table_class = import_cache.pyarrow.table();

		if (py::isinstance(obj, table_class))
			return PyArrowObjectType::Table;
	}

	return PyArrowObjectType::Invalid;
}

bool PyArrowTable::isPyArrowTable(const py::object & object)
{
    try
    {
        return getArrowType(object) == PyArrowObjectType::Table;
    }
    catch (const py::error_already_set &)
    {
        return false;
    }
}

ColumnsDescription PyArrowTable::getActualTableStructure(const py::object & object, ContextPtr & context)
{
    chassert(py::gil_check());
    chassert(isPyArrowTable(object));

    NamesAndTypesList names_and_types;

    auto obj_schema = object.attr("schema");
	auto export_to_c = obj_schema.attr("_export_to_c");
	ArrowSchemaWrapper schema;
	export_to_c(reinterpret_cast<uint64_t>(&schema.arrow_schema));

    ArrowSchemaWrapper::convertArrowSchema(schema, names_and_types, context);

    return ColumnsDescription(names_and_types);
}

} // namespace CHDB
