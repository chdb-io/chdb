#include "PyArrowTable.h"
#include "ArrowSchema.h"
#include "PyArrowCacheItem.h"
#include "PythonImporter.h"

#include <Common/Exception.h>
#include <Interpreters/Context.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypeDate.h>
#include <DataTypes/DataTypeDateTime.h>
#include <arrow/c/bridge.h>
#include <Processors/Formats/Impl/ArrowColumnToCHColumn.h>
#include <Formats/FormatFactory.h>

namespace DB
{

namespace ErrorCodes
{
extern const int BAD_ARGUMENTS;
extern const int PY_EXCEPTION_OCCURED;
}

}

using namespace DB;

namespace CHDB
{

static void convertArrowSchema(
    ArrowSchemaWrapper & schema,
    NamesAndTypesList & names_and_types,
    ContextPtr & context)
{
    if (!schema.arrow_schema.release)
    {
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "ArrowSchema is already released");
    }

    /// Import ArrowSchema to arrow::Schema
    auto arrow_schema_result = arrow::ImportSchema(&schema.arrow_schema);
    if (!arrow_schema_result.ok())
    {
        throw Exception(ErrorCodes::BAD_ARGUMENTS,
                        "Failed to import Arrow schema: {}", arrow_schema_result.status().message());
    }

    const auto & arrow_schema = arrow_schema_result.ValueOrDie();

    const auto format_settings = getFormatSettings(context);

    /// Convert Arrow schema to ClickHouse header
    auto block = ArrowColumnToCHColumn::arrowSchemaToCHHeader(
        *arrow_schema,
        nullptr,
        "Arrow",
        format_settings.arrow.skip_columns_with_unsupported_types_in_schema_inference,
        format_settings.schema_inference_make_columns_nullable != 0,
        false,
        format_settings.parquet.allow_geoparquet_parser);

    for (const auto & column : block)
    {
        names_and_types.emplace_back(column.name, column.type);
    }
}

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

    convertArrowSchema(schema, names_and_types, context);

    return ColumnsDescription(names_and_types);
}

} // namespace CHDB
