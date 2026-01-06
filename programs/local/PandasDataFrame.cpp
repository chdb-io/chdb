#include "PandasDataFrame.h"
#include "NumpyType.h"
#include "PandasAnalyzer.h"
#include "PandasCacheItem.h"
#include "PythonImporter.h"

#include <Common/Exception.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypeObject.h>
#include <Interpreters/Context.h>
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

static DataTypePtr inferDataTypeFromPandasColumn(PandasBindColumn & column, ContextPtr & context)
{
    auto numpy_type = ConvertNumpyType(column.type);

    /// TODO: support masked object, timezone and category.

    if (numpy_type.type == NumpyNullableType::OBJECT)
    {
		PandasAnalyzer analyzer(context->getSettingsRef());
		if (analyzer.Analyze(column.handle))
        {
            const auto & analyzed_type = analyzer.analyzedType();
            const bool use_string_fallback = !context->getQueryContext() || !context->getQueryContext()->isJSONSupported();
            const bool is_json_type = typeid_cast<const DataTypeObject *>(analyzed_type.get()) != nullptr;

            if (!is_json_type || !use_string_fallback)
                return analyzed_type;
		}

        numpy_type.type = NumpyNullableType::STRING;
	}

    auto data_type = NumpyToDataType(numpy_type);

    if (!data_type->isNullable())
    {
        bool column_has_mask = py::hasattr(column.handle.attr("array"), "_mask");
        if (column_has_mask)
            return std::make_shared<DataTypeNullable>(data_type);
    }

    return data_type;
}

ColumnsDescription PandasDataFrame::getActualTableStructure(const py::object & object, ContextPtr & context)
{
#if USE_JEMALLOC
    ::Memory::MemoryCheckScope memory_check_scope;
#endif
    chassert(py::gil_check());
    NamesAndTypesList names_and_types;

    PandasDataFrameBind df(object);
	size_t column_count = py::len(df.names);
	if (column_count == 0 || py::len(df.types) == 0)
		throw DB::Exception(ErrorCodes::BAD_ARGUMENTS, "Unexpected empty DataFrame");

    if (column_count != py::len(df.types))
        throw DB::Exception(ErrorCodes::BAD_ARGUMENTS,
                            "Unexpected DataFrame with column count: {} and type count: {}", column_count, py::len(df.types));

	for (size_t col_idx = 0; col_idx < column_count; col_idx++) {
		auto col_name = py::str(df.names[col_idx]);
		auto column = df[col_idx];
		auto data_type = inferDataTypeFromPandasColumn(column, context);

        names_and_types.push_back({col_name, data_type});
	}

    return ColumnsDescription(names_and_types);
}

bool PandasDataFrame::isPandasDataframe(const py::object & object)
{
#if USE_JEMALLOC
    ::Memory::MemoryCheckScope memory_check_scope;
#endif
    chassert(py::gil_check());

    if (!ModuleIsLoaded<PandasCacheItem>())
		return false;

	auto & importer_cache = PythonImporter::ImportCache();
	bool is_df = py::isinstance(object, importer_cache.pandas.DataFrame());

    if (!is_df)
        return false;

	auto arrow_dtype = importer_cache.pandas.ArrowDtype();
	py::list dtypes = object.attr("dtypes");
	for (auto & dtype : dtypes)
    {
		if (py::isinstance(dtype, arrow_dtype))
			return false;
	}

	return true;
}

bool PandasDataFrame::isPyArrowBacked(const py::handle & /*object*/)
{
    /// TODO: check if object is pyarrow backed
    return false;
}

} // namespace CHDB
