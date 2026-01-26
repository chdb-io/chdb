#include "PandasDataFrame.h"
#include "NumpyType.h"
#include "PandasAnalyzer.h"
#include "PandasCacheItem.h"
#include "PythonImporter.h"

#include <Columns/ColumnNullable.h>
#include <Columns/ColumnString.h>
#include <Common/Exception.h>
#include <DataTypes/DataTypeLowCardinality.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypeObject.h>
#include <DataTypes/DataTypeString.h>
#include <Interpreters/Context.h>
#if USE_JEMALLOC
#    include <Common/memory.h>
#endif

namespace DB
{

namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
    extern const int LOGICAL_ERROR;
}

}

using namespace DB;

namespace CHDB
{

static DataTypePtr inferDataTypeFromPandasColumn(
    PandasBindColumn & column,
    ContextPtr & context,
    DataSourceWrapper & wrapper)
{
    auto numpy_type = ConvertNumpyType(column.type);

    if (numpy_type.type == NumpyNullableType::CATEGORY)
    {
        chassert(py::hasattr(column.handle, "cat"));
		chassert(py::hasattr(column.handle.attr("cat"), "categories"));

        py::array categories = py::array(column.handle.attr("cat").attr("categories"));
        auto categories_numpy_type = ConvertNumpyType(categories.attr("dtype"));
        if (categories_numpy_type.type == NumpyNullableType::OBJECT)
        {
            auto inner_type = std::make_shared<DataTypeNullable>(std::make_shared<DataTypeString>());
            return std::make_shared<DataTypeLowCardinality>(inner_type);
        }
        else
        {
            py::array values = py::array(column.handle.attr("to_numpy")());
            auto col_name = py::str(column.name).cast<std::string>();
            wrapper.cacheColumnData(col_name, values);
            numpy_type = ConvertNumpyType(values.attr("dtype"));
        }
    }

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

ColumnsDescription PandasDataFrame::getActualTableStructure(DataSourceWrapper & wrapper, ContextPtr & context)
{
#if USE_JEMALLOC
    ::Memory::MemoryCheckScope memory_check_scope;
#endif
    chassert(py::gil_check());
    NamesAndTypesList names_and_types;

    const auto & object = wrapper.getDataSource();
    PandasDataFrameBind df(object);
    size_t column_count = py::len(df.names);
    if (column_count == 0 || py::len(df.types) == 0)
        throw DB::Exception(ErrorCodes::BAD_ARGUMENTS, "Unexpected empty DataFrame");

    if (column_count != py::len(df.types))
        throw DB::Exception(ErrorCodes::BAD_ARGUMENTS,
                            "Unexpected DataFrame with column count: {} and type count: {}", column_count, py::len(df.types));

    for (size_t col_idx = 0; col_idx < column_count; col_idx++)
    {
        auto col_name = py::str(df.names[col_idx]);
        auto column = df[col_idx];
        auto data_type = inferDataTypeFromPandasColumn(column, context, wrapper);

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

void PandasDataFrame::fillColumn(
    const py::handle & data_source,
    const std::string & col_name,
    DB::ColumnWrapper & column,
    DataSourceWrapper & wrapper)
{
    chassert(py::gil_check());

    py::object series = data_source[py::str(col_name)];
    py::object dtype = data_source.attr("dtypes")[py::str(col_name)];

    auto numpy_type = ConvertNumpyType(dtype);
    column.is_object_type = (numpy_type.type == NumpyNullableType::OBJECT || numpy_type.type == NumpyNullableType::STRING);

    if (numpy_type.type == NumpyNullableType::CATEGORY)
    {
        chassert(py::hasattr(series, "cat"));
		chassert(py::hasattr(series.attr("cat"), "categories"));

        py::array categories = py::array(series.attr("cat").attr("categories"));
        auto categories_numpy_type = ConvertNumpyType(categories.attr("dtype"));

        if (categories_numpy_type.type == NumpyNullableType::OBJECT)
        {
            column.is_category = true;

            std::vector<std::string> category_strings;
            try
            {
                category_strings = py::cast<std::vector<std::string>>(categories);
            }
            catch (const py::cast_error &)
            {
                throw DB::Exception(
                    DB::ErrorCodes::BAD_ARGUMENTS,
                    "Categorical column '{}' contains non-string categories. "
                    "Only string categories are supported for LowCardinality conversion",
                    col_name);
            }
            auto dict_column = DB::ColumnString::create();
            dict_column->insertDefault();
            for (const auto & s : category_strings)
                dict_column->insertData(s.data(), s.size());

            auto nullable_string_type = std::make_shared<DB::DataTypeNullable>(std::make_shared<DB::DataTypeString>());
            column.category_unique = DB::DataTypeLowCardinality::createColumnUnique(*nullable_string_type, std::move(dict_column));

            chassert(py::hasattr(series.attr("cat"), "codes"));
            py::array codes = py::array(series.attr("cat").attr("codes"));
            column.row_count = static_cast<size_t>(codes.size());
            column.data = codes;
            column.tmp = codes;
            column.tmp.inc_ref();
            column.buf = const_cast<void *>(codes.data());
            column.stride = static_cast<size_t>(codes.strides(0));
            column.category_codes_type = py::str(codes.attr("dtype")).cast<std::string>();
            return;
        }
        else
        {
            py::array * cached = wrapper.getCachedColumnData(col_name);
            if (!cached)
                throw DB::Exception(DB::ErrorCodes::LOGICAL_ERROR, "Column {} not found in cache", col_name);
            column.row_count = static_cast<size_t>(cached->size());
            column.data = *cached;
            column.buf = const_cast<void *>(cached->data());
            column.stride = static_cast<size_t>(cached->strides(0));
            return;
        }
    }

    py::array array = series.attr("values");
    column.row_count = py::len(series);
    chassert(py::hasattr(array, "strides"));
    column.stride = array.attr("strides").attr("__getitem__")(0).cast<size_t>();

    if (column.row_count > 0)
    {
        auto elem_type = series.attr("iloc").attr("__getitem__")(0).attr("__class__").attr("__name__").cast<std::string>();
        if (elem_type == "str" || elem_type == "unicode")
        {
            column.data = array;
            column.buf = const_cast<void *>(array.data());
            return;
        }

        if (elem_type == "bytes" || elem_type == "object")
        {
            auto str_obj = series.attr("astype")(py::dtype("str"));
            array = str_obj.attr("values");
            column.data = array;
            column.tmp = array;
            column.tmp.inc_ref();
            column.buf = const_cast<void *>(array.data());
            return;
        }
    }

    py::object underlying_array = series.attr("array");
    if (py::hasattr(underlying_array, "_data") && py::hasattr(underlying_array, "_mask"))
    {
        py::array data_array = underlying_array.attr("_data");
        py::array mask_array = underlying_array.attr("_mask");
        column.data = data_array;
        column.buf = const_cast<void *>(data_array.data());
        column.stride = data_array.attr("strides").attr("__getitem__")(0).cast<size_t>();
        column.mask_stride = mask_array.attr("strides").attr("__getitem__")(0).cast<size_t>();
        column.registered_array = std::make_unique<DB::RegisteredArray>(mask_array);
        return;
    }

    column.data = array;
    column.buf = const_cast<void *>(array.data());
}

} // namespace CHDB
