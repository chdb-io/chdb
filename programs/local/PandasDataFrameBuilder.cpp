#include "PandasDataFrameBuilder.h"
#include "NumpyType.h"
#include "PythonUtils.h"
#include "PythonConversion.h"
#include "PythonImporter.h"

#include <Columns/IColumn.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeDate.h>
#include <DataTypes/DataTypeDateTime.h>

using namespace CHDB;

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

PandasDataFrameBuilder::PandasDataFrameBuilder(const Block & sample)
{
    column_names.reserve(sample.columns());
    column_types.reserve(sample.columns());

    for (const auto & column : sample)
    {
        column_names.push_back(column.name);
        column_types.push_back(column.type);
    }
}

void PandasDataFrameBuilder::addChunk(const Chunk & chunk)
{
    if (chunk.hasRows())
    {
        chunks.push_back(chunk.clone());
        total_rows += chunk.getNumRows();
    }
}

py::object PandasDataFrameBuilder::genDataFrame(const py::handle & dict)
{
    auto & import_cache = PythonImporter::ImportCache();
	auto pandas = import_cache.pandas();
	if (!pandas)
    {
		throw Exception(ErrorCodes::LOGICAL_ERROR, "Pandas is not installed");
	}

	py::object items = dict.attr("items")();
	for (const py::handle & item : items) {
		auto key_value = py::cast<py::tuple>(item);
		py::handle key = key_value[0];
		py::handle value = key_value[1];

		auto dtype = ConvertNumpyDtype(value);
		if (py::isinstance(value, import_cache.numpy.ma.masked_array()))
        {
			auto series = pandas.attr("Series")(value.attr("data"), py::arg("dtype") = dtype);
			series.attr("__setitem__")(value.attr("mask"), import_cache.pandas.NA());
			dict.attr("__setitem__")(key, series);
		}
	}

	auto df = pandas.attr("DataFrame").attr("from_dict")(dict);
	return df;
}

void PandasDataFrameBuilder::finalize()
{
    if (is_finalized)
        return;

    columns_data.reserve(column_types.size());
    for (const auto & type : column_types)
    {
        columns_data.emplace_back(type);
    }

    for (auto & column_data : columns_data)
    {
        column_data.init(total_rows);
    }

    /// Process all chunks and append column data
    for (const auto & chunk : chunks)
    {
        const auto & columns = chunk.getColumns();
        for (size_t col_idx = 0; col_idx < columns.size(); ++col_idx)
        {
            columns_data[col_idx].append(columns[col_idx]);
        }
    }

    /// Create pandas DataFrame
    py::dict res;
	for (size_t col_idx = 0; col_idx < column_names.size(); ++col_idx) {
		auto & name = column_names[col_idx];
        auto & column_data = columns_data[col_idx];
        res[name.c_str()] = column_data.toArray();
	}
    final_dataframe = genDataFrame(res);

    is_finalized = true;
}

}
