#include "PandasDataFrameBuilder.h"
#include "PythonImporter.h"
#include "NumpyType.h"
#include "QueryResult.h"

#include <DataTypes/Serializations/SerializationNullable.h>
#include <DataTypes/DataTypeDateTime.h>
#include <DataTypes/DataTypeDateTime64.h>
#include <DataTypes/DataTypeTime.h>
#include <DataTypes/DataTypeTime64.h>
#include <DataTypes/DataTypeLowCardinality.h>
#include <Common/DateLUTImpl.h>
#include <Processors/Chunk.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeDate.h>
#include <base/Decimal.h>

namespace DB
{

namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
}

}

using namespace DB;

namespace CHDB
{

PandasDataFrameBuilder::PandasDataFrameBuilder(const ChunkQueryResult & chunk_result)
{
    chunks = std::move(const_cast<ChunkQueryResult &>(chunk_result).chunks);

    for (const auto & chunk : chunks)
        total_rows += chunk.getNumRows();

    if (total_rows)
    {
        if (!chunk_result.header)
            throw Exception(ErrorCodes::LOGICAL_ERROR, "ChunkQueryResult header is empty");

        const auto & sample = *chunk_result.header;
        column_names.reserve(sample.columns());
        column_types.reserve(sample.columns());

        std::unordered_map<String, size_t> name_map;

        for (const auto & column : sample)
        {
            const auto & col_name = column.name;
            String final_name = col_name;

            if (name_map.contains(col_name))
            {
                auto idx = name_map[col_name];
                final_name = col_name + "_" + std::to_string(idx);
                while (name_map.contains(final_name))
                {
                    ++name_map[col_name];
                    final_name = col_name + "_" + std::to_string(name_map[col_name]);
                }

                ++name_map[final_name];
            }
            else
            {
                name_map[col_name] = 1;
            }

            column_names.push_back(final_name);
            column_types.push_back(column.type);

            /// Record timezone for timezone-aware types
            if (const auto * dt = typeid_cast<const DataTypeDateTime *>(column.type.get()))
                column_timezones[final_name] = dt->getTimeZone().getTimeZone();
            else if (const auto * dt64 = typeid_cast<const DataTypeDateTime64 *>(column.type.get()))
                column_timezones[final_name] = dt64->getTimeZone().getTimeZone();
        }
    }

    finalize();
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

		if (py::isinstance(value, import_cache.numpy.ma.masked_array()))
        {
		    auto dtype = ConvertNumpyDtype(value);
			auto series = pandas.attr("Series")(value.attr("data"), py::arg("dtype") = dtype);
			series.attr("__setitem__")(value.attr("mask"), import_cache.pandas.NA());
			dict.attr("__setitem__")(key, series);
		}
	}

	auto df = pandas.attr("DataFrame").attr("from_dict")(dict);

	/// Apply timezone conversion for timezone-aware columns
	changeToTZType(df);

	return df;
}

void PandasDataFrameBuilder::changeToTZType(py::object & df)
{
    if (column_timezones.empty())
        return;

    for (const auto & [column_name, timezone_str] : column_timezones)
    {
        /// Check if column exists in DataFrame
        if (!df.attr("__contains__")(column_name).cast<bool>())
            continue;

        /// Get the column
        auto column = df[column_name.c_str()];

        /// First localize to UTC (assuming the timestamps are in UTC)
        auto utc_localized = column.attr("dt").attr("tz_localize")("UTC");

        /// Then convert to the target timezone
        auto tz_converted = utc_localized.attr("dt").attr("tz_convert")(timezone_str);

        /// Update the column in DataFrame
        df.attr("__setitem__")(column_name.c_str(), tz_converted);
    }
}

void PandasDataFrameBuilder::finalize()
{
    const auto types_size = column_types.size();
    columns_data.reserve(types_size);

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

        if (columns.size() != types_size)
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Chunk column size not match");

        for (size_t col_idx = 0; col_idx < columns.size(); ++col_idx)
        {
            auto column = columns[col_idx];

            columns_data[col_idx].append(column);
        }
    }

    chunks.clear();

    /// Create pandas DataFrame
    py::dict res;
	for (size_t col_idx = 0; col_idx < column_names.size(); ++col_idx) {
		auto & name = column_names[col_idx];
        auto & column_data = columns_data[col_idx];
        res[name.c_str()] = column_data.toArray();
	}
    final_dataframe = genDataFrame(res);
}

py::object PandasDataFrameBuilder::getDataFrame()
{
    return std::move(final_dataframe);
}

}
