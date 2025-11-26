#pragma once

#include "PybindWrapper.h"
#include "NumpyArray.h"

#include <Core/Block.h>
#include <Processors/Chunk.h>
#include <DataTypes/IDataType.h>
#include <Common/logger_useful.h>
#include <unordered_map>

namespace CHDB
{

class ChunkQueryResult;

class PandasDataFrameBuilder
{
public:
    explicit PandasDataFrameBuilder(const ChunkQueryResult & chunk_result);

    ~PandasDataFrameBuilder() = default;

    pybind11::object getDataFrame();

private:
    pybind11::object genDataFrame(const pybind11::handle & dict);
    void changeToTZType(pybind11::object & df);
    void finalize();

    std::vector<String> column_names;
    std::vector<DB::DataTypePtr> column_types;

    /// Map column name to timezone string for timezone-aware types
    std::unordered_map<String, String> column_timezones;

    std::vector<DB::Chunk> chunks;
    std::vector<CHDB::NumpyArray> columns_data;

    size_t total_rows = 0;
    pybind11::object final_dataframe;

    Poco::Logger * log = &Poco::Logger::get("PandasDataFrameBuilder");
};

}
