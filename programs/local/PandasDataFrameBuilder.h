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

/// Builder class to convert ClickHouse Chunks to Pandas DataFrame
/// Accumulates chunks and provides conversion to Python pandas DataFrame object
class PandasDataFrameBuilder
{
public:
    explicit PandasDataFrameBuilder(const DB::Block & sample);

    ~PandasDataFrameBuilder() = default;

    /// Add data chunk
    void addChunk(const DB::Chunk & chunk);

    /// Finalize and build pandas DataFrame from all collected chunks
    void finalize();

    /// Get the finalized pandas DataFrame
    pybind11::object getDataFrame();

private:
    pybind11::object genDataFrame(const pybind11::handle & dict);
    void changeToTZType(pybind11::object & df);

    std::vector<String> column_names;
    std::vector<DB::DataTypePtr> column_types;

    /// Map column name to timezone string for timezone-aware types
    std::unordered_map<String, String> column_timezones;

    std::vector<DB::Chunk> chunks;
    std::vector<CHDB::NumpyArray> columns_data;

    size_t total_rows = 0;
    bool is_finalized = false;
    pybind11::object final_dataframe;

    Poco::Logger * log = &Poco::Logger::get("PandasDataFrameBuilder");
};

}
