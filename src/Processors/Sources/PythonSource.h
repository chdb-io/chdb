#pragma once

#include <cstddef>
#include <Core/Block.h>

#include <Core/ExternalResultDescription.h>
#include <Processors/ISource.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <Poco/Logger.h>
#include "DataTypes/IDataType.h"

namespace DB
{

namespace py = pybind11;

class PyReader;

struct ColumnWrapper
{
    void * buf; // we may modify the data when cast it to PyObject **, so we need a non-const pointer
    size_t row_count;
    py::handle data;
    DataTypePtr dest_type;
    std::string py_type; //py::handle type, eg. numpy.ndarray;
    std::string row_format;
    std::string encoding; // utf8, utf16, utf32, etc.
    std::string name;
};

using PyObjectVec = std::vector<py::object>;
using PyObjectVecPtr = std::shared_ptr<PyObjectVec>;
using PyColumnVec = std::vector<ColumnWrapper>;
using PyColumnVecPtr = std::shared_ptr<PyColumnVec>;


class PythonSource : public ISource
{
public:
    PythonSource(py::object & data_source_, const Block & sample_block_, UInt64 max_block_size_, size_t stream_index, size_t num_streams);

    ~PythonSource() override = default;

    String getName() const override { return "Python"; }
    Chunk genChunk(size_t & num_rows, PyObjectVecPtr data);
    Chunk generate() override;


private:
    py::object & data_source; // Do not own the reference

    Block sample_block;
    PyColumnVecPtr column_cache;

    const UInt64 max_block_size;
    // Caller will only pass stream index and total stream count
    // to the constructor, we need to calculate the start offset and end offset.
    const size_t stream_index;
    const size_t num_streams;
    size_t cursor;
    size_t data_source_row_count;
    Poco::Logger * logger = &Poco::Logger::get("TableFunctionPython");
    ExternalResultDescription description;

    PyObjectVecPtr scanData(const py::object & data, const std::vector<std::string> & col_names, size_t & cursor, size_t count);
    Chunk scanDataToChunk();
    void destory(PyObjectVecPtr & data);
};
}
