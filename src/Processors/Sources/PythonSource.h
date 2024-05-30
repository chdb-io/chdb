#pragma once

#include <cstddef>
#include <Core/Block.h>

#include <Core/ExternalResultDescription.h>
#include <Processors/ISource.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <Poco/Logger.h>

namespace DB
{
class PyReader;

namespace py = pybind11;
class PythonSource : public ISource
{
public:
    PythonSource(py::object & data_source_, const Block & sample_block_, UInt64 max_block_size_, size_t stream_index, size_t num_streams);

    ~PythonSource() override = default;

    String getName() const override { return "Python"; }
    Chunk generate() override;


private:
    py::object & data_source; // Do not own the reference
    Block sample_block;
    const UInt64 max_block_size;
    // Caller will only pass stream index and total stream count
    // to the constructor, we need to calculate the start offset and end offset.
    const size_t stream_index;
    const size_t num_streams;
    size_t cursor;
    Poco::Logger * logger = &Poco::Logger::get("TableFunctionPython");
    ExternalResultDescription description;

    void destory(std::shared_ptr<std::vector<py::object>> & data);
};
}
