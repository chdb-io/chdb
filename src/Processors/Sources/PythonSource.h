#pragma once

#include <Core/Block.h>

#include <Core/ExternalResultDescription.h>
#include <Processors/ISource.h>
#include <pybind11/pybind11.h>
#include <Poco/Logger.h>

namespace DB
{
class PyReader;

namespace py = pybind11;
class PythonSource : public ISource
{
public:
    PythonSource(py::object reader_, const Block & sample_block_, UInt64 max_block_size_);
    ~PythonSource() override
    {
        // Acquire the GIL before destroying the reader object
        py::gil_scoped_acquire acquire;
        reader.dec_ref();
        reader.release();
    }

    String getName() const override { return "Python"; }
    Chunk generate() override;

private:
    py::object reader;
    Block sample_block;
    const UInt64 max_block_size;
    Poco::Logger * logger = &Poco::Logger::get("TableFunctionPython");
    ExternalResultDescription description;
};
}
