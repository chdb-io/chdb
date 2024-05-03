#pragma once

#include <memory>
#include <Core/Block.h>

#include <Core/ExternalResultDescription.h>
#include <Processors/ISource.h>
#include <Poco/Logger.h>

namespace DB
{
class PyReader;

class PythonSource : public ISource
{
    std::shared_ptr<PyReader> reader;
    Block sample_block;
    const UInt64 max_block_size;

public:
    PythonSource(std::shared_ptr<PyReader> reader_, const Block & sample_block_, UInt64 max_block_size_);

    String getName() const override { return "Python"; }
    Chunk generate() override;

private:
    Poco::Logger * logger = &Poco::Logger::get("TableFunctionPython");
    ExternalResultDescription description;
};
}
