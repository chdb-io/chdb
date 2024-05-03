#pragma once

#include <memory>
#include <string>
#include <vector>
#include <Storages/IStorage.h>
#include <Storages/StorageFactory.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Common/Exception.h>


namespace DB
{

namespace py = pybind11;
class PyReader
{
public:
    explicit PyReader(const py::object & data) : data(data) { }
    virtual ~PyReader() = default;

    virtual std::vector<py::object> read(const std::vector<std::string> & col_names, int count) = 0;

protected:
    py::object data;
};

// Trampoline class
// Zsee: https://pybind11.readthedocs.io/en/stable/advanced/classes.html#trampolines
class PyReaderTrampoline : public PyReader
{
public:
    using PyReader::PyReader; // Inherit constructors

    std::vector<py::object> read(const std::vector<std::string> & col_names, int count) override
    {
        PYBIND11_OVERRIDE_PURE(
            std::vector<py::object>, // Return type List[object]
            PyReader,   // Parent class
            read,       // Name of the function in C++ (must match Python name)
            col_names,  // Argument(s)
            count);
    }
};

class StoragePython : public IStorage, public WithContext
{
    std::shared_ptr<PyReader> reader;

public:
    StoragePython(
        const StorageID & table_id_,
        const ColumnsDescription & columns_,
        const ConstraintsDescription & constraints_,
        std::shared_ptr<PyReader> reader_,
        ContextPtr context_);

    std::string getName() const override { return "Python"; }

    Pipe read(
        const Names & column_names,
        const StorageSnapshotPtr & storage_snapshot,
        SelectQueryInfo & query_info,
        ContextPtr context_,
        QueryProcessingStage::Enum processed_stage,
        size_t max_block_size,
        size_t num_streams) override;

    Block prepareSampleBlock(const Names & column_names, const StorageSnapshotPtr & storage_snapshot);
};

void registerStoragePython(StorageFactory & factory);


}
