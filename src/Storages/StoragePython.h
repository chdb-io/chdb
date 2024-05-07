#pragma once

#include <memory>
#include <string>
#include <vector>
#include <Storages/ColumnsDescription.h>
#include <Storages/IStorage.h>
#include <Storages/StorageFactory.h>
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <Common/Exception.h>


namespace DB
{

namespace py = pybind11;
class PyReader
{
public:
    explicit PyReader(const py::object & data) : data(data) { }
    virtual ~PyReader() = default;

    // Read `count` rows from the data, and return a list of columns
    // chdb todo: maybe return py::list is better, but this is just a shallow copy
    virtual std::vector<py::object> read(const std::vector<std::string> & col_names, int count) = 0;

    // Return a vector of column names and their types, as a list of pairs.
    // The order is important, and should match the order of the data.
    // This is the default implementation, which trys to infer the schema from the every first row
    // of this.data column.
    // The logic is:
    //  1. If the data is a map with column names as keys and column data as values, then we use
    //    the key and type of every first element in the value list.
    //      eg:
    //          d = {'a': [1, 2, 3], 'b': ['x', 'y', 'z'], 'c': [1.0, 1e10, 1.2e100]}
    //          schema = {name: repr(type(value[0])) for name, value in d.items()}
    //      out:
    //          schema = {'a': "<class 'int'>", 'b': "<class 'str'>", 'c': "<class 'float'>"}
    //  2. If the data is a Pandas DataFrame, then we use the column names and dtypes.
    //    We use the repr of the dtype, which is a string representation of the dtype.
    //      eg:
    //          df = pd.DataFrame(d)
    //          schema = {name: repr(dtype) for name, dtype in zip(df.columns, df.dtypes)}
    //      out:
    //          schema = {'a': "dtype('int64')", 'b': "dtype('O')", 'c': "dtype('float64')"}
    //      Note:
    //          1. dtype('O') means object type, which is a catch-all for any types. we just treat it as string.
    //          2. the dtype of a Pandas DataFrame is a numpy.dtype object, which is not a Python type object.
    //
    //      When using Pandas >= 2.0, we can use the pyarrow as dtype_backend:
    //      eg:
    //          df_arr = pd.read_json('{"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.0, 1.111, 2.222]}', dtype_backend="pyarrow")
    //          schema = {name: repr(dtype) for name, dtype in zip(df_arr.columns, df_arr.dtypes)}
    //      out:
    //          schema = {'a': 'int64[pyarrow]', 'b': 'string[pyarrow]', 'c': 'double[pyarrow]'}
    //  3. if the data is a Pyarrow Table, then we use the column names and types.
    //      eg:
    //          tbl = pa.Table.from_pandas(df)
    //          schema = {field.name: repr(field.type) for field in tbl.schema}
    //      out:
    //          schema = {'a': 'DataType(int64)', 'b': 'DataType(string)', 'c': 'DataType(double)'}
    //  4. User can override this function to provide a more accurate schema.
    //      eg: "DataTypeUInt8", "DataTypeUInt16", "DataTypeUInt32", "DataTypeUInt64", "DataTypeUInt128", "DataTypeUInt256",
    //      "DataTypeInt8", "DataTypeInt16", "DataTypeInt32", "DataTypeInt64", "DataTypeInt128", "DataTypeInt256",
    //      "DataTypeFloat32", "DataTypeFloat64", "DataTypeString",

    std::vector<std::pair<std::string, std::string>> getSchema()
    {
        std::vector<std::pair<std::string, std::string>> schema;

        if (py::isinstance<py::dict>(data))
        {
            // If the data is a Python dictionary
            for (auto item : data.cast<py::dict>())
            {
                std::string key = py::str(item.first).cast<std::string>();
                py::list values = py::cast<py::list>(item.second);
                std::string dtype = py::str(values[0].attr("__class__").attr("__name__")).cast<std::string>();
                if (!values.empty())
                    schema.emplace_back(key, dtype);
            }
        }
        else if (py::hasattr(data, "dtypes"))
        {
            // If the data is a Pandas DataFrame
            py::object dtypes = data.attr("dtypes");
            py::list columns = data.attr("columns");
            for (size_t i = 0; i < py::len(columns); ++i)
            {
                std::string name = py::str(columns[i]).cast<std::string>();
                std::string dtype = py::str(py::repr(dtypes[columns[i]])).cast<std::string>();
                schema.emplace_back(name, dtype);
            }
        }
        else if (py::hasattr(data, "schema"))
        {
            // If the data is a Pyarrow Table
            py::object schema_fields = data.attr("schema").attr("fields");
            for (auto field : schema_fields)
            {
                std::string name = py::str(field.attr("name")).cast<std::string>();
                std::string dtype = py::str(py::repr(field.attr("type"))).cast<std::string>();
                schema.emplace_back(name, dtype);
            }
        }
        return schema;
    }

protected:
    py::object data;
};

// Trampoline class
// see: https://pybind11.readthedocs.io/en/stable/advanced/classes.html#trampolines
class PyReaderTrampoline : public PyReader
{
public:
    using PyReader::PyReader; // Inherit constructors

    // Just forward the virtual function call to Python
    std::vector<py::object> read(const std::vector<std::string> & col_names, int count) override
    {
        PYBIND11_OVERRIDE_PURE(
            std::vector<py::object>, // Return type List[object]
            PyReader, // Parent class
            read, // Name of the function in C++ (must match Python name)
            col_names, // Argument(s)
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

    static ColumnsDescription getTableStructureFromData(std::shared_ptr<PyReader> reader);

private:
    Poco::Logger * logger = &Poco::Logger::get("StoragePython");
};

void registerStoragePython(StorageFactory & factory);


}
