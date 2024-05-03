#include <Columns/ColumnString.h>
#include <Columns/IColumn.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <Processors/Sources/PythonSource.h>
#include <Storages/StoragePython.h>
#include <pybind11/gil.h>
#include <pybind11/pytypes.h>
#include <Common/Exception.h>
#include <Common/logger_useful.h>

namespace DB
{
PythonSource::PythonSource(std::shared_ptr<PyReader> reader_, const Block & sample_block_, const UInt64 max_block_size_)
    : ISource(sample_block_.cloneEmpty()), reader(std::move(reader_)), max_block_size(max_block_size_)
{
    description.init(sample_block_);
}

template <typename T>
ColumnPtr convert_and_insert(py::object obj)
{
    auto column = ColumnVector<T>::create();
    // if obj is a list
    if (py::isinstance<py::list>(obj))
    {
        py::list list = obj.cast<py::list>();
        for (auto && i : list)
            column->insert(i.cast<T>());
        // free the list
        list.dec_ref();
    }
    else if (py::isinstance<py::array>(obj)) // if obj is a numpy array
    {
        py::array array = obj.cast<py::array>();
        //chdb: array is a numpy array, so we can directly cast it to a vector?
        for (auto && i : array)
            column->insert(i.cast<T>());
        // free the array, until we implement with zero copy
        array.dec_ref();
    }
    else
    {
        throw Exception(ErrorCodes::BAD_TYPE_OF_FIELD, "Unsupported type {}", obj.get_type().attr("__name__").cast<std::string>());
    }
    return column;
}

template <>
ColumnPtr convert_and_insert<String>(py::object obj)
{
    auto column = ColumnString::create();
    if (py::isinstance<py::list>(obj))
    {
        py::list list = obj.cast<py::list>();
        for (auto && i : list)
            column->insert(i.cast<String>());
        // free the list
        list.dec_ref();
    }
    else if (py::isinstance<py::array>(obj))
    {
        py::array array = obj.cast<py::array>();
        for (auto && i : array)
            column->insert(i.cast<String>());
        // free the array, until we implement with zero copy
        array.dec_ref();
    }
    else
    {
        throw Exception(ErrorCodes::BAD_TYPE_OF_FIELD, "Unsupported type {}", obj.get_type().attr("__name__").cast<std::string>());
    }
    return column;
}

Chunk PythonSource::generate()
{
    size_t num_rows = 0;

    try
    {
        // GIL is held when called from Python code. Release it to avoid deadlock
        py::gil_scoped_release release;
        std::vector<py::object> data = reader->read(description.sample_block.getNames(), max_block_size);

        LOG_DEBUG(logger, "Read {} columns", data.size());
        LOG_DEBUG(logger, "Need {} columns", description.sample_block.columns());
        LOG_DEBUG(logger, "Max block size: {}", max_block_size);

        py::gil_scoped_acquire acquire;

        // if log level is debug, print all the data
        if (logger->debug())
        {
            // print all the data
            for (auto && col : data)
            {
                if (py::isinstance<py::list>(col))
                {
                    py::list list = col.cast<py::list>();
                    for (auto && i : list)
                        LOG_DEBUG(logger, "Data: {}", py::str(i).cast<std::string>());
                }
                else if (py::isinstance<py::array>(col))
                {
                    py::array array = col.cast<py::array>();
                    for (auto && i : array)
                        LOG_DEBUG(logger, "Data: {}", py::str(i).cast<std::string>());
                }
                else
                {
                    LOG_DEBUG(logger, "Data: {}", py::str(col).cast<std::string>());
                }
            }
        }

        Columns columns(description.sample_block.columns());
        // fill in the columns
        for (size_t i = 0; i < data.size(); ++i)
        {
            if (i == 0)
                num_rows = py::len(data[i]);
            const auto & column = data[i];
            const auto & type = description.sample_block.getByPosition(i).type;

            if (type->equals(*std::make_shared<DataTypeUInt8>()))
                columns[i] = convert_and_insert<UInt8>(column);
            else if (type->equals(*std::make_shared<DataTypeUInt16>()))
                columns[i] = convert_and_insert<UInt16>(column);
            else if (type->equals(*std::make_shared<DataTypeUInt32>()))
                columns[i] = convert_and_insert<UInt32>(column);
            else if (type->equals(*std::make_shared<DataTypeUInt64>()))
                columns[i] = convert_and_insert<UInt64>(column);
            else if (type->equals(*std::make_shared<DataTypeInt8>()))
                columns[i] = convert_and_insert<Int8>(column);
            else if (type->equals(*std::make_shared<DataTypeInt16>()))
                columns[i] = convert_and_insert<Int16>(column);
            else if (type->equals(*std::make_shared<DataTypeInt32>()))
                columns[i] = convert_and_insert<Int32>(column);
            else if (type->equals(*std::make_shared<DataTypeInt64>()))
                columns[i] = convert_and_insert<Int64>(column);
            else if (type->equals(*std::make_shared<DataTypeFloat32>()))
                columns[i] = convert_and_insert<Float32>(column);
            else if (type->equals(*std::make_shared<DataTypeFloat64>()))
                columns[i] = convert_and_insert<Float64>(column);
            else if (type->equals(*std::make_shared<DataTypeString>()))
                columns[i] = convert_and_insert<String>(column);
            else
                throw Exception(ErrorCodes::BAD_TYPE_OF_FIELD, "Unsupported type {}", type->getName());
        }
        // Set data vector to empty to avoid trigger py::object destructor without GIL
        // Note: we have already manually decremented the reference count of the list or array in `convert_and_insert` function
        data.clear();
        if (num_rows == 0)
            return {};

        return Chunk(std::move(columns), num_rows);
    }
    catch (const std::exception & e)
    {
        py::gil_scoped_release release;
        throw Exception(ErrorCodes::LOGICAL_ERROR, e.what());
    }
}


}
