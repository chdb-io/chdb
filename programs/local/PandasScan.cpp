#include "PandasScan.h"

#include <Columns/ColumnObject.h>
#include <DataTypes/Serializations/SerializationJSON.h>

using namespace DB;

namespace CHDB {

ColumnPtr PandasScan::scanObject(
    const DB::ColumnWrapper & col_wrap,
    const size_t offset,
    const size_t count,
    const FormatSettings & format_settings)
{
    auto & data_type = col_wrap.dest_type;
    auto column = data_type->createColumn();
    auto ** object_array = static_cast<PyObject **>(col_wrap.buf);

    auto serialization = data_type->doGetDefaultSerialization();

    for (size_t i = offset; i < offset + count; ++i)
    {
        auto * obj = object_array[i];

        if (PyList_Check(obj) || PyTuple_Check(obj) || PyDict_Check(obj))
        {
            py::gil_scoped_acquire acquire;
            String json_str = py::module::import("json").attr("dumps")(py::reinterpret_borrow<py::object>(obj)).cast<std::string>();
            serialization->deserializeTextImpl(column, json_str, format_settings);
        }
        else
        {
            column->insertDefault();
        }
    }

    return column;
}

} // namespace CHDB
