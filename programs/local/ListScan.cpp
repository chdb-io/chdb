#include "ListScan.h"
#include "PythonConversion.h"

#include <Columns/ColumnNullable.h>
#include <Columns/ColumnObject.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/IDataType.h>
#include <DataTypes/Serializations/SerializationJSON.h>
#include <IO/WriteHelpers.h>
#include <Common/typeid_cast.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
    extern const int PY_EXCEPTION_OCCURED;
}

}

using namespace DB;

namespace CHDB {

ColumnPtr ListScan::scanObject(
    const ColumnWrapper & col_wrap,
    const size_t cursor,
    const size_t count,
    const FormatSettings & format_settings)
{
    innerCheck(col_wrap);

    const auto & data_type = col_wrap.dest_type;
    auto column = data_type->createColumn();
    auto nested_type = removeNullable(data_type);
    auto serialization = nested_type->getDefaultSerialization();

    innerScanObject(cursor, count, format_settings, serialization, col_wrap.data, column);

    return column;
}

void ListScan::scanObject(
    const size_t cursor,
    const size_t count,
    const FormatSettings & format_settings,
    const py::handle & obj,
    MutableColumnPtr & column)
{
    auto data_type = std::make_shared<DataTypeObject>(DataTypeObject::SchemaFormat::JSON);
    SerializationPtr serialization = data_type->getDefaultSerialization();

    innerScanObject(cursor, count, format_settings, serialization, obj, column);
}

void ListScan::innerScanObject(
    const size_t cursor,
    const size_t count,
    const FormatSettings & format_settings,
    SerializationPtr & serialization,
    const py::handle & obj,
    MutableColumnPtr & column)
{
    py::gil_scoped_acquire acquire;

    auto list = obj.cast<py::list>();

    auto & nullable_column = typeid_cast<ColumnNullable &>(*column);
    auto data_column = nullable_column.getNestedColumnPtr()->assumeMutable();
    auto & null_map = nullable_column.getNullMapData();

    for (size_t i = cursor; i < cursor + count; ++i)
    {
        auto item = list.attr("__getitem__")(i);

        if (!py::isinstance<py::dict>(item))
        {
            null_map.push_back(1);
            data_column->insertDefault();
            continue;
        }

        null_map.push_back(0);
        tryInsertJsonResult(item, format_settings, data_column, serialization);
    }
}

void ListScan::innerCheck(const ColumnWrapper & col_wrap)
{
    if (col_wrap.data.is_none())
        throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED, "Column data is None");

    if (!col_wrap.buf)
        throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED, "Column buffer is null");
}

} // namespace CHDB
