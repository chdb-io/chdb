#pragma once

#include "DataSourceWrapper.h"
#include "PybindWrapper.h"
#include "PythonUtils.h"

#include <Storages/ColumnsDescription.h>

namespace CHDB
{

struct PandasBindColumn
{
public:
    PandasBindColumn(py::handle name, py::handle type, py::object column)
        : name(name), type(type), handle(std::move(column))
    {}

    py::handle name;
    py::handle type;
    py::object handle;
};

struct PandasDataFrameBind
{
public:
    explicit PandasDataFrameBind(const py::handle & df)
    {
        names = py::list(df.attr("columns"));
        types = py::list(df.attr("dtypes"));
        getter = df.attr("__getitem__");
    }

    PandasBindColumn operator[](size_t index) const {
        auto column = py::reinterpret_borrow<py::object>(getter(names[index]));
        auto type = types[index];
        auto name = names[index];
        return PandasBindColumn(name, type, column);
     }

public:
     py::list names;
     py::list types;

private:
    py::object getter;
};

class PandasDataFrame
{
public:
    static DB::ColumnsDescription getActualTableStructure(DataSourceWrapper & wrapper, DB::ContextPtr & context);

    static bool isPandasDataframe(const py::object & object);

    static bool isPyArrowBacked(const py::handle & object);

    static void fillColumn(
        const py::handle & data_source,
        const std::string & col_name,
        DB::ColumnWrapper & column,
        DataSourceWrapper & wrapper);
};

} // namespace CHDB
