#include "StoragePython.h"

#include "PandasDataframe.h"
#include "TableFunctionPython.h"

#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <Interpreters/evaluateConstantExpression.h>
#include <Parsers/ASTFunction.h>
#include <Storages/StorageInMemoryMetadata.h>
#include <TableFunctions/TableFunctionFactory.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <Poco/Logger.h>
#include <Common/Exception.h>
#include "PythonUtils.h"
#include <Common/logger_useful.h>

using namespace CHDB;

namespace py = pybind11;
// Global storage for Python Table Engine queriable object
py::handle global_query_obj = nullptr;

namespace DB
{

namespace ErrorCodes
{
extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
extern const int PY_OBJECT_NOT_FOUND;
extern const int PY_EXCEPTION_OCCURED;
}

// Function to find instance of PyReader, pandas DataFrame, or PyArrow Table, filtered by variable name
py::object findQueryableObj(const std::string & var_name)
{
    py::module inspect = py::module_::import("inspect");
    py::object current_frame = inspect.attr("currentframe")();

    while (!current_frame.is_none())
    {
        // Get f_locals and f_globals
        py::object locals_obj = current_frame.attr("f_locals");
        py::object globals_obj = current_frame.attr("f_globals");

        // For each namespace (locals and globals)
        for (const auto & namespace_obj : {locals_obj, globals_obj})
        {
            // Use Python's __contains__ method to check if the key exists
            // This works with both regular dicts and FrameLocalsProxy (Python 3.13+)
            if (py::bool_(namespace_obj.attr("__contains__")(var_name)))
            {
                py::object obj;
                try
                {
                    // Get the object using Python's indexing syntax
                    obj = namespace_obj[py::cast(var_name)];
                    if (isInheritsFromPyReader(obj) || isPandasDf(obj) || isPyarrowTable(obj) || hasGetItem(obj))
                    {
                        return obj;
                    }
                }
                catch (const py::error_already_set &)
                {
                    continue; // If getting the value fails, continue to the next namespace
                }
            }
        }

        // Move to the parent frame
        current_frame = current_frame.attr("f_back");
    }

    // Object not found
    return py::none();
}

void TableFunctionPython::parseArguments(const ASTPtr & ast_function, ContextPtr context)
{
    // py::gil_scoped_acquire acquire;
    const auto & func_args = ast_function->as<ASTFunction &>();

    if (!func_args.arguments)
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH, "Table function 'python' must have arguments.");

    ASTs & args = func_args.arguments->children;

    if (args.size() != 1)
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH, "Python table requires 1 argument: PyReader object");

    auto py_reader_arg = evaluateConstantExpressionOrIdentifierAsLiteral(args[0], context);

    try
    {
        // get the py_reader_arg without quotes
        auto py_reader_arg_str = py_reader_arg->as<ASTLiteral &>().value.safeGet<String>();
        LOG_DEBUG(logger, "Python object name: {}", py_reader_arg_str);

        // strip all quotes like '"` if any. eg. 'PyReader' -> PyReader, "PyReader" -> PyReader
        py_reader_arg_str.erase(
            std::remove_if(py_reader_arg_str.begin(), py_reader_arg_str.end(), [](char c) { return c == '\'' || c == '\"' || c == '`'; }),
            py_reader_arg_str.end());

        auto instance = global_query_obj;
        if (instance == nullptr || instance.is_none())
            throw Exception(
                ErrorCodes::PY_OBJECT_NOT_FOUND,
                "Python object not found in the Python environment\n"
                "Ensure that the object is type of PyReader, pandas DataFrame, or PyArrow Table and is in the global or local scope");

        LOG_DEBUG(
            logger,
            "Python object found in Python environment with name: {} type: {}",
            py_reader_arg_str,
            py::str(instance.attr("__class__")).cast<std::string>());
        py::gil_scoped_acquire acquire;
        reader = instance.cast<py::object>();
    }
    catch (py::error_already_set & e)
    {
        throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED, "Python exception occured: {}", e.what());
    }
}

StoragePtr TableFunctionPython::executeImpl(
    const ASTPtr & /*ast_function*/,
    ContextPtr context,
    const String & table_name,
    ColumnsDescription /*cached_columns*/,
    bool is_insert_query) const
{
    if (!reader)
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH, "Python data source not initialized");

    auto columns = getActualTableStructure(context, is_insert_query);

    std::shared_ptr<StoragePython> storage;
    {
        py::gil_scoped_acquire acquire;
        storage = std::make_shared<StoragePython>(
            StorageID(getDatabaseName(), table_name), columns, ConstraintsDescription{}, reader, context);
    }
    storage->startup();
    return storage;
}

ColumnsDescription TableFunctionPython::getActualTableStructure(ContextPtr context, bool /*is_insert_query*/) const
{
    py::gil_scoped_acquire acquire;

    if (PandasDataFrame::isPandasDataframe(reader))
        return PandasDataFrame::getActualTableStructure(reader, context);

    return StoragePython::getTableStructureFromData(reader);
}

void registerTableFunctionPython(TableFunctionFactory & factory)
{
    factory.registerFunction<TableFunctionPython>(
        {.documentation
         = {.description = R"(
Passing Pandas DataFrame or Pyarrow Table to ClickHouse engine.
For any other data structure, you can also create a table interface to a Python data source and reads data 
from a PyReader object.
This table function requires a single argument which is a PyReader object used to read data from Python.
)",
            .examples = {{"1", "SELECT * FROM Python(PyReader)", ""}}}},
        TableFunctionFactory::Case::Insensitive);
}

}
