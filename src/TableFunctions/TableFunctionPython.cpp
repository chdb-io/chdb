#include <memory>
#include <TableFunctions/TableFunctionPython.h>

#if USE_PYTHON
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <Interpreters/evaluateConstantExpression.h>
#include <Parsers/ASTFunction.h>
#include <Storages/StorageInMemoryMetadata.h>
#include <Storages/StoragePython.h>
#include <TableFunctions/TableFunctionFactory.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <Poco/Logger.h>
#include <Common/Exception.h>
#include <Common/PythonUtils.h>
#include <Common/logger_useful.h>

namespace DB
{

namespace ErrorCodes
{
extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
extern const int PY_OBJECT_NOT_FOUND;
extern const int PY_EXCEPTION_OCCURED;
}

// Function to find instance of PyReader, pandas DataFrame, or PyArrow Table, filtered by variable name
py::object find_instances_of_pyreader(const std::string & var_name)
{
    py::module inspect = py::module_::import("inspect");
    py::object current_frame = inspect.attr("currentframe")();

    while (!current_frame.is_none())
    {
        auto local_dict = py::reinterpret_borrow<py::dict>(current_frame.attr("f_locals"));
        auto global_dict = py::reinterpret_borrow<py::dict>(current_frame.attr("f_globals"));

        for (const auto & dict : {local_dict, global_dict})
        {
            if (dict.contains(var_name))
            {
                py::object obj = dict[var_name.data()];
                if (isInheritsFromPyReader(obj) || isPandasDf(obj) || isPyarrowTable(obj))
                    return obj;
            }
        }

        current_frame = current_frame.attr("f_back");
    }

    // not found
    return py::none();
}

void TableFunctionPython::parseArguments(const ASTPtr & ast_function, ContextPtr context)
{
    py::gil_scoped_acquire acquire;
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

        auto instance = find_instances_of_pyreader(py_reader_arg_str);
        if (instance.is_none())
            throw Exception(
                ErrorCodes::PY_OBJECT_NOT_FOUND,
                "Python object not found in the Python environment\n"
                "Ensure that the object is type of PyReader, pandas DataFrame, or PyArrow Table and is in the global or local scope");

        LOG_DEBUG(
            logger,
            "Python object found in Python environment with name: {} type: {}",
            py_reader_arg_str,
            py::str(instance.attr("__class__")).cast<std::string>());

        reader = instance;
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

ColumnsDescription TableFunctionPython::getActualTableStructure(ContextPtr /*context*/, bool /*is_insert_query*/) const
{
    py::gil_scoped_acquire acquire;
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
        TableFunctionFactory::CaseInsensitive);
}

}
#endif
