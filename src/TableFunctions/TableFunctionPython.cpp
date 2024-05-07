#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <Interpreters/evaluateConstantExpression.h>
#include <Parsers/ASTFunction.h>
#include <Storages/StorageInMemoryMetadata.h>
#include <Storages/StoragePython.h>
#include <TableFunctions/TableFunctionFactory.h>
#include <TableFunctions/TableFunctionPython.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <Common/Exception.h>
#include <Common/logger_useful.h>

namespace DB
{

namespace ErrorCodes
{
extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
extern const int PY_OBJECT_NOT_FOUND;
extern const int PY_EXCEPTION_OCCURED;
}

void TableFunctionPython::parseArguments(const ASTPtr & ast_function, ContextPtr context)
{
    const auto & func_args = ast_function->as<ASTFunction &>();

    if (!func_args.arguments)
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH, "Table function 'python' must have arguments.");

    ASTs & args = func_args.arguments->children;

    if (args.size() != 1)
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH, "Python table requires 1 argument: PyReader object");

    auto py_reader_arg = evaluateConstantExpressionOrIdentifierAsLiteral(args[0], context);

    try
    {
        py::dict global_vars = py::globals();
        LOG_DEBUG(logger, "Globals content: {}", String(py::str(global_vars)));
        py::dict main_vars = py::reinterpret_borrow<py::dict>(py::module_::import("__main__").attr("__dict__").ptr());
        LOG_DEBUG(logger, "Main content: {}", String(py::str(main_vars)));

        // get the py_reader_arg without quotes
        auto py_reader_arg_str = py_reader_arg->as<ASTLiteral &>().value.safeGet<String>();
        LOG_DEBUG(logger, "PyReader object name: {}", py_reader_arg_str);

        // strip all quotes like '"` if any. eg. 'PyReader' -> PyReader, "PyReader" -> PyReader
        py_reader_arg_str.erase(
            std::remove_if(py_reader_arg_str.begin(), py_reader_arg_str.end(), [](char c) { return c == '\'' || c == '\"' || c == '`'; }),
            py_reader_arg_str.end());

        // try global_vars first, if not found, try main_vars
        py::object obj_by_name
            = global_vars.contains(py_reader_arg_str.data()) ? global_vars[py_reader_arg_str.data()] : main_vars[py_reader_arg_str.data()];

        // check if obj_by_name is a PyReader object or a subclass object of PyReader
        if (!py::isinstance<PyReader>(obj_by_name))
            throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH, "Python object is not a PyReader object");
        reader = std::dynamic_pointer_cast<PyReader>(obj_by_name.cast<std::shared_ptr<PyReader>>());
    }
    catch (py::error_already_set & e)
    {
        throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED, e.what());
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
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH, "Python reader not initialized");

    auto columns = getActualTableStructure(context, is_insert_query);

    auto storage
        = std::make_shared<StoragePython>(StorageID(getDatabaseName(), table_name), columns, ConstraintsDescription{}, reader, context);
    storage->startup();
    return storage;
}

ColumnsDescription TableFunctionPython::getActualTableStructure(ContextPtr /*context*/, bool /*is_insert_query*/) const
{
    return StoragePython::getTableStructureFromData(reader);
}

void registerTableFunctionPython(TableFunctionFactory & factory)
{
    factory.registerFunction<TableFunctionPython>(
        {.documentation
         = {.description = R"(
                Creates a table interface to a Python data source and reads data from a PyReader object.
                This table function requires a single argument which is a PyReader object used to read data from Python.
            )",
            .examples = {{"1", "SELECT * FROM Python(PyReader)", ""}}}},
        TableFunctionFactory::CaseInsensitive);
}

}
