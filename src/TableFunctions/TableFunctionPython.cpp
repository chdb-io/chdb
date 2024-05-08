#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <Interpreters/evaluateConstantExpression.h>
#include <Parsers/ASTFunction.h>
#include <Storages/StorageInMemoryMetadata.h>
#include <Storages/StoragePython.h>
#include <TableFunctions/TableFunctionFactory.h>
#include <TableFunctions/TableFunctionPython.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <Poco/Logger.h>
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

// Helper function to check if an object's class is or inherits from PyReader with a maximum depth
bool is_or_inherits_from_pyreader(const py::handle & obj, int depth = 3)
{
    // Base case: if depth limit reached, stop the recursion
    if (depth == 0)
        return false;

    // Check directly if obj is an instance of PyReader
    if (py::isinstance(obj, py::module_::import("chdb").attr("PyReader")))
        return true;

    // Check if obj's class or any of its bases is PyReader
    py::object cls = obj.attr("__class__");
    if (py::hasattr(cls, "__bases__"))
    {
        for (auto base : cls.attr("__bases__"))
            if (py::str(base.attr("__name__")).cast<std::string>() == "PyReader" || is_or_inherits_from_pyreader(base, depth - 1))
                return true;
    }
    return false;
}

// Function to find instances of PyReader or classes derived from PyReader, filtered by variable name
std::vector<py::object> find_instances_of_pyreader(const std::string & var_name)
{
    std::vector<py::object> instances;

    // Access the main module and its global dictionary
    py::dict globals = py::reinterpret_borrow<py::dict>(py::module_::import("__main__").attr("__dict__"));

    // Search in global scope
    if (globals.contains(var_name))
    {
        py::object obj = globals[var_name.data()];
        if (py::isinstance<py::object>(obj) && py::hasattr(obj, "__class__"))
        {
            if (is_or_inherits_from_pyreader(obj))
                instances.push_back(obj);
        }
    }
    if (!instances.empty())
        return instances;

    // Check objects in the garbage collector if nothing found, filtering by var_name
    // typically used to find objects that are not in the global scope, like in functions
    LOG_DEBUG(&Poco::Logger::get("TableFunctionPython"), "Searching for PyReader objects in the garbage collector");
    py::module_ gc = py::module_::import("gc");
    py::list all_objects = gc.attr("get_objects")();

    for (auto obj : all_objects)
    {
        if (py::isinstance<py::object>(obj) && py::hasattr(obj, "__class__"))
        {
            if (is_or_inherits_from_pyreader(obj) && py::str(obj.attr("__class__").attr("__name__")).cast<std::string>() == var_name)
            {
                if (std::find(instances.begin(), instances.end(), obj) == instances.end())
                    instances.push_back(obj.cast<py::object>());
            }
        }
    }

    return instances;
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
        // py::dict global_vars = py::globals();
        // LOG_DEBUG(logger, "Globals content: {}", String(py::str(global_vars)));
        // py::dict main_vars = py::reinterpret_borrow<py::dict>(py::module_::import("__main__").attr("__dict__").ptr());
        // LOG_DEBUG(logger, "Main content: {}", String(py::str(main_vars)));

        // get the py_reader_arg without quotes
        auto py_reader_arg_str = py_reader_arg->as<ASTLiteral &>().value.safeGet<String>();
        LOG_DEBUG(logger, "PyReader object name: {}", py_reader_arg_str);

        // strip all quotes like '"` if any. eg. 'PyReader' -> PyReader, "PyReader" -> PyReader
        py_reader_arg_str.erase(
            std::remove_if(py_reader_arg_str.begin(), py_reader_arg_str.end(), [](char c) { return c == '\'' || c == '\"' || c == '`'; }),
            py_reader_arg_str.end());

        auto instances = find_instances_of_pyreader(py_reader_arg_str);
        if (instances.empty())
            throw Exception(ErrorCodes::PY_OBJECT_NOT_FOUND, "PyReader object not found in the Python environment");
        if (instances.size() > 1)
            throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED, "Multiple PyReader objects found in the Python environment");

        LOG_DEBUG(
            logger,
            "PyReader object found in Python environment with name: {} type: {}",
            py_reader_arg_str,
            py::str(instances[0].attr("__class__")).cast<std::string>());

        reader = instances[0];
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
    py::gil_scoped_acquire acquire;
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
