#include "TableFunctionArrowStream.h"
#include "ArrowSchema.h"
#include "ArrowStreamWrapper.h"
#include "StorageArrowStream.h"

#include <TableFunctions/TableFunctionFactory.h>
#include <Parsers/ASTFunction.h>
#include <Parsers/ASTLiteral.h>
#include <Interpreters/evaluateConstantExpression.h>
#include <arrow/c/abi.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int UNKNOWN_IDENTIFIER;
    extern const int BAD_ARGUMENTS;
}

void TableFunctionArrowStream::parseArguments(const ASTPtr & ast_function, ContextPtr context)
{
    const auto & func_args = ast_function->as<ASTFunction &>();

    if (!func_args.arguments)
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH,
                       "Table function 'arrowstream' must have arguments.");

    ASTs & args = func_args.arguments->children;

    if (args.size() != 1)
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH,
                       "ArrowStream table requires 1 argument: stream name");

    auto stream_name_arg = evaluateConstantExpressionOrIdentifierAsLiteral(args[0], context);

    try
    {
        stream_name = stream_name_arg->as<ASTLiteral &>().value.safeGet<String>();

        stream_name.erase(
            std::remove_if(stream_name.begin(), stream_name.end(),
                          [](char c) { return c == '\'' || c == '\"' || c == '`'; }),
            stream_name.end());

        auto stream_opt = CHDB::ArrowStreamRegistry::instance().getArrowStream(stream_name);
        if (!stream_opt)
        {
            throw Exception(ErrorCodes::UNKNOWN_IDENTIFIER,
                           "ArrowStream '{}' not found in registry. "
                           "Please register it first using chdb_arrow_scan.",
                           stream_name);
        }

        stream_info = *stream_opt;
    }
    catch (const Exception &)
    {
        throw;
    }
    catch (const std::exception & e)
    {
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "Error parsing arrowstream argument: {}", e.what());
    }
    catch (...)
    {
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "Error parsing arrowstream argument");
    }
}

StoragePtr TableFunctionArrowStream::executeImpl(
    const ASTPtr & /*ast_function*/,
    ContextPtr context,
    const String & table_name,
    ColumnsDescription /*cached_columns*/,
    bool is_insert_query) const
{
    if (stream_name.empty() || !stream_info.stream)
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "ArrowStream name not initialized");

    auto columns = getActualTableStructure(context, is_insert_query);

    auto storage = std::make_shared<StorageArrowStream>(
        StorageID(getDatabaseName(), table_name),
        stream_info,
        columns,
        context);

    storage->startup();
    return storage;
}

ColumnsDescription TableFunctionArrowStream::getActualTableStructure(
    ContextPtr context, bool /*is_insert_query*/) const
{
    auto * arrow_stream = reinterpret_cast<ArrowArrayStream *>(stream_info.stream);
    CHDB::ArrowSchemaWrapper schema;

    if (arrow_stream->get_schema(arrow_stream, &schema.arrow_schema) != 0)
    {
        throw Exception(ErrorCodes::BAD_ARGUMENTS,
                        "Failed to get schema from ArrowStream '{}'", stream_name);
    }

    NamesAndTypesList names_and_types;
    CHDB::ArrowSchemaWrapper::convertArrowSchema(schema, names_and_types, context);

    return ColumnsDescription(names_and_types);
}

void registerTableFunctionArrowStream(TableFunctionFactory & factory)
{
    factory.registerFunction<TableFunctionArrowStream>(
        {.documentation = {
            .description = R"(
Creates a table from a registered ArrowStream.
This table function requires a single argument which is the name of a registered ArrowStream.
Use chdb_arrow_register_table() to register ArrowStreams first.
)",
            .examples = {{"arrowstream", "SELECT * FROM arrowstream('my_data')", ""}},
            .category = FunctionDocumentation::Category::TableFunction
        }},
        TableFunctionFactory::Case::Insensitive);
}

}
