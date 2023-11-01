#include <Interpreters/DatabaseCatalog.h>
#include <Interpreters/ExpressionAnalyzer.h>
#include <Interpreters/StorageID.h>
#include <Interpreters/TreeRewriter.h>
#include <Interpreters/getTableExpressions.h>
#include <Optimizer/SelectQueryInfoHelper.h>
#include <Parsers/ASTSelectQuery.h>
#include <Parsers/ASTTablesInSelectQuery.h>
#include <Storages/IStorage.h>
#include <Common/Exception.h>

namespace DB
{

SelectQueryInfo buildSelectQueryInfoForQuery(const ASTPtr & query, ContextPtr context)
{
    SelectQueryInfo query_info;
    query_info.query = query;

    const auto * select_query = query->as<ASTSelectQuery>();
    if (!select_query)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Unexpected AST type found in buildSelectQueryInfoForQuery");

    StoragePtr storage;
    StorageMetadataPtr metadata_snapshot;

    const auto * table_expression = getTableExpression(*select_query, 0);
    if (table_expression && table_expression->database_and_table_name)
    {
        storage = DatabaseCatalog::instance().getTable(StorageID{table_expression->database_and_table_name}, context);
        metadata_snapshot = storage->getInMemoryMetadataPtr();
    }

    // fill syntax_analyzer_result
    query_info.syntax_analyzer_result
        = TreeRewriter(context).analyzeSelect(query_info.query, TreeRewriterResult({}, storage, metadata_snapshot));

    // fill prepared_set
    auto query_analyzer
        = std::make_unique<SelectQueryExpressionAnalyzer>(query_info.query, query_info.syntax_analyzer_result, context, metadata_snapshot);

    query_analyzer->makeSetsForIndex(select_query->where());
    query_analyzer->makeSetsForIndex(select_query->prewhere());
    query_info.sets = std::move(query_analyzer->getPreparedSets());

    return query_info;
}

}
