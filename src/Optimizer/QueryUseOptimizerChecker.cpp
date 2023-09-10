/*
 * Copyright (2022) Bytedance Ltd. and/or its affiliates
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <Optimizer/QueryUseOptimizerChecker.h>

#include <Analyzers/QueryAnalyzer.h>
#include <Analyzers/QueryRewriter.h>
#include <Interpreters/SegmentScheduler.h>
#include <Interpreters/SelectQueryOptions.h>
#include <Interpreters/getTableExpressions.h>
#include <Interpreters/misc.h>
#include <MergeTreeCommon/MergeTreeMetaBase.h>
#include <Parsers/ASTDumpInfoQuery.h>
#include <Parsers/ASTExplainQuery.h>
#include <Parsers/ASTSelectIntersectExceptQuery.h>
#include <Parsers/ASTWithElement.h>
#include <QueryPlan/QueryPlan.h>
#include <Storages/StorageDistributed.h>
#include <Storages/StorageView.h>
#include <Storages/StorageCnchHive.h>
//#include <Common/TestLog.h>

namespace DB
{
void changeDistributedStages(ASTPtr & node)
{
    if (!node)
        return;

    if (auto * select = node->as<ASTSelectQuery>())
    {
        auto & settings_ptr = select->settings();
        if (!settings_ptr)
            return;
        auto & ast = settings_ptr->as<ASTSetQuery &>();
        for (auto it = ast.changes.begin(); it != ast.changes.end(); ++it)
        {
            if (it->name == "enable_distributed_stages")
            {
                it->value = Field(false);
                return;
            }
        }
    }
    else
    {
        for (auto & child : node->children)
            changeDistributedStages(child);
    }
}
void turnOffOptimizer(ContextMutablePtr context, ASTPtr & node)
{
    SettingsChanges setting_changes;

    setting_changes.emplace_back("enable_optimizer", false);

    context->applySettingsChanges(setting_changes);
    changeDistributedStages(node);
}

bool QueryUseOptimizerChecker::check(ASTPtr & node, const ContextMutablePtr & context, bool use_distributed_stages)
{
    if (!node || (!context->getSettingsRef().enable_optimizer && !use_distributed_stages))
    {
        turnOffOptimizer(context, node);
        return false;
    }

    // Optimizer only work for Server.
    // for example INSERT INTO parallel_replicas_backup(d, x, u, s) SELECT d, x, u, s FROM parallel_replicas;
    // will execute query : INSERT INTO test.parallel_replicas_backup_4313395779120660490 (d, x, u, s) SELECT d, x, u, s FROM test.parallel_replicas )
    // will execute query : SELECT d, x, u, s FROM test.parallel_replicas_4313395779120660490
    // in worker.
    if (context->getApplicationType() != Context::ApplicationType::SERVER)
    {
        turnOffOptimizer(context, node);
        return false;
    }

    bool support = false;

    if (auto * explain = node->as<ASTExplainQuery>())
    {
        bool explain_plan = explain->getKind() == ASTExplainQuery::ExplainKind::OptimizerPlan
            || explain->getKind() == ASTExplainQuery::ExplainKind::QueryPlan
            || explain->getKind() == ASTExplainQuery::ExplainKind::QueryPipeline
            || explain->getKind() ==  ASTExplainQuery::AnalyzedSyntax
            || explain->getKind() ==  ASTExplainQuery::DistributedAnalyze
            || explain->getKind() ==  ASTExplainQuery::LogicalAnalyze
            || explain->getKind() ==  ASTExplainQuery::Distributed
            || explain->getKind() ==  ASTExplainQuery::TraceOptimizerRule
            || explain->getKind() ==  ASTExplainQuery::TraceOptimizer;
        support = explain_plan && check(explain->getExplainedQuery(), context);
    }
    else if (auto * dump = node->as<ASTDumpInfoQuery>())
    {
        return check(dump->dump_query, context);
    }
    else if (node->as<ASTSelectQuery>() || node->as<ASTSelectWithUnionQuery>() || node->as<ASTSelectIntersectExceptQuery>())
    {
        // disable system query, array join, table function, no merge tree table
        NameSet with_tables;

        QueryUseOptimizerContext query_with_plan_context{
            .context = context, .with_tables = with_tables, .external_tables = context->getExternalTables()};
        QueryUseOptimizerVisitor checker;
        try
        {
            support = ASTVisitorUtil::accept(node, checker, query_with_plan_context);
        }
        catch (Exception &)
        {
            //            if (e.code() != ErrorCodes::NOT_IMPLEMENTED)
            throw;
            //            support = false;
        }


        if (support && context->getSettingsRef().enable_optimizer_white_list)
        {
            QuerySupportOptimizerVisitor support_checker;
            try
            {
                support = ASTVisitorUtil::accept(node, support_checker, query_with_plan_context);
            }
            catch (Exception &)
            {
                //            if (e.code() != ErrorCodes::NOT_IMPLEMENTED)
                throw;
                //            support = false;
            }
        }
        if (!support)
            LOG_INFO(
                &Poco::Logger::get("QueryUseOptimizerChecker"), "query is unsupported for optimizer, reason: " + checker.getReason());
    }

    if (!support)
        turnOffOptimizer(context, node);

    return support;
}

bool QueryUseOptimizerVisitor::visitNode(ASTPtr & node, QueryUseOptimizerContext & context)
{
    for (auto & child : node->children)
    {
        if (!ASTVisitorUtil::accept(child, *this, context))
        {
            return false;
        }
    }
    return true;
}

static bool
checkDatabaseAndTable(const ASTTableExpression & table_expression, const ContextMutablePtr & context, const NameSet & with_tables)
{
    if (table_expression.database_and_table_name)
    {
        auto db_and_table = DatabaseAndTableWithAlias(table_expression.database_and_table_name);

        auto table_name = db_and_table.table;
        auto database_name = db_and_table.database;

        /// not with table
        if (!(database_name.empty() && with_tables.find(table_name) != with_tables.end()))
        {
            /// If the database is not specified - use the current database.
            auto table_id = context->tryResolveStorageID(table_expression.database_and_table_name);
            auto storage_table = DatabaseCatalog::instance().tryGetTable(table_id, context);
            if (database_name.empty() && !storage_table)
                database_name = context->getCurrentDatabase();

            if (!storage_table)
                return false;

            if (database_name == "system")
                return true;

            if (dynamic_cast<const StorageView *>(storage_table.get()))
            {
                auto table_metadata_snapshot = storage_table->getInMemoryMetadataPtr();
                auto subquery = table_metadata_snapshot->getSelectQuery().inner_query->clone();
                return QueryUseOptimizerChecker::check(subquery, context);
            }

            if (!dynamic_cast<const MergeTreeMetaBase *>(storage_table.get())
                && !dynamic_cast<const StorageCnchHive *>(storage_table.get()))
                return false;
        }
    }
    return true;
}

bool QueryUseOptimizerVisitor::visitASTSelectQuery(ASTPtr & node, QueryUseOptimizerContext & query_with_plan_context)
{
    auto * select = node->as<ASTSelectQuery>();
    const ContextMutablePtr & context = query_with_plan_context.context;

    if (select->group_by_with_totals)
    {
        reason = "group by with totals";
        return false;
    }

    NameSet old_with_tables = query_with_plan_context.with_tables;
    collectWithTableNames(*select, query_with_plan_context.with_tables);

    for (const auto * table_expression : getTableExpressions(*select))
    {
        if (!checkDatabaseAndTable(*table_expression, context, query_with_plan_context.with_tables))
        {
            reason = "unsupported storage";
            return false;
        }
        if (table_expression->table_function)
        {
            reason = "table function";
            return false;
        }
    }

    bool result = visitNode(node, query_with_plan_context);
    query_with_plan_context.with_tables = std::move(old_with_tables);
    return result;
}

bool QueryUseOptimizerVisitor::visitASTTableJoin(ASTPtr & node, QueryUseOptimizerContext & query_with_plan_context)
{
    return visitNode(node, query_with_plan_context);
}

bool QueryUseOptimizerVisitor::visitASTIdentifier(ASTPtr & node, QueryUseOptimizerContext & context)
{
    bool support = !context.external_tables.contains(node->as<ASTIdentifier>()->name());
    if (!support)
        reason = "external table";
    return support;
}

bool QueryUseOptimizerVisitor::visitASTFunction(ASTPtr & node, QueryUseOptimizerContext & query_with_plan_context)
{
    auto & fun = node->as<ASTFunction &>();
    if (fun.name == "untuple")
    {
        reason = "unsupported function";
        return false;
    }
    else if (functionIsInOrGlobalInOperator(fun.name) && fun.arguments->getChildren().size() == 2)
    {
        if (auto * identifier = fun.arguments->getChildren()[1]->as<ASTIdentifier>())
        {
            ASTTableExpression table_expression;
            table_expression.database_and_table_name = std::make_shared<ASTTableIdentifier>(identifier->name());
            if (!checkDatabaseAndTable(table_expression, query_with_plan_context.context, query_with_plan_context.with_tables))
            {
                reason = "unsupported storage";
                return false;
            }
        }
    }
    return visitNode(node, query_with_plan_context);
}

bool QueryUseOptimizerVisitor::visitASTQuantifiedComparison(ASTPtr & node, QueryUseOptimizerContext & query_with_plan_context)
{
    return visitNode(node, query_with_plan_context);
}

void QueryUseOptimizerVisitor::collectWithTableNames(ASTSelectQuery & query, NameSet & with_tables)
{
    if (auto with = query.with())
    {
        for (const auto & child : with->children)
            if (auto * with_elem = child->as<ASTWithElement>())
                with_tables.emplace(with_elem->name);
    }
}

bool QuerySupportOptimizerVisitor::visitNode(ASTPtr & node, QueryUseOptimizerContext & context)
{
    for (auto & child : node->children)
    {
        if (ASTVisitorUtil::accept(child, *this, context))
        {
            return true;
        }
    }
    return false;
}

bool QuerySupportOptimizerVisitor::visitASTTableJoin(ASTPtr &, QueryUseOptimizerContext &)
{
    return true;
}

bool QuerySupportOptimizerVisitor::visitASTSelectQuery(ASTPtr & node, QueryUseOptimizerContext & query_with_plan_context)
{
    auto * select = node->as<ASTSelectQuery>();

    if (select->groupBy())
        return true;

    return visitNode(node, query_with_plan_context);
}

bool QuerySupportOptimizerVisitor::visitASTSelectIntersectExceptQuery(ASTPtr & node, QueryUseOptimizerContext & query_with_plan_context)
{
    auto & intersect_or_except = node->as<ASTSelectIntersectExceptQuery &>();
    switch (intersect_or_except.final_operator)
    {
        case ASTSelectIntersectExceptQuery::Operator::INTERSECT_ALL:
        case ASTSelectIntersectExceptQuery::Operator::INTERSECT_DISTINCT:
        case ASTSelectIntersectExceptQuery::Operator::EXCEPT_ALL:
        case ASTSelectIntersectExceptQuery::Operator::EXCEPT_DISTINCT:
            return true;
        default:
            break;
    }
    return visitNode(node, query_with_plan_context);
}

bool QuerySupportOptimizerVisitor::visitASTSelectWithUnionQuery(ASTPtr & node, QueryUseOptimizerContext & query_with_plan_context)
{
    auto * select = node->as<ASTSelectWithUnionQuery>();

    switch (select->union_mode)
    {
        case ASTSelectWithUnionQuery::Mode::INTERSECT_ALL:
        case ASTSelectWithUnionQuery::Mode::INTERSECT_DISTINCT:
        case ASTSelectWithUnionQuery::Mode::EXCEPT_ALL:
        case ASTSelectWithUnionQuery::Mode::EXCEPT_DISTINCT:
            return true;
        case ASTSelectWithUnionQuery::Mode::UNION_ALL:
        case ASTSelectWithUnionQuery::Mode::UNION_DISTINCT:
        default:
            break;
    }
    return visitNode(node, query_with_plan_context);
}


bool QuerySupportOptimizerVisitor::visitASTFunction(ASTPtr & node, QueryUseOptimizerContext & query_with_plan_context)
{
    auto & fun = node->as<ASTFunction &>();
    static const std::set<String> distinct_func{"uniqexact", "countdistinct"};
    if (distinct_func.contains(Poco::toLower(fun.name)))
    {
        return true;
    }
    if (fun.is_window_function && query_with_plan_context.context->getSettingsRef().enable_optimizer_support_window)
    {
        return true;
    }
    return visitNode(node, query_with_plan_context);
}

bool QuerySupportOptimizerVisitor::visitASTQuantifiedComparison(ASTPtr &, QueryUseOptimizerContext &)
{
    return true;
}

}
