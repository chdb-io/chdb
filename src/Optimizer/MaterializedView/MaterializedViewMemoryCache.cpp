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

#include <Optimizer/MaterializedView/MaterializedViewMemoryCache.h>

#include <Analyzers/QueryAnalyzer.h>
#include <Analyzers/QueryRewriter.h>
#include <Interpreters/InterpreterSelectQueryUseOptimizer.h>
#include <Interpreters/SegmentScheduler.h>
#include <Optimizer/Iterative/IterativeRewriter.h>
#include <Optimizer/PlanOptimizer.h>
#include <Optimizer/Rewriter/PredicatePushdown.h>
#include <Optimizer/Rule/Rules.h>
#include <QueryPlan/QueryPlanner.h>
#include <Storages/StorageMaterializedView.h>
#include <Storages/StorageDistributed.h>

namespace DB
{
class MaterializedViewMemoryCache::LocalTableRewriter
{
public:
    using Visitor = InDepthNodeVisitor<LocalTableRewriter, true>;

    struct Data
    {
        const std::map<String, StorageID> & table_to_distributed_table;

        explicit Data(const std::map<String, StorageID> & table_to_distributed_table_)
            : table_to_distributed_table(table_to_distributed_table_) {}
        bool success = true;
    };


    static void visit(ASTPtr & ast, Data & data)
    {
        if (auto * node = ast->as<ASTTableExpression>())
            visit(*node, data);
    }

    static bool needChildVisit(ASTPtr &, const ASTPtr &)
    {
        return true;
    }

private:
    static void visit(ASTTableExpression & table, Data & data) {
        if (auto * database_and_table = table.database_and_table_name->as<ASTTableIdentifier>()) {
            auto table_id = database_and_table->getTableId();
            if (data.table_to_distributed_table.contains(table_id.getFullTableName())) {
                table.children.clear();
                table.database_and_table_name =
                    std::make_shared<ASTTableIdentifier>(data.table_to_distributed_table.at(table_id.getFullTableName()));
                table.children.emplace_back(table.database_and_table_name);
            } else {
                data.success = false;
            }
        }
    }
};

MaterializedViewMemoryCache & MaterializedViewMemoryCache::instance()
{
    static MaterializedViewMemoryCache cache;
    return cache;
}

std::optional<MaterializedViewStructurePtr>
MaterializedViewMemoryCache::getMaterializedViewStructure(
    const StorageID & database_and_table_name,
    ContextMutablePtr context,
    bool local_materialized_view,
    const std::map<String, StorageID> & local_table_to_distributed_table)
{
    auto dependent_table = DatabaseCatalog::instance().tryGetTable(database_and_table_name, context);
    if (!dependent_table)
        return {};

    auto materialized_view = dynamic_pointer_cast<StorageMaterializedView>(dependent_table);
    if (!materialized_view)
        return {};

    ASTPtr query = materialized_view->getInnerQuery();
    StorageID materialized_view_id = materialized_view->getStorageID();
    std::optional<StorageID> target_table_id = findTargetTable(
        local_materialized_view, *materialized_view, materialized_view_id, context);
    if (!target_table_id) {
        return {};
    }

    if (local_materialized_view) {
        LocalTableRewriter::Data data{local_table_to_distributed_table};
        LocalTableRewriter::Visitor(data).visit(query);
    }

    auto query_ptr = QueryRewriter().rewrite(query, context, false);
    AnalysisPtr analysis = QueryAnalyzer::analyze(query_ptr, context);

    if (!analysis->non_deterministic_functions.empty())
        return {};
    QueryPlanPtr query_plan = QueryPlanner().plan(query_ptr, *analysis, context);

    static Rewriters rewriters
        = {std::make_shared<PredicatePushdown>(),
           std::make_shared<IterativeRewriter>(Rules::simplifyExpressionRules(), "SimplifyExpression"),
           std::make_shared<IterativeRewriter>(Rules::removeRedundantRules(), "RemoveRedundant"),
           std::make_shared<IterativeRewriter>(Rules::inlineProjectionRules(), "InlineProjection"),
           std::make_shared<IterativeRewriter>(Rules::normalizeExpressionRules(), "NormalizeExpression")};

    for (auto & rewriter : rewriters)
        rewriter->rewrite(*query_plan, context);

    GraphvizPrinter::printLogicalPlan(*query_plan, context, "MaterializedViewMemoryCache");
    return MaterializedViewStructure::buildFrom(materialized_view_id, target_table_id.value(), query_plan->getPlanNode(), context);
}

std::optional<StorageID> MaterializedViewMemoryCache::findTargetTable(
    bool local_materialized_view, const StorageMaterializedView & view, const StorageID & view_id, ContextMutablePtr context) {
    if (!local_materialized_view) {
        return std::make_optional(view.getTargetTableId());
    }

    if (view_id.getTableName().ends_with("_local")) {
        StorageID distributed_materialized_view_id{
            view_id.getDatabaseName(),
            view_id.getTableName().substr(0, view_id.getTableName().size() - 6)};
        auto distributed = DatabaseCatalog::instance().tryGetTable(
            distributed_materialized_view_id, context);
        if (auto * storage_distributed = distributed->as<StorageDistributed>()) {
            return std::make_optional(distributed_materialized_view_id);
        }
    } else {
        StorageID distributed_materialized_view_id{
            view_id.getDatabaseName(),
            view_id.getTableName() + "_distributed"};
        auto distributed = DatabaseCatalog::instance().tryGetTable(
            distributed_materialized_view_id, context);
        if (auto * storage_distributed = distributed->as<StorageDistributed>()) {
            return std::make_optional(distributed_materialized_view_id);
        }
    }
    return {};
}
}
