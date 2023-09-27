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

#include <Analyzers/QueryAnalyzer.h>
#include <Analyzers/QueryRewriter.h>
#include <Interpreters/Context.h>
// #include <Interpreters/DistributedStages/MPPQueryCoordinator.h>
#include <Interpreters/InterpreterSelectQueryUseOptimizer.h>
#include <Interpreters/SegmentScheduler.h>
#include <Optimizer/JoinOrderUtils.h>
// #include <Optimizer/PlanNodeSearcher.h>
#include <Optimizer/PlanOptimizer.h>
#include <Processors/QueryPlan/GraphvizPrinter.h>
#include <Processors/QueryPlan/PlanCache.h>
#include <Processors/QueryPlan/QueryPlan.h>
#include <Processors/QueryPlan/QueryPlanner.h>
// #include <Storages/StorageCnchMergeTree.h>
// #include <Storages/StorageCnchHive.h>
// #include <Interpreters/WorkerStatusManager.h>
#include <Storages/StorageDistributed.h>
#include <Common/logger_useful.h>
#include "Processors/QueryPlan/QueryPlan.h"

namespace DB
{
QueryPlanPtr InterpreterSelectQueryUseOptimizer::buildQueryPlan()
{
        // When interpret sub query, reuse context info, e.g. PlanNodeIdAllocator, SymbolAllocator.
    if (interpret_sub_query) 
    {
        QueryPlanPtr sub_query_plan = std::make_unique<QueryPlan>(sub_plan_ptr, cte_info, context->getPlanNodeIdAllocator());
        PlanOptimizer::optimize(*sub_query_plan, context);
        return sub_query_plan;
    }

    QueryPlanPtr query_plan;
    // UInt128 query_hash;
    context->createPlanNodeIdAllocator();
    context->createSymbolAllocator();
    context->createOptimizerMetrics();

    {
        // if (context->getSettingsRef().enable_plan_cache)
        // {
        //     query_hash = PlanCacheManager::hash(query_ptr, context->getSettingsRef());
        //     auto & cache = PlanCacheManager::instance();
        //     auto cached = cache.get(query_hash);
        //     if (cached)
        //     {
        //         query_plan = std::make_unique<QueryPlan>();
        //         ReadBufferFromString query_buffer(*cached);
        //         query_plan->addInterpreterContext(context);
        //         query_plan->deserialize(query_buffer);
        //         LOG_INFO(log, "hit plan cache");
        //     }
        // }

        if (!query_plan)
        {
            Stopwatch stage_watch;
            stage_watch.start();
            auto cloned_query = query_ptr->clone();
            cloned_query = QueryRewriter().rewrite(cloned_query, context);
            context->logOptimizerProfile(log, "Optimizer stage run time: ", "Rewrite", std::to_string(stage_watch.elapsedMillisecondsAsDouble()) + "ms");

            stage_watch.restart();
            AnalysisPtr analysis = QueryAnalyzer::analyze(cloned_query, context);
            context->logOptimizerProfile(log, "Optimizer stage run time: ", "Analyzer", std::to_string(stage_watch.elapsedMillisecondsAsDouble()) + "ms");

            stage_watch.restart();
            query_plan = QueryPlanner().plan(cloned_query, *analysis, context);
            context->logOptimizerProfile(log, "Optimizer stage run time: ", "Planning", std::to_string(stage_watch.elapsedMillisecondsAsDouble()) + "ms");

            stage_watch.restart();
            PlanOptimizer::optimize(*query_plan, context);
            context->logOptimizerProfile(log, "Optimizer stage run time: ", "Optimizer", std::to_string(stage_watch.elapsedMillisecondsAsDouble()) + "ms");
            // if (context->getSettingsRef().enable_plan_cache)
            // {
            //     WriteBufferFromOwnString query_buffer;
            //     query_plan->serialize(query_buffer);
            //     auto & query_str = query_buffer.str();
            //     if (query_str.size() <= context->getSettingsRef().max_plan_mem_size)
            //     {
            //         auto & cache = PlanCacheManager::instance();
            //         cache.add(query_hash, query_str);
            //     }
            // }
        }
    }

    LOG_DEBUG(log, "join order {}", JoinOrderUtils::getJoinOrder(*query_plan));
    return query_plan;
}

// PlanSegmentTreePtr InterpreterSelectQueryUseOptimizer::getPlanSegment()
// {
//     Stopwatch stage_watch, total_watch;
//     total_watch.start();
//     QueryPlanPtr query_plan;
//     if (context->getSettingsRef().enable_plan_cache)
//     {
//         auto hash = PlanCacheManager::hash(query_ptr, context->getSettingsRef());
//         auto & cache = PlanCacheManager::instance();
//         if (cache.has(hash))
//         {
//             query_plan = std::make_unique<QueryPlan>();
//             String plan_str = *cache.get(hash);

//             ReadBufferFromString query_buffer(plan_str);
//             query_plan->addInterpreterContext(context);
//             query_plan->deserialize(query_buffer);
//             LOG_INFO(log, "hit plan cache");
//         }
//         else
//         {
//             query_plan = buildQueryPlan();

//             WriteBufferFromOwnString query_buffer;
//             query_plan->serialize(query_buffer);
//             auto & query_str = query_buffer.str();
//             if (query_str.size() <= context->getSettingsRef().max_plan_mem_size)
//             {
//                 cache.add(hash, query_str);
//             }
//         }
//     }

//     if (!query_plan)
//     {
//         query_plan = buildQueryPlan();
//     }

//     query_plan->setResetStepId(false);
//     stage_watch.start();
//     QueryPlan plan = PlanNodeToNodeVisitor::convert(*query_plan);

//     LOG_DEBUG(log, "optimizer stage run time: plan normalize, {} ms", stage_watch.elapsedMillisecondsAsDouble());
//     stage_watch.restart();

//     PlanSegmentTreePtr plan_segment_tree = std::make_unique<PlanSegmentTree>();
//     ClusterInfoContext cluster_info_context{.query_plan = *query_plan, .context = context, .plan_segment_tree = plan_segment_tree};
//     PlanSegmentContext plan_segment_context = ClusterInfoFinder::find(*query_plan, cluster_info_context);

//     stage_watch.restart();
//     plan.allocateLocalTable(context);

//     // select health worker before split
//     if (context->getSettingsRef().enable_adaptive_scheduler && context->tryGetCurrentWorkerGroup())
//     {
//         context->selectWorkerNodesWithMetrics();
//         auto wg_health = context->getWorkerGroupStatusPtr()->getWorkerGroupHealth();
//         if (wg_health == WorkerGroupHealthStatus::Critical)
//             throw Exception("no worker available", ErrorCodes::LOGICAL_ERROR);
//     }

//     PlanSegmentSplitter::split(plan, plan_segment_context);
//     context->logOptimizerProfile(log, "Optimizer total run time: ", "PlanSegment build", std::to_string(stage_watch.elapsedMillisecondsAsDouble()) + "ms");

//     GraphvizPrinter::printPlanSegment(plan_segment_tree, context);
//     context->logOptimizerProfile(log, "Optimizer total run time: ", "Optimizer Total", std::to_string(total_watch.elapsedMillisecondsAsDouble()) + "ms");
//     return plan_segment_tree;
// }
//
// BlockIO InterpreterSelectQueryUseOptimizer::execute()
// {
//     PlanSegmentTreePtr plan_segment_tree = getPlanSegment();

//     auto coodinator = std::make_shared<MPPQueryCoordinator>(std::move(plan_segment_tree), context, MPPQueryOptions());
//     return coodinator->execute();
// }

// QueryPlan PlanNodeToNodeVisitor::convert(QueryPlan & query_plan)
// {
//     QueryPlan plan;
//     PlanNodeToNodeVisitor visitor(plan);
//     Void c;
//     auto * root = VisitorUtil::accept(query_plan.getPlanNode(), visitor, c);
//     plan.setRoot(root);

//     for (const auto & cte : query_plan.getCTEInfo().getCTEs())
//         plan.getCTENodes().emplace(cte.first, VisitorUtil::accept(cte.second, visitor, c));
//     return plan;
// }

QueryPlan::Node * PlanNodeToNodeVisitor::visitPlanNode(PlanNodeBase & node, Void & c)
{
    if (node.getChildren().empty())
    {
        auto res = QueryPlan::Node{.step = std::const_pointer_cast<IQueryPlanStep>(node.getStep()), .children = {}};
        node.setStep(res.step);
        plan.addNode(std::move(res));
        return plan.getLastNode();
    }

    std::vector<QueryPlan::Node *> children;
    for (const auto & item : node.getChildren())
    {
        auto * child = VisitorUtil::accept(*item, *this, c);
        children.emplace_back(child);
    }
    QueryPlan::Node query_plan_node{
        .step = std::const_pointer_cast<IQueryPlanStep>(node.getStep()), .children = children};
    node.setStep(query_plan_node.step);
    plan.addNode(std::move(query_plan_node));
    return plan.getLastNode();
}

// PlanSegmentContext ClusterInfoFinder::find(QueryPlan & plan, ClusterInfoContext & cluster_info_context)
// {
//     ClusterInfoFinder visitor{plan.getCTEInfo()};

//     // default schedule to worker cluster
//     std::optional<PlanSegmentContext> result = VisitorUtil::accept(plan.getPlanNode(), visitor, cluster_info_context);
//     if (result.has_value())
//     {
//         return result.value();
//     }

//     // if query is a constant query, like, select 1, schedule to server (coordinator)
//     PlanSegmentContext plan_segment_context{
//         .context = cluster_info_context.context,
//         .query_plan = cluster_info_context.query_plan,
//         .query_id = cluster_info_context.context->getCurrentQueryId(),
//         .shard_number = 1,
//         .cluster_name = "",
//         .plan_segment_tree = cluster_info_context.plan_segment_tree.get()};
//     return plan_segment_context;
// }

// std::optional<PlanSegmentContext> ClusterInfoFinder::visitPlanNode(PlanNodeBase & node, ClusterInfoContext & cluster_info_context)
// {
//     for (const auto & child : node.getChildren())
//     {
//         auto result = VisitorUtil::accept(child, *this, cluster_info_context);
//         if (result.has_value())
//             return result;
//     }
//     return std::nullopt;
// }

// std::optional<PlanSegmentContext> ClusterInfoFinder::visitTableScanNode(TableScanNode & node, ClusterInfoContext & cluster_info_context)
// {
//     auto source_step = node.getStep();
//     const auto * cnch_table = dynamic_cast<StorageCnchMergeTree *>(source_step->getStorage().get());
//     const auto * cnch_hive = dynamic_cast<StorageCnchHive *>(source_step->getStorage().get());

//     if (cnch_table || cnch_hive)
//     {
//         const auto & worker_group = cluster_info_context.context->getCurrentWorkerGroup();
//         auto worker_group_status_ptr = cluster_info_context.context->getWorkerGroupStatusPtr();
//         PlanSegmentContext plan_segment_context{
//             .context = cluster_info_context.context,
//             .query_plan = cluster_info_context.query_plan,
//             .query_id = cluster_info_context.context->getCurrentQueryId(),
//             .shard_number =  worker_group->getShardsInfo().size(),
//             .cluster_name = worker_group->getID(),
//             .plan_segment_tree = cluster_info_context.plan_segment_tree.get(),
//             .health_parallel = worker_group_status_ptr ?
//                 std::optional<size_t>(worker_group_status_ptr->getAvaiableComputeWorkerSize()) : std::nullopt};
//         return plan_segment_context;
//     }
//     return std::nullopt;
// }

// std::optional<PlanSegmentContext> ClusterInfoFinder::visitCTERefNode(CTERefNode & node, ClusterInfoContext & cluster_info_context)
// {
//     const auto * cte = dynamic_cast<const CTERefStep *>(node.getStep().get());
//     return cte_helper.accept(cte->getId(), *this, cluster_info_context);
// }
}
