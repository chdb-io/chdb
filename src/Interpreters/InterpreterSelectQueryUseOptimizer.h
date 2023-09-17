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

#pragma once
#include <Common/Stopwatch.h>
#include <Interpreters/IInterpreter.h>
#include <Interpreters/SelectQueryOptions.h>
#include <Processors/QueryPlan/CTEVisitHelper.h>
#include <Processors/QueryPlan/PlanVisitor.h>
// #include <Interpreters/DistributedStages/PlanSegmentSplitter.h>
#include <Interpreters/QueryLog.h>
#include <Poco/Logger.h>

namespace Poco
{
class Logger;
}

namespace DB
{
class InterpreterSelectQueryUseOptimizer : public IInterpreter
{
public:
    InterpreterSelectQueryUseOptimizer(const ASTPtr & query_ptr_, ContextMutablePtr & context_, const SelectQueryOptions & options_)
        : query_ptr(query_ptr_), context(context_), options(options_), log(&Poco::Logger::get("InterpreterSelectQueryUseOptimizer"))
    {
        interpret_sub_query = false;
    }

    InterpreterSelectQueryUseOptimizer(PlanNodePtr sub_plan_ptr_, CTEInfo cte_info_, ContextMutablePtr & context_, const SelectQueryOptions & options_)
        : sub_plan_ptr(sub_plan_ptr_), cte_info(std::move(cte_info_)), context(context_), options(options_), log(&Poco::Logger::get("InterpreterSelectQueryUseOptimizer"))
    {
        interpret_sub_query = true;
    }

    QueryPlanPtr buildQueryPlan();
    PlanSegmentTreePtr getPlanSegment();

    BlockIO execute() override;

    void extendQueryLogElemImpl(QueryLogElement & elem, const ASTPtr &, ContextPtr) const override
    {
        elem.query_kind = "Select";
    }

private:
    ASTPtr query_ptr;
    PlanNodePtr sub_plan_ptr;
    CTEInfo cte_info;
    ContextMutablePtr context;
    SelectQueryOptions options;
    Poco::Logger * log;
    bool interpret_sub_query;
};

/**
 * Convert PlanNode to QueryPlan::Node.
 */
class PlanNodeToNodeVisitor : public PlanNodeVisitor<QueryPlan::Node *, Void>
{
public:
    static QueryPlan convert(QueryPlan &);
    explicit PlanNodeToNodeVisitor(QueryPlan & plan_) : plan(plan_) { }
    QueryPlan::Node * visitPlanNode(PlanNodeBase & node, Void & c) override;

private:
    QueryPlan & plan;
};

struct ClusterInfoContext
{
    QueryPlan & query_plan;
    ContextMutablePtr context;
    PlanSegmentTreePtr & plan_segment_tree;
};

class ClusterInfoFinder : public PlanNodeVisitor<std::optional<PlanSegmentContext>, ClusterInfoContext>
{
public:
    static PlanSegmentContext find(QueryPlan & plan, ClusterInfoContext & cluster_info_context);
    explicit ClusterInfoFinder(CTEInfo & cte_info_) : cte_helper(cte_info_) { }
    std::optional<PlanSegmentContext> visitPlanNode(PlanNodeBase & node, ClusterInfoContext & cluster_info_context) override;
    std::optional<PlanSegmentContext> visitTableScanNode(TableScanNode & node, ClusterInfoContext & cluster_info_context) override;
    std::optional<PlanSegmentContext> visitCTERefNode(CTERefNode & node, ClusterInfoContext & cluster_info_context) override;
private:
    SimpleCTEVisitHelper<std::optional<PlanSegmentContext>> cte_helper;
};
}
