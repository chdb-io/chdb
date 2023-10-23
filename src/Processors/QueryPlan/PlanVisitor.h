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

#include <Processors/QueryPlan/AggregatingStep.h>
#include <Processors/QueryPlan/ApplyStep.h>
#include <Processors/QueryPlan/ArrayJoinStep.h>
#include <Processors/QueryPlan/AssignUniqueIdStep.h>
#include <Processors/QueryPlan/CTERefStep.h>
#include <Processors/QueryPlan/CreatingSetsStep.h>
#include <Processors/QueryPlan/CubeStep.h>
#include <Processors/QueryPlan/DistinctStep.h>
#include <Processors/QueryPlan/EnforceSingleRowStep.h>
#include <Processors/QueryPlan/ExceptStep.h>
// #include <Processors/QueryPlan/ExchangeStep.h>
#include <Processors/QueryPlan/ExpressionStep.h>
#include <Processors/QueryPlan/ExtremesStep.h>
#include <Processors/QueryPlan/ExplainAnalyzeStep.h>
#include <Processors/QueryPlan/FillingStep.h>
#include <Processors/QueryPlan/FilterStep.h>
#include <Processors/QueryPlan/FinalSampleStep.h>
// #include <Processors/QueryPlan/FinishSortingStep.h>
#include <Processors/QueryPlan/IQueryPlanStep.h>
#include <Processors/QueryPlan/IntersectStep.h>
#include <Processors/QueryPlan/JoinStep.h>
#include <Processors/QueryPlan/MarkDistinctStep.h>
#include <Processors/QueryPlan/LimitByStep.h>
#include <Processors/QueryPlan/LimitStep.h>
#include <Processors/QueryPlan/MergeSortingStep.h>
#include <Processors/QueryPlan/MergingAggregatedStep.h>
#include <Processors/QueryPlan/MergingSortedStep.h>
#include <Processors/QueryPlan/OffsetStep.h>
#include <Processors/QueryPlan/PartialSortingStep.h>
#include <Processors/QueryPlan/PlanNode.h>
#include <Processors/QueryPlan/PlanSegmentSourceStep.h>
#include <Processors/QueryPlan/ProjectionStep.h>
// #include <Processors/QueryPlan/QueryCacheStep.h>
#include <Processors/QueryPlan/ReadFromMergeTree.h>
#include <Processors/QueryPlan/ReadFromPreparedSource.h>
#include <Processors/QueryPlan/ReadNothingStep.h>
// #include <Processors/QueryPlan/RemoteExchangeSourceStep.h>
#include <Processors/QueryPlan/RollupStep.h>
// #include <Processors/QueryPlan/SettingQuotaAndLimitsStep.h>
#include <Processors/QueryPlan/SortingStep.h>
#include <Processors/QueryPlan/SymbolAllocator.h>
#include <Processors/QueryPlan/TableScanStep.h>
#include <Processors/QueryPlan/TotalsHavingStep.h>
#include <Processors/QueryPlan/UnionStep.h>
#include <Processors/QueryPlan/ValuesStep.h>
#include <Processors/QueryPlan/Void.h>
#include <Processors/QueryPlan/WindowStep.h>

namespace DB
{
/// PlanNode visitor, for optimizer only.
template <typename R, typename C>
class PlanNodeVisitor
{
public:
    virtual ~PlanNodeVisitor() = default;

    virtual R visitPlanNode(PlanNodeBase &, C &)
    {
        throw Exception("Visitor does not supported this plan node.", ErrorCodes::NOT_IMPLEMENTED);
    }

#define VISITOR_DEF(TYPE) \
    virtual R visit##TYPE##Node(TYPE##Node & node, C & context) { return visitPlanNode(static_cast<PlanNodeBase &>(node), context); }
    APPLY_STEP_TYPES(VISITOR_DEF)
#undef VISITOR_DEF
};

/// IQueryPlanStep visitor
template <typename R, typename C>
class StepVisitor
{
public:
    virtual ~StepVisitor() = default;

    virtual R visitStep(const IQueryPlanStep &, C &)
    {
        throw Exception("Visitor does not supported this step.", ErrorCodes::NOT_IMPLEMENTED);
    }

#define VISITOR_DEF(TYPE) \
    virtual R visit##TYPE##Step(const TYPE##Step & step, C & context) \
    { \
        return visitStep(static_cast<const IQueryPlanStep &>(step), context); \
    }
    APPLY_STEP_TYPES(VISITOR_DEF)
#undef VISITOR_DEF
};

/// QueryPlan::Node visitor
template <typename R, typename C>
class NodeVisitor
{
public:
    virtual ~NodeVisitor() = default;

    virtual R visitNode(QueryPlan::Node *, C &) { throw Exception("Visitor does not supported this step.", ErrorCodes::NOT_IMPLEMENTED); }

#define VISITOR_DEF(TYPE) \
    virtual R visit##TYPE##Node(QueryPlan::Node * node, C & context) { return visitNode(node, context); }
    APPLY_STEP_TYPES(VISITOR_DEF)
#undef VISITOR_DEF
};

class VisitorUtil
{
public:
    template <typename R, typename C>
    static R accept(PlanNodeBase & node, PlanNodeVisitor<R, C> & visitor, C & context)
    {
        switch (node.getType())
        {
#define VISITOR_DEF(TYPE) \
    case IQueryPlanStep::Type::TYPE: { \
        return visitor.visit##TYPE##Node(static_cast<TYPE##Node &>(node), context); \
    }
            APPLY_STEP_TYPES(VISITOR_DEF)

#undef VISITOR_DEF
            default:
                return visitor.visitPlanNode(node, context);
        }
    }

    template <typename R, typename C>
    static R accept(const PlanNodePtr & node, PlanNodeVisitor<R, C> & visitor, C & context)
    {
        switch (node->getType())
        {
#define VISITOR_DEF(TYPE) \
    case IQueryPlanStep::Type::TYPE: { \
        return visitor.visit##TYPE##Node(static_cast<TYPE##Node &>(*node), context); \
    }
            APPLY_STEP_TYPES(VISITOR_DEF)

#undef VISITOR_DEF
            default:
                return visitor.visitPlanNode(*node, context);
        }
    }

    template <typename R, typename C>
    static R accept(const QueryPlanStepPtr & step, StepVisitor<R, C> & visitor, C & context)
    {
        switch (step->getType())
        {
#define VISITOR_DEF(TYPE) \
    case IQueryPlanStep::Type::TYPE: { \
        return visitor.visit##TYPE##Step(static_cast<const TYPE##Step &>(*step), context); \
    }
            APPLY_STEP_TYPES(VISITOR_DEF)
#undef VISITOR_DEF
            default:
                return visitor.visitStep(*step, context);
        }
    }

    template <typename R, typename C>
    static R accept(QueryPlan::Node * node, NodeVisitor<R, C> & visitor, C & context)
    {
        switch (node ? node->step->getType() : IQueryPlanStep::Type::Any)
        {
#define VISITOR_DEF(TYPE) \
    case IQueryPlanStep::Type::TYPE: { \
        return visitor.visit##TYPE##Node(node, context); \
    }
            APPLY_STEP_TYPES(VISITOR_DEF)
#undef VISITOR_DEF
            default:
                return visitor.visitNode(node, context);
        }
    }
};

}
