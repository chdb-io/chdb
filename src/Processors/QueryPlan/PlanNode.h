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
#include <Core/Types.h>
#include <Optimizer/CardinalityEstimate/PlanNodeStatisticsEstimate.h>
#include <Processors/QueryPlan/AggregatingStep.h>
#include <Processors/QueryPlan/AnyStep.h>
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
#include <Parsers/IAST_fwd.h>
#include <Processors/QueryPlan/IQueryPlanStep.h>
#include <Processors/QueryPlan/ExplainAnalyzeStep.h>
#include <Processors/QueryPlan/ExpressionStep.h>
#include <Processors/QueryPlan/ExtremesStep.h>
#include <Processors/QueryPlan/FillingStep.h>
#include <Processors/QueryPlan/FilterStep.h>
#include <Processors/QueryPlan/FinalSampleStep.h>
#include <Processors/QueryPlan/FinishSortingStep.h>
#include <Processors/QueryPlan/IntersectOrExceptStep.h>
#include <Processors/QueryPlan/IntersectStep.h>
#include <Processors/QueryPlan/JoinStep.h>
#include <Processors/QueryPlan/LimitByStep.h>
#include <Processors/QueryPlan/LimitStep.h>
#include <Processors/QueryPlan/MarkDistinctStep.h>
#include <Processors/QueryPlan/MergeSortingStep.h>
#include <Processors/QueryPlan/MergingAggregatedStep.h>
#include <Processors/QueryPlan/MergingSortedStep.h>
#include <Processors/QueryPlan/OffsetStep.h>
#include <Processors/QueryPlan/PartialSortingStep.h>
#include <Processors/QueryPlan/PartitionTopNStep.h>
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
#include <Processors/QueryPlan/TopNFilteringStep.h>
#include <Processors/QueryPlan/TotalsHavingStep.h>
#include <Processors/QueryPlan/UnionStep.h>
#include <Processors/QueryPlan/ValuesStep.h>
#include <Processors/QueryPlan/Void.h>
#include <Processors/QueryPlan/WindowStep.h>

#include <memory>
#include <utility>

namespace DB
{
template <class Step>
class PlanNode;

class PlanNodeBase;
using PlanNodePtr = std::shared_ptr<PlanNodeBase>;
using PlanNodes = std::vector<PlanNodePtr>;

using QueryPlanStepPtr = std::shared_ptr<IQueryPlanStep>;
using PlanNodeId = UInt32;

class PlanNodeBase : public std::enable_shared_from_this<PlanNodeBase>
{
public:
    PlanNodeBase(PlanNodeId id_, PlanNodes children_) : id(id_), children(std::move(children_)) { }
    virtual ~PlanNodeBase() = default;
    PlanNodeId getId() const { return id; }

    PlanNodes & getChildren() { return children; }
    const PlanNodes & getChildren() const { return children; }
    void replaceChildren(const PlanNodes & children_) { replaceChildrenImpl(children_); }

    void replaceChildren(PlanNodes && children_) { children = std::move(children_); }
    void setStatistics(const PlanNodeStatisticsEstimate & statistics_) { statistics = statistics_; }
    const PlanNodeStatisticsEstimate & getStatistics() const { return statistics; }
    QueryPlanStepPtr getStep() const { return getStepImpl(); }
    void setStep(QueryPlanStepPtr & step_) { setStepImpl(step_); }


    virtual PlanNodePtr addStep(PlanNodeId new_id, QueryPlanStepPtr new_step, PlanNodes new_children = {}) = 0;
    virtual PlanNodePtr copy(PlanNodeId new_id, ContextPtr context) = 0;
    virtual IQueryPlanStep::Type getType() const = 0;
    virtual const DataStream & getCurrentDataStream() const = 0;

    NamesAndTypes getOutputNamesAndTypes() const { return getCurrentDataStream().header.getNamesAndTypes(); }
    NameToType getOutputNamesToTypes() const { return getCurrentDataStream().header.getNamesToTypes(); }
    Names getOutputNames() const { return getCurrentDataStream().header.getNames(); }
    PlanNodePtr getNodeById(PlanNodeId node_id) const;

    static PlanNodePtr createPlanNode(
        [[maybe_unused]] PlanNodeId id_,
        [[maybe_unused]] QueryPlanStepPtr step_,
        [[maybe_unused]] const PlanNodes & children_ = {},
        [[maybe_unused]] const PlanNodeStatisticsEstimate & statistics_ = {})
    {
        PlanNodePtr plan_node;
#define CREATE_PLAN_NODE(TYPE) \
    if (step_->getType() == IQueryPlanStep::Type::TYPE) \
    { \
        auto spec_step = std::dynamic_pointer_cast<TYPE##Step>(step_); \
        plan_node = std::dynamic_pointer_cast<PlanNodeBase>(std::make_shared<PlanNode<TYPE##Step>>(id_, std::move(spec_step), children_)); \
    }

        APPLY_STEP_TYPES(CREATE_PLAN_NODE)
        CREATE_PLAN_NODE(Any)
#undef CREATE_PLAN_NODE
        plan_node->setStatistics(statistics_);
        return plan_node;
    }

protected:
    PlanNodeId id;
    PlanNodes children;
    PlanNodeStatisticsEstimate statistics;

private:
    virtual QueryPlanStepPtr getStepImpl() const = 0;
    virtual void setStepImpl(QueryPlanStepPtr & step_) = 0;
    virtual void replaceChildrenImpl(const PlanNodes & children_) = 0;
};

template <class Step>
class PlanNode : public PlanNodeBase
{
public:
    using StepPtr = std::shared_ptr<Step>;
    PlanNode(const PlanNode &) = delete;
    PlanNode(const PlanNode &&) = delete;
    PlanNode(PlanNode &&) = delete;
    PlanNode & operator=(const PlanNode &) = delete;
    PlanNode & operator=(PlanNode &&) = delete;

    IQueryPlanStep::Type getType() const override { return step->getType(); }
    StepPtr & getStep() { return step; }

    void setStep(StepPtr & step_) { step = step_; }
    const DataStream & getCurrentDataStream() const override { return step->getOutputStream(); }

    static PlanNodePtr
    createPlanNode(PlanNodeId id_, StepPtr step_, const PlanNodes & children_ = {}, const PlanNodeStatisticsEstimate & statistics_ = {})
    {
        PlanNodePtr plan_node = std::make_shared<PlanNode<Step>>(id_, std::move(step_), children_);
        plan_node->setStatistics(statistics_);
        return plan_node;
    }


    PlanNodePtr copy(PlanNodeId new_id, ContextPtr context) override
    {
        auto new_step = dynamic_pointer_cast<Step>(step->copy(context));
        return createPlanNode(new_id, std::move(new_step), children, statistics);
    }

    PlanNodePtr addStep(PlanNodeId new_id, QueryPlanStepPtr new_step, PlanNodes new_children) override
    {
        if (new_children.empty() && new_step->getInputStreams().size() == 1)
        {
            new_children.emplace_back(this->shared_from_this());
        }
        else if (children.size() != step->getInputStreams().size())
        {
            throw Exception(
                "Expected " + std::to_string(step->getInputStreams().size()) + " children, but input arguments have "
                    + std::to_string(children.size()),
                ErrorCodes::LOGICAL_ERROR);
        }
        return PlanNodeBase::createPlanNode(new_id, std::move(new_step), new_children);
    }

    PlanNode(PlanNodeId id_, StepPtr step_, PlanNodes children_ = {}) : PlanNodeBase(id_, children_), step(std::move(step_)) { }

private:
    QueryPlanStepPtr getStepImpl() const override { return step; }

    void replaceChildrenImpl(const PlanNodes & children_) override
    {
        children = children_;

        DataStreams inputs;
        for (const auto & child : children)
        {
            inputs.emplace_back(child->getCurrentDataStream());
        }
    }

    void setStepImpl(QueryPlanStepPtr & step_) override
    {
        auto new_step = std::dynamic_pointer_cast<Step>(step_);
        if (new_step)
        {
            step = new_step;
        }
    }

    StepPtr step;
};

#define PLAN_NODE_DEF(TYPE) \
    extern template class PlanNode<TYPE##Step>; \
    using TYPE##Node = PlanNode<TYPE##Step>;

APPLY_STEP_TYPES(PLAN_NODE_DEF)
PLAN_NODE_DEF(Any)
#undef PLAN_NODE_DEF

}
