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

#include <Core/Names.h>
#include <Processors/QueryPlan/PlanNode.h>
#include <Processors/QueryPlan/SymbolAllocator.h>
#include <Processors/QueryPlan/SymbolMapper.h>
#include <Interpreters/Context.h>
#include <Interpreters/Context_fwd.h>
#include <Processors/QueryPlan/CTEInfo.h>
#include <Processors/QueryPlan/ExceptStep.h>
#include <Processors/QueryPlan/PlanVisitor.h>
#include <Processors/QueryPlan/SimplePlanRewriter.h>

#include <memory>

namespace DB
{
class PlanCopier 
{
public:
    static std::shared_ptr<ProjectionStep> reallocateWithProjection(
        const DataStream & data_stream, SymbolAllocator & symbolAllocator,
        std::unordered_map<std::string, std::string> & reallocated_names);

    /// copy will copy the current PlanNode, and replace all of its symbols to new symbols recursively
    static PlanNodePtr copy(const PlanNodePtr & plan, ContextMutablePtr & context, std::unordered_map<SymbolMapper::Symbol, SymbolMapper::Symbol> & mapping);
    static QueryPlanPtr copy(const QueryPlanPtr & plan, ContextMutablePtr & context);

    static PlanNodePtr copyWithoutReplacingSymbol(const PlanNodePtr & plan, ContextMutablePtr & context);

    static QueryPlanPtr copyWithoutReplacingSymbol(const QueryPlanPtr & plan, ContextMutablePtr & context);

    static bool isOverlapping(const DataStream & data_stream, const DataStream & data_stream2);

private:
    static PlanNodePtr copyPlanNode(const PlanNodePtr & plan, ContextMutablePtr & context, std::unordered_map<SymbolMapper::Symbol, SymbolMapper::Symbol>& mapping);
};

class PlanCopierVisitor : public PlanNodeVisitor<PlanNodePtr, Void>
{
public:
    
    explicit PlanCopierVisitor(ContextMutablePtr context_,  std::unordered_map<SymbolMapper::Symbol, SymbolMapper::Symbol>& mapping_) : id_allocator(context_->getPlanNodeIdAllocator()), mapping(mapping_), context_mut_ptr(context_), symbol_mapper(nullptr, nullptr) {
        symbol_mapper = SymbolMapper::symbolReallocator(this->mapping, *context_->getSymbolAllocator(), context_);
    }

    std::unordered_map<SymbolMapper::Symbol, SymbolMapper::Symbol> getMappings() { return mapping; }
    /// this function will copy planNode and reallocate new symbols from bottom to top
    PlanNodePtr copy(const PlanNodePtr & plan, ContextMutablePtr & context);
    static QueryPlanPtr copy(const QueryPlanPtr & plan, ContextMutablePtr & context);

    static bool isOverlapping(const DataStream & data_stream, const DataStream & data_stream2);


    PlanNodePtr visitPlanNode(PlanNodeBase &, Void &) override;
    PlanNodePtr visitProjectionNode(ProjectionNode &, Void &) override;
    PlanNodePtr visitAggregatingNode(AggregatingNode &, Void &) override;
    PlanNodePtr visitApplyNode(ApplyNode &, Void &) override;
    PlanNodePtr visitArrayJoinNode(ArrayJoinNode &, Void &) override;
    PlanNodePtr visitAssignUniqueIdNode(AssignUniqueIdNode &, Void &) override;
    PlanNodePtr visitCreatingSetsNode(CreatingSetsNode &, Void &) override;
    PlanNodePtr visitCubeNode(CubeNode &, Void &) override;
    PlanNodePtr visitDistinctNode(DistinctNode &, Void &) override;
    PlanNodePtr visitEnforceSingleRowNode(EnforceSingleRowNode &, Void &) override;
    PlanNodePtr visitExpressionNode(ExpressionNode &, Void &) override;
    PlanNodePtr visitExceptNode(ExceptNode &, Void &) override;
    PlanNodePtr visitExtremesNode(ExtremesNode &, Void &) override;
    PlanNodePtr visitExchangeNode(ExchangeNode &, Void &) override;
    PlanNodePtr visitFillingNode(FillingNode &, Void &) override;
    PlanNodePtr visitFinalSampleNode(FinalSampleNode &, Void &) override;
    PlanNodePtr visitFinishSortingNode(FinishSortingNode &, Void &) override;
    PlanNodePtr visitFilterNode(FilterNode &, Void &) override;
    PlanNodePtr visitIntersectNode(IntersectNode &, Void &) override;
    PlanNodePtr visitJoinNode(JoinNode &, Void &) override;
    PlanNodePtr visitTableScanNode(TableScanNode &, Void &) override;
    PlanNodePtr visitLimitNode(LimitNode &, Void &) override;
    PlanNodePtr visitLimitByNode(LimitByNode &, Void &) override;
    PlanNodePtr visitTopNFilteringNode(TopNFilteringNode &, Void &) override;
    PlanNodePtr visitSortingNode(SortingNode &, Void &) override;
    PlanNodePtr visitOffsetNode(OffsetNode &, Void &) override;
    PlanNodePtr visitPartitionTopNNode(PartitionTopNNode &, Void &) override;
    PlanNodePtr visitPartialSortingNode(PartialSortingNode &, Void &) override;
    PlanNodePtr visitReadNothingNode(ReadNothingNode &, Void &) override;
    PlanNodePtr visitRemoteExchangeSourceNode(RemoteExchangeSourceNode &, Void &) override;
    PlanNodePtr visitTotalsHavingNode(TotalsHavingNode &, Void &) override;
    PlanNodePtr visitUnionNode(UnionNode &, Void &) override;
    PlanNodePtr visitValuesNode(ValuesNode &, Void &) override;
    PlanNodePtr visitWindowNode(WindowNode &, Void &) override;
    PlanNodePtr visitMergeSortingNode(MergeSortingNode &, Void &) override;
    PlanNodePtr visitMergingSortedNode(MergingSortedNode &, Void &) override;
    PlanNodePtr visitMergingAggregatedNode(MergingAggregatedNode &, Void &) override;
private:
    PlanNodeIdAllocatorPtr id_allocator;
    std::unordered_map<SymbolMapper::Symbol, SymbolMapper::Symbol> mapping;
    ContextMutablePtr context_mut_ptr;
    SymbolMapper symbol_mapper;
};

}

