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

#include <Processors/QueryPlan/PlanCopier.h>

#include <Parsers/ASTIdentifier.h>
#include <Interpreters/Context_fwd.h>
#include <Processors/QueryPlan/AssignUniqueIdStep.h>
#include <Processors/QueryPlan/CTEInfo.h>
#include <Processors/QueryPlan/CubeStep.h>
#include <Processors/QueryPlan/FinishSortingStep.h>
#include <Processors/QueryPlan/PartialSortingStep.h>
#include <Processors/QueryPlan/PlanNode.h>
#include <Processors/QueryPlan/QueryPlan.h>
// #include <Processors/QueryPlan/RemoteExchangeSourceStep.h>

namespace DB
{

std::shared_ptr<ProjectionStep> PlanCopier::reallocateWithProjection(
    const DataStream & data_stream, SymbolAllocator & symbolAllocator,
    std::unordered_map<std::string, std::string> & reallocated_names)
{
    Assignments assignments;
    NameToType name_to_type;
    for (const auto & name_and_type : data_stream.header)
    {
        const auto & name = name_and_type.name;
        auto reallocated_name = symbolAllocator.newSymbol(name);
        reallocated_names.emplace(name, reallocated_name);
        assignments.emplace_back(reallocated_name, std::make_shared<ASTIdentifier>(name));
        name_to_type.emplace(reallocated_name, name_and_type.type);
    }
    return std::make_shared<ProjectionStep>(data_stream, assignments, name_to_type);
}

PlanNodePtr PlanCopier::copyWithoutReplacingSymbol(const PlanNodePtr & plan, ContextMutablePtr & context) // NOLINT(misc-no-recursion)
{
    PlanNodes children;
    for (auto & child : plan->getChildren())
        children.emplace_back(copyWithoutReplacingSymbol(child, context));

    auto new_node = plan->copy(context->nextNodeId(), context);
    new_node->replaceChildren(children);
    new_node->setStatistics(plan->getStatistics());
    return new_node;
}

QueryPlanPtr PlanCopier::copyWithoutReplacingSymbol(const QueryPlanPtr & plan, ContextMutablePtr & context)
{
    auto plan_node = copyWithoutReplacingSymbol(plan->getPlanNode(), context);
    CTEInfo cte_info;
    for (const auto & [cte_id, cte_def] : plan->getCTEInfo().getCTEs())
        cte_info.add(cte_id, copyWithoutReplacingSymbol(cte_def, context));
    return std::make_unique<QueryPlan>(plan_node, cte_info, context->getPlanNodeIdAllocator());
}

bool PlanCopier::isOverlapping(const DataStream & lho, const DataStream & rho)
{
    NameSet name_set;
    std::transform(
        lho.header.begin(),
        lho.header.end(),
        std::inserter(name_set, name_set.end()),
        [] (const auto & nameAndType) { return nameAndType.name; });

    return std::any_of(rho.header.begin(), rho.header.end(),
                       [&] (const auto & nameAndType) { return name_set.contains(nameAndType.name); } );
}

PlanNodePtr PlanCopier::copy(const PlanNodePtr & plan, ContextMutablePtr & context, std::unordered_map<SymbolMapper::Symbol, SymbolMapper::Symbol> & mapping) // NOLINT(misc-no-recursion)
{
    PlanNodes children;
    for (auto & child : plan->getChildren())
        children.emplace_back(copy(child, context, mapping));
    auto new_node = copyPlanNode(plan, context, mapping);
    new_node->replaceChildren(children);
    new_node->setStatistics(plan->getStatistics());
    return new_node;
}

QueryPlanPtr PlanCopier::copy(const QueryPlanPtr & plan, ContextMutablePtr & context)
{
    std::unordered_map<SymbolMapper::Symbol, SymbolMapper::Symbol> mapping;
    auto plan_node = copy(plan->getPlanNode(), context, mapping);
    CTEInfo cte_info;
    for (const auto & [cte_id, cte_def] : plan->getCTEInfo().getCTEs()) {
        mapping = {};
        cte_info.add(cte_id, copy(cte_def, context, mapping));
    } 
    return std::make_unique<QueryPlan>(plan_node, cte_info, context->getPlanNodeIdAllocator());
}

PlanNodePtr PlanCopier::copyPlanNode(const PlanNodePtr & plan, ContextMutablePtr & context, std::unordered_map<SymbolMapper::Symbol, SymbolMapper::Symbol>& mapping)
{
    PlanCopierVisitor visitor(context, mapping);
    Void v;
    // copy PlanNode with new Symbols
    auto result = VisitorUtil::accept(plan, visitor, v);
    mapping = visitor.getMappings();
    return result;
}

PlanNodePtr PlanCopierVisitor::visitPlanNode(PlanNodeBase & node, Void & )
{
    return node.copy(id_allocator->nextId(), context_mut_ptr);
}

PlanNodePtr PlanCopierVisitor::visitProjectionNode(ProjectionNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return ProjectionNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitAggregatingNode(AggregatingNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return AggregatingNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitApplyNode(ApplyNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return ApplyNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitArrayJoinNode(ArrayJoinNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return ArrayJoinNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitAssignUniqueIdNode(AssignUniqueIdNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return AssignUniqueIdNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitCreatingSetsNode(CreatingSetsNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return CreatingSetsNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitCubeNode(CubeNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return CubeNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitDistinctNode(DistinctNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return DistinctNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitEnforceSingleRowNode(EnforceSingleRowNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return EnforceSingleRowNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitExpressionNode(ExpressionNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return ExpressionNode::createPlanNode(id_allocator->nextId(), step);
}

// PlanNodePtr PlanCopierVisitor::visitExchangeNode(ExchangeNode & node, Void &) {
//     auto step = symbol_mapper.map(*node.getStep());
//     return ExchangeNode::createPlanNode(id_allocator->nextId(), step);
// }

PlanNodePtr PlanCopierVisitor::visitExceptNode(ExceptNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return ExceptNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitExtremesNode(ExtremesNode & node, Void &) {
    auto step = symbol_mapper.map(*node.getStep());
    return ExtremesNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitFillingNode(FillingNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return FillingNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitFinalSampleNode(FinalSampleNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return FinalSampleNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitFinishSortingNode(FinishSortingNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return FinishSortingNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitFilterNode(FilterNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return FilterNode::createPlanNode(id_allocator->nextId(), step);
}


PlanNodePtr PlanCopierVisitor::visitIntersectNode(IntersectNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return IntersectNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitJoinNode(JoinNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return JoinNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitTableScanNode(TableScanNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return TableScanNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitLimitNode(LimitNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return LimitNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitLimitByNode(LimitByNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return LimitByNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitPartitionTopNNode(PartitionTopNNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return PartitionTopNNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitPartialSortingNode(PartialSortingNode & node, Void &) 
{
    auto step = symbol_mapper.map(*node.getStep());
    return PartialSortingNode::createPlanNode(id_allocator->nextId(), step);
} 

// PlanNodePtr PlanCopierVisitor::visitRemoteExchangeSourceNode(RemoteExchangeSourceNode & node, Void &) 
// {
//     auto step = symbol_mapper.map(*node.getStep());
//     return RemoteExchangeSourceNode::createPlanNode(id_allocator->nextId(), step);
// } 

PlanNodePtr PlanCopierVisitor::visitReadNothingNode(ReadNothingNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return ReadNothingNode::createPlanNode(id_allocator->nextId(), step);
} 

PlanNodePtr PlanCopierVisitor::visitTotalsHavingNode(TotalsHavingNode & node, Void &) 
{
    auto step = symbol_mapper.map(*node.getStep());
    return TotalsHavingNode::createPlanNode(id_allocator->nextId(), step);
} 

PlanNodePtr PlanCopierVisitor::visitTopNFilteringNode(TopNFilteringNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return TopNFilteringNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitSortingNode(SortingNode & node, DB::Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return SortingNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitOffsetNode(OffsetNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return OffsetNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitUnionNode(UnionNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return UnionNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitWindowNode(WindowNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return WindowNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitValuesNode(ValuesNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return ValuesNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitMergeSortingNode(MergeSortingNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return MergeSortingNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitMergingSortedNode(MergingSortedNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return MergingSortedNode::createPlanNode(id_allocator->nextId(), step);
}

PlanNodePtr PlanCopierVisitor::visitMergingAggregatedNode(MergingAggregatedNode & node, Void &)
{
    auto step = symbol_mapper.map(*node.getStep());
    return MergingAggregatedNode::createPlanNode(id_allocator->nextId(), step);
}

}
