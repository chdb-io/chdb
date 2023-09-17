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

#include <memory>
#include <Core/Names.h>
#include <Functions/IFunction.h>
// #include <Interpreters/DistributedStages/PlanSegment.h>
#include <Interpreters/WindowDescription.h>
#include <Processors/QueryPlan/AggregatingStep.h>
#include <Processors/QueryPlan/ApplyStep.h>
#include <Processors/QueryPlan/ArrayJoinStep.h>
#include <Processors/QueryPlan/AssignUniqueIdStep.h>
#include <Processors/QueryPlan/Assignment.h>
#include <Processors/QueryPlan/CreatingSetsStep.h>
#include <Processors/QueryPlan/CubeStep.h>
#include <Processors/QueryPlan/DistinctStep.h>
#include <Processors/QueryPlan/EnforceSingleRowStep.h>
#include <Processors/QueryPlan/ExceptStep.h>
#include <Processors/QueryPlan/ExpressionStep.h>
#include <Processors/QueryPlan/ExtremesStep.h>
#include <Processors/QueryPlan/FillingStep.h>
#include <Processors/QueryPlan/FilterStep.h>
#include <Processors/QueryPlan/FinalSampleStep.h>
#include <Processors/QueryPlan/FinishSortingStep.h>
#include <Processors/QueryPlan/IntersectStep.h>
#include <Processors/QueryPlan/JoinStep.h>
#include <Processors/QueryPlan/LimitByStep.h>
#include <Processors/QueryPlan/LimitStep.h>
#include <Processors/QueryPlan/MergeSortingStep.h>
#include <Processors/QueryPlan/MergingAggregatedStep.h>
#include <Processors/QueryPlan/MergingSortedStep.h>
#include <Processors/QueryPlan/OffsetStep.h>
#include <Processors/QueryPlan/PartialSortingStep.h>
#include <Processors/QueryPlan/PartitionTopNStep.h>
#include <Processors/QueryPlan/ProjectionStep.h>
// #include <Processors/QueryPlan/RemoteExchangeSourceStep.h>
#include <Optimizer/Property/Property.h>
#include <Processors/QueryPlan/RollupStep.h>
#include <Processors/QueryPlan/SortingStep.h>
#include <Processors/QueryPlan/SymbolAllocator.h>
#include <Processors/QueryPlan/TableScanStep.h>
#include <Processors/QueryPlan/TopNFilteringStep.h>
#include <Processors/QueryPlan/TotalsHavingStep.h>
#include <Processors/QueryPlan/UnionStep.h>
#include <Processors/QueryPlan/ValuesStep.h>
#include <Processors/QueryPlan/WindowStep.h>
#include "Interpreters/Context_fwd.h"
#include "QueryPlan/ReadNothingStep.h"


namespace DB
{
using ASTPtr = std::shared_ptr<IAST>;

// todo@kaixi: implement for all plan nodes
class SymbolMapper
{
public:
    using Symbol = std::string;
    using MappingFunction = std::function<std::string(const std::string &)>;

    explicit SymbolMapper(MappingFunction mapping_function_, ContextPtr context_ = nullptr)
        : mapping_function(std::move(mapping_function_)), context(context_)
    {
    }

    static SymbolMapper simpleMapper(std::unordered_map<Symbol, Symbol> & mapping);
    static SymbolMapper symbolMapper(const std::unordered_map<Symbol, Symbol> & mapping);
    static SymbolMapper
    symbolReallocator(std::unordered_map<Symbol, Symbol> & mapping, SymbolAllocator & symbolAllocator, ContextPtr context);

    std::string map(const std::string & symbol) { return mapping_function(symbol); }
    template <typename T>
    std::vector<T> map(const std::vector<T> & items)
    {
        std::vector<T> ret;
        std::transform(items.begin(), items.end(), std::back_inserter(ret), [&](const auto & param) { return map(param); });
        return ret;
    }

    NameSet mapToDistinct(const Names & symbols);
    NamesAndTypes map(const NamesAndTypes & name_and_types);
    NameSet map(const NameSet & names);

    NameToType map(const NameToType & name_to_type);
    NamesWithAliases map(const NamesWithAliases & name_with_aliases);
    Assignments map(const Assignments & assignments);
    Assignment map(const Assignment & assignment);
    Block map(const Block & name_and_types);
    DataStream map(const DataStream & data_stream);
    ASTPtr map(const ASTPtr & expr);
    ASTPtr map(const ConstASTPtr & expr);
    Partitioning map(const Partitioning & partition);
    AggregateDescription map(const AggregateDescription & desc);
    GroupingSetsParams map(const GroupingSetsParams & param);
    WindowFunctionDescription map(const WindowFunctionDescription & desc);
    WindowDescription map(const WindowDescription & desc);
    SortColumnDescription map(const SortColumnDescription & desc);
    Aggregator::Params map(const Aggregator::Params & params);
    AggregatingTransformParamsPtr map(const AggregatingTransformParamsPtr & param);
    ArrayJoinActionPtr map(const ArrayJoinActionPtr & array_join_action);

    GroupingDescription map(const GroupingDescription & desc);

    std::shared_ptr<AggregatingStep> map(const AggregatingStep & agg);
    std::shared_ptr<ApplyStep> map(const ApplyStep & apply);
    std::shared_ptr<ArrayJoinStep> map(const ArrayJoinStep & array_join);
    std::shared_ptr<AssignUniqueIdStep> map(const AssignUniqueIdStep & assign);
    std::shared_ptr<CreatingSetsStep> map(const CreatingSetsStep & creating_sets);
    std::shared_ptr<CubeStep> map(const CubeStep & cube);
    std::shared_ptr<DistinctStep> map(const DistinctStep & distinct);
    std::shared_ptr<EnforceSingleRowStep> map(const EnforceSingleRowStep & row);
    std::shared_ptr<ExpressionStep> map(const ExpressionStep & expression);
    std::shared_ptr<ExceptStep> map(const ExceptStep & except);
    std::shared_ptr<ExtremesStep> map(const ExtremesStep & extreme);
    // std::shared_ptr<ExchangeStep> map(const ExchangeStep & exchange);
    std::shared_ptr<FillingStep> map(const FillingStep & filling);
    std::shared_ptr<FinalSampleStep> map(const FinalSampleStep & final_sample);
    std::shared_ptr<FinishSortingStep> map(const FinishSortingStep & finish_sorting);
    std::shared_ptr<FilterStep> map(const FilterStep & filter);
    std::shared_ptr<IntersectStep> map(const IntersectStep & intersect);
    std::shared_ptr<JoinStep> map(const JoinStep & join);
    std::shared_ptr<LimitStep> map(const LimitStep & limit);
    std::shared_ptr<LimitByStep> map(const LimitByStep & limit);
    std::shared_ptr<MergeSortingStep> map(const MergeSortingStep & sorting);
    std::shared_ptr<MergingSortedStep> map(const MergingSortedStep & merging);
    std::shared_ptr<MergingAggregatedStep> map(const MergingAggregatedStep & merge_agg);
    std::shared_ptr<OffsetStep> map(const OffsetStep & offset);
    std::shared_ptr<ProjectionStep> map(const ProjectionStep & projection);
    std::shared_ptr<PartitionTopNStep> map(const PartitionTopNStep & partition_topn);
    std::shared_ptr<PartialSortingStep> map(const PartialSortingStep & partition_sorting);
    std::shared_ptr<ReadNothingStep> map(const ReadNothingStep & read_nothing);
    // std::shared_ptr<RemoteExchangeSourceStep> map(const RemoteExchangeSourceStep & remote_exchange);
    std::shared_ptr<SortingStep> map(const SortingStep & sorting);
    std::shared_ptr<TopNFilteringStep> map(const TopNFilteringStep & topn_filter);
    std::shared_ptr<TableScanStep> map(const TableScanStep & table_scan);
    std::shared_ptr<TotalsHavingStep> map(const TotalsHavingStep & total_having);
    std::shared_ptr<UnionStep> map(const UnionStep & union_);
    std::shared_ptr<ValuesStep> map(const ValuesStep & values);
    std::shared_ptr<WindowStep> map(const WindowStep & window);


private:
    MappingFunction mapping_function;
    ContextPtr context;
    class IdentifierRewriter;
};

}
