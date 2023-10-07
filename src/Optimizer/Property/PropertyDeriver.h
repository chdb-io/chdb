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

#include <Optimizer/Property/Property.h>
#include <Processors/QueryPlan/PlanVisitor.h>

#include <utility>

namespace DB
{
class PropertyDeriver
{
public:
    static Property deriveProperty(QueryPlanStepPtr step, ContextMutablePtr & context);
    static Property deriveProperty(QueryPlanStepPtr step, Property & input_property, ContextMutablePtr & context);
    static Property deriveProperty(QueryPlanStepPtr step, PropertySet & input_properties, ContextMutablePtr & context);
    static Property deriveStorageProperty(const StoragePtr& storage, ContextMutablePtr & context);
};

class DeriverContext
{
public:
    DeriverContext(PropertySet input_properties_, ContextMutablePtr & context_)
        : input_properties(std::move(input_properties_)), context(context_) { }
    const PropertySet & getInput() { return input_properties; }
    ContextMutablePtr & getContext() { return context; }

private:
    PropertySet input_properties;
    ContextMutablePtr & context;
};

class DeriverVisitor : public StepVisitor<Property, DeriverContext>
{
public:
    Property visitStep(const IQueryPlanStep &, DeriverContext &) override;

    Property visitProjectionStep(const ProjectionStep & step, DeriverContext & context) override;
    Property visitFilterStep(const FilterStep &, DeriverContext & context) override;
    Property visitJoinStep(const JoinStep & step, DeriverContext & context) override;
    Property visitArrayJoinStep(const ArrayJoinStep & step, DeriverContext & context) override;
    Property visitAggregatingStep(const AggregatingStep & step, DeriverContext & context) override;
    Property visitMarkDistinctStep(const MarkDistinctStep & step, DeriverContext & context) override;
    Property visitMergingAggregatedStep(const MergingAggregatedStep &, DeriverContext & context) override;
    Property visitUnionStep(const UnionStep & step, DeriverContext & context) override;
    Property visitExceptStep(const ExceptStep &, DeriverContext & context) override;
    Property visitIntersectStep(const IntersectStep &, DeriverContext & context) override;
    // Property visitExchangeStep(const ExchangeStep & step, DeriverContext & context) override;
    // Property visitRemoteExchangeSourceStep(const RemoteExchangeSourceStep &, DeriverContext & context) override;
    Property visitTableScanStep(const TableScanStep &, DeriverContext &) override;
    Property visitReadNothingStep(const ReadNothingStep &, DeriverContext &) override;
    Property visitValuesStep(const ValuesStep &, DeriverContext &) override;
    Property visitLimitStep(const LimitStep &, DeriverContext & context) override;
    Property visitLimitByStep(const LimitByStep &, DeriverContext & context) override;
    Property visitSortingStep(const SortingStep &, DeriverContext & context) override;
    Property visitMergeSortingStep(const MergeSortingStep &, DeriverContext & context) override;
    Property visitPartialSortingStep(const PartialSortingStep &, DeriverContext & context) override;
    Property visitMergingSortedStep(const MergingSortedStep &, DeriverContext & context) override;
    Property visitDistinctStep(const DistinctStep &, DeriverContext & context) override;
    Property visitExtremesStep(const ExtremesStep &, DeriverContext & context) override;
    Property visitWindowStep(const WindowStep &, DeriverContext & context) override;
    Property visitApplyStep(const ApplyStep &, DeriverContext & context) override;
    Property visitEnforceSingleRowStep(const EnforceSingleRowStep &, DeriverContext & context) override;
    Property visitAssignUniqueIdStep(const AssignUniqueIdStep &, DeriverContext & context) override;
    Property visitCTERefStep(const CTERefStep &, DeriverContext & context) override;
    Property visitExplainAnalyzeStep(const ExplainAnalyzeStep &, DeriverContext & context) override;
    Property visitTopNFilteringStep(const TopNFilteringStep &, DeriverContext & context) override;
    Property visitFillingStep(const FillingStep &, DeriverContext & context) override;
};

}
