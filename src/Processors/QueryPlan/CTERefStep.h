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

#include <Processors/QueryPlan/ISourceStep.h>
#include <Processors/QueryPlan/ProjectionStep.h>
#include <Processors/QueryPlan/SymbolMapper.h>

#include <memory>

namespace DB
{
using CTEId = UInt32;
class CTEInfo;
class PlanNodeBase;
using PlanNodePtr = std::shared_ptr<PlanNodeBase>;

/**
 * CTE is model as two parts: CTERef and CTEDef.
 * CTERefStep is a source node reference to CTEDef by id.
 * CTEDef is a virtual node, the plan is stored in CTEInfo.
 */
class CTERefStep : public ISourceStep
{
public:
    CTERefStep(DataStream output_, CTEId id_, std::unordered_map<String, String> output_columns_, bool has_filter_);

    CTEId getId() const { return id; }
    const std::unordered_map<String, String> & getOutputColumns() const { return output_columns; }
    std::unordered_map<String, String> getReverseOutputColumns() const;

    void initializePipeline(QueryPipelineBuilder &, const BuildQueryPipelineSettings &) override
    {
        throw Exception("Not supported", ErrorCodes::NOT_IMPLEMENTED);
    }
    String getName() const override { return "CTERef"; }
    Type getType() const override { return Type::CTERef; }
    std::shared_ptr<IQueryPlanStep> copy(ContextPtr context) const override;
    bool hasFilter() const { return has_filter; }
    void setFilter(bool has_filter_) { has_filter = has_filter_;}
    void serialize(WriteBuffer &) const override;
    static QueryPlanStepPtr deserialize(ReadBuffer & buffer, ContextPtr context);

    std::shared_ptr<ProjectionStep> toProjectionStep() const;
    std::shared_ptr<ProjectionStep> toProjectionStepWithNewSymbols(SymbolMapper & mapper) const;
    PlanNodePtr toInlinedPlanNode(CTEInfo & cte_info, ContextMutablePtr & context) const;

private:
    /**
     * CTE id reference to CTEInfo in QueryPlan.
     */
    CTEId id;

    /**
     * Map of output column name to cte column name.
     */
    std::unordered_map<String, String> output_columns;

    /**
     * CTE follows a filter.
     */
    bool has_filter;
};
}
