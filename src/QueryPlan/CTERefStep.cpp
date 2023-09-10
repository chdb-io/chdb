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

#include <QueryPlan/CTERefStep.h>

#include <Parsers/ASTIdentifier.h>
#include <Parsers/ASTSerDerHelper.h>
#include <QueryPlan/CTEInfo.h>
#include <QueryPlan/FilterStep.h>
#include <QueryPlan/PlanCopier.h>
#include <QueryPlan/PlanNode.h>
#include <QueryPlan/PlanSerDerHelper.h>
#include <QueryPlan/SymbolMapper.h>
#include <QueryPlan/ProjectionStep.h>
#include <IO/ReadHelpers.h>
#include <IO/VarInt.h>


namespace DB
{

CTERefStep::CTERefStep(DataStream output_, CTEId id_, std::unordered_map<String, String> output_columns_, bool has_filter_)
        : ISourceStep(std::move(output_)), id(id_), output_columns(std::move(output_columns_)), has_filter(has_filter_)
{
}

std::shared_ptr<IQueryPlanStep> CTERefStep::copy(ContextPtr) const
{
    return std::make_shared<CTERefStep>(output_stream.value(), id, output_columns, has_filter);
}

void CTERefStep::serialize(WriteBuffer & buffer) const
{
    serializeDataStream(output_stream.value(), buffer);
    writeBinary(id, buffer);

    writeVarUInt(output_columns.size(), buffer);
    for (const auto & item : output_columns)
    {
        writeStringBinary(item.first, buffer);
        writeStringBinary(item.second, buffer);
    }

    writeBinary(has_filter, buffer);
}

QueryPlanStepPtr CTERefStep::deserialize(ReadBuffer & buffer, ContextPtr)
{
    DataStream output_tmp = deserializeDataStream(buffer);
    
    CTEId id_tmp;
    readBinary(id_tmp, buffer);

    UInt64 output_columns_num;
    readVarUInt(output_columns_num, buffer);
    
    std::unordered_map<String, String> output_columns_tmp;
    for (size_t i = 0 ; i < output_columns_num ; ++i)
    {
        String elem1, elem2;
        readStringBinary(elem1, buffer);
        readStringBinary(elem2, buffer);
        output_columns_tmp[elem1] = elem2;
    }

    bool has_filter;
    readBinary(has_filter, buffer);

    return std::make_shared<CTERefStep>(output_tmp, id_tmp, output_columns_tmp, has_filter);
}


std::shared_ptr<ProjectionStep> CTERefStep::toProjectionStep() const
{
    NamesAndTypes inputs;
    Assignments assignments;
    NameToType name_to_type;
    for (const auto & item : output_stream.value().header)
    {
        if (output_columns.contains(item.name))
        {
            assignments.emplace_back(item.name, std::make_shared<ASTIdentifier>(output_columns.at(item.name)));
            name_to_type.emplace(item.name, item.type);
            inputs.emplace_back(NameAndTypePair{output_columns.at(item.name), item.type});
        }
    }
    return std::make_shared<ProjectionStep>(DataStream{inputs}, assignments, name_to_type);
}

PlanNodePtr CTERefStep::toInlinedPlanNode(CTEInfo & cte_info, ContextMutablePtr & context) const
{
    std::unordered_map<SymbolMapper::Symbol, SymbolMapper::Symbol> mapping;
    auto with_clause_plan = PlanCopier::copy(cte_info.getCTEDef(id), context, mapping);
    SymbolMapper mapper = SymbolMapper::symbolReallocator(mapping, *(context->getSymbolAllocator()), context);
    auto node = PlanNodeBase::createPlanNode(context->nextNodeId(), toProjectionStepWithNewSymbols(mapper), {with_clause_plan});
    return node;
}

std::shared_ptr<ProjectionStep> CTERefStep::toProjectionStepWithNewSymbols(SymbolMapper & mapper) const 
{
    NamesAndTypes inputs;
    Assignments assignments;
    NameToType name_to_type;
    for (const auto & item : output_stream.value().header)
    {
        if (output_columns.contains(item.name))
        {
            assignments.emplace_back(item.name, std::make_shared<ASTIdentifier>(mapper.map(output_columns.at(item.name))));
            name_to_type.emplace(item.name, item.type);
            inputs.emplace_back(NameAndTypePair{output_columns.at(item.name), item.type});
        }
    }
    return std::make_shared<ProjectionStep>(mapper.map(DataStream{inputs}), assignments, name_to_type);
}

std::unordered_map<String, String> CTERefStep::getReverseOutputColumns() const
{
    std::unordered_map<String, String> reverse;
    for (const auto & item : output_columns)
        reverse.emplace(item.second, item.first);
    return reverse;
}
}
