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

#include <Optimizer/ImplementSetOperation.h>

#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <Interpreters/ExpressionActions.h>
#include <QueryPlan/AggregatingStep.h>
#include <QueryPlan/UnionStep.h>

namespace DB
{
String SetOperationNodeTranslator::MARKER = "marker";

TranslationResult SetOperationNodeTranslator::makeSetContainmentPlanForDistinct(PlanNodeBase & node)
{
    Names markers = allocateSymbols(node.getChildren().size(), MARKER);
    // identity projection for all the fields in each of the sources plus marker columns
    PlanNodes with_markers = appendMarkers(markers, node.getChildren(), node.shared_from_this());

    // add a union over all the rewritten sources. The outputs of the union have the same name as the
    // original intersect node
    NamesAndTypes output;
    for (const auto & item : node.getStep()->getOutputStream().header)
    {
        output.emplace_back(item.name, item.type);
    }
    for (auto & marker : markers)
    {
        output.emplace_back(marker, std::make_shared<DataTypeUInt8>());
    }
    PlanNodePtr union_node = unionNodes(with_markers, output);

    // add count aggregations
    auto aggregation_outputs = allocateSymbols(markers.size(), "count");

    Names group_by_keys;
    for (const auto & item : node.getStep()->getOutputStream().header)
        group_by_keys.emplace_back(item.name);

    AggregateDescriptions aggregates;
    for (size_t i = 0; i < markers.size(); i++)
    {
        AggregateDescription aggregate_desc;
        aggregate_desc.column_name = aggregation_outputs[i];
        aggregate_desc.argument_names = {markers[i]};
        DataTypes types{std::make_shared<DataTypeUInt8>()};
        Array params;
        AggregateFunctionProperties properties;
        aggregate_desc.function = AggregateFunctionFactory::instance().get("sum", types, params, properties);

        aggregates.push_back(aggregate_desc);
    }

    auto agg_step = std::make_shared<AggregatingStep>(
        union_node->getStep()->getOutputStream(), std::move(group_by_keys), std::move(aggregates), GroupingSetsParamsList{}, true);
    PlanNodes children{union_node};
    PlanNodePtr agg_node = std::make_shared<AggregatingNode>(context.nextNodeId(), std::move(agg_step), children);

    return TranslationResult{agg_node, aggregation_outputs, std::make_optional<String>()};
}

TranslationResult SetOperationNodeTranslator::makeSetContainmentPlanForDistinctAll(PlanNodeBase & node)
{
    Names markers = allocateSymbols(node.getChildren().size(), MARKER);
    // identity projection for all the fields in each of the sources plus marker columns
    PlanNodes with_markers = appendMarkers(markers, node.getChildren(), node.shared_from_this());

    // add a union over all the rewritten sources.
    NamesAndTypes output_with_mark;
    Names output;
    for (const auto & item : node.getStep()->getOutputStream().header)
    {
        output_with_mark.emplace_back(item.name, item.type);
        output.emplace_back(item.name);
    }
    for (auto & marker : markers)
    {
        output_with_mark.emplace_back(marker, std::make_shared<DataTypeUInt8>());
    }

    PlanNodePtr union_node = unionNodes(with_markers, output_with_mark);

    // add counts and row number
    Names marker_outputs = allocateSymbols(markers.size(), "count");
    String row_number_symbol = context.getSymbolAllocator()->newSymbol("row_number");
    PlanNodePtr window = appendCounts(union_node, output, markers, marker_outputs, row_number_symbol);

    Assignments assignments;
    NameToType name_to_type;
    for (const auto & symbol : node.getStep()->getOutputStream().header)
    {
        if (!assignments.count(symbol.name))
        {
            assignments.emplace_back(symbol.name, std::make_shared<ASTIdentifier>(symbol.name));
            name_to_type[symbol.name] = symbol.type;
        }
    }
    for (auto & symbol : marker_outputs)
    {
        if (!assignments.count(symbol))
        {
            assignments.emplace_back(symbol, std::make_shared<ASTIdentifier>(symbol));
            name_to_type[symbol] = std::make_shared<DataTypeUInt64>();
        }
    }
    if (!assignments.count(row_number_symbol))
    {
        assignments.emplace_back(row_number_symbol, std::make_shared<ASTIdentifier>(row_number_symbol));
        name_to_type[row_number_symbol] = std::make_shared<DataTypeUInt64>();
    }

    // prune markers
    auto step = std::make_shared<ProjectionStep>(window->getStep()->getOutputStream(), assignments, name_to_type);
    PlanNodePtr project_node = PlanNodeBase::createPlanNode(context.nextNodeId(), std::move(step), PlanNodes{window});

    return TranslationResult{project_node, marker_outputs, std::make_optional<String>(row_number_symbol)};
}

PlanNodePtr SetOperationNodeTranslator::unionNodes(const PlanNodes & children, const NamesAndTypes cols)
{
    DataStreams input_stream;
    for (const auto & item : children)
        input_stream.emplace_back(item->getStep()->getOutputStream());
    DataStream output;
    for (const auto & col : cols)
    {
        output.header.insert(ColumnWithTypeAndName{col.type, col.name});
    }
    auto union_step = std::make_unique<UnionStep>(input_stream, output, false);

    PlanNodePtr union_node = std::make_shared<UnionNode>(context.nextNodeId(), std::move(union_step), children);
    return union_node;
}

PlanNodePtr SetOperationNodeTranslator::appendCounts(
    PlanNodePtr sourceNode,
    Names & originalColumns,
    std::vector<String> & markers,
    std::vector<String> & countOutputs,
    String & row_number_symbol)
{
    WindowDescription desc;
    for (const auto & column : originalColumns)
    {
        desc.partition_by.push_back(SortColumnDescription(column, 1 /* direction */, 1 /* nulls_direction */));
    }
    for (const auto & column : originalColumns)
    {
        desc.full_sort_description.push_back(SortColumnDescription(column, 1 /* direction */, 1 /* nulls_direction */));
    }
    WindowFrame default_frame{
        true, WindowFrame::FrameType::Range, WindowFrame::BoundaryType::Unbounded, 0, true, WindowFrame::BoundaryType::Current, 0, false};
    desc.frame = default_frame;

    std::vector<WindowFunctionDescription> functions;
    for (size_t i = 0; i < markers.size(); i++)
    {
        String output = countOutputs.at(i);

        Names argument_names = {markers[i]};
        DataTypes types{std::make_shared<DataTypeUInt8>()};
        Array params;
        AggregateFunctionProperties properties;
        AggregateFunctionPtr aggregate_function = AggregateFunctionFactory::instance().get("sum", types, params, properties);
        WindowFunctionDescription function{output, nullptr, aggregate_function, params, types, argument_names};
        functions.emplace_back(function);
    }
    {
        Names argument_names;
        DataTypes types;
        Array params;
        AggregateFunctionProperties properties;

        AggregateFunctionPtr aggregate_function = AggregateFunctionFactory::instance().get("row_number", types, params, properties);
        WindowFunctionDescription function{row_number_symbol, nullptr, aggregate_function, params, types, argument_names};
        functions.emplace_back(function);
    }

    auto step = std::make_shared<WindowStep>(sourceNode->getStep()->getOutputStream(), desc, functions, true);
    return PlanNodeBase::createPlanNode(context.nextNodeId(), std::move(step), PlanNodes{sourceNode});
};

}
