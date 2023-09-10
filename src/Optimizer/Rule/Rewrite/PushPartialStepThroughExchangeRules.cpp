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

#include <Optimizer/Rule/Rewrite/PushPartialStepThroughExchangeRules.h>

#include <Core/SortDescription.h>
#include <Optimizer/ExpressionDeterminism.h>
#include <Optimizer/PlanNodeCardinality.h>
#include <Optimizer/Rule/Pattern.h>
#include <Optimizer/Rule/Patterns.h>
#include <Optimizer/SymbolUtils.h>
#include <Optimizer/SymbolsExtractor.h>
#include <Processors/Transforms/AggregatingTransform.h>
#include <QueryPlan/ExchangeStep.h>
#include <QueryPlan/QueryPlan.h>
#include <QueryPlan/SymbolMapper.h>

namespace DB
{

NameSet PushPartialAggThroughExchange::BLOCK_AGGS{"pathCount", "attributionAnalysis", "attributionCorrelationMerge"};

PatternPtr PushPartialAggThroughExchange::getPattern() const
{
    return Patterns::aggregating().withSingle(Patterns::exchange()).result();
}

TransformResult split(const PlanNodePtr & node, RuleContext & context)
{
    const auto *step = dynamic_cast<const AggregatingStep *>(node->getStep().get());
    QueryPlanStepPtr partial_agg = std::make_shared<AggregatingStep>(
        node->getChildren()[0]->getStep()->getOutputStream(),
        step->getKeys(),
        step->getAggregates(),
        step->getGroupingSetsParams(),
        false,
        step->getGroupBySortDescription(),
        step->getGroupings(),
        false,
        context.context->getSettingsRef().distributed_aggregation_memory_efficient);


    auto partial_agg_node
        = PlanNodeBase::createPlanNode(context.context->nextNodeId(), std::move(partial_agg), node->getChildren(), node->getStatistics());


    ColumnNumbers keys;
    auto exchange_header = partial_agg_node->getStep()->getOutputStream().header;
    if (!step->getGroupingSetsParams().empty())
    {
        keys.push_back(exchange_header.getPositionByName("__grouping_set"));
    }

    for (const auto & key : step->getKeys())
    {
        keys.emplace_back(exchange_header.getPositionByName(key));
    }

    Aggregator::Params new_params(
        partial_agg_node->getStep()->getOutputStream().header, keys, step->getAggregates(), step->getParams().overflow_row, context.context->getSettingsRef().max_threads);
    auto transform_params = std::make_shared<AggregatingTransformParams>(new_params, step->isFinal());

    QueryPlanStepPtr final_agg = std::make_shared<MergingAggregatedStep>(
        partial_agg_node->getStep()->getOutputStream(),
        step->getKeys(),
        step->getGroupingSetsParams(),
        step->getGroupings(),
        transform_params,
        false,
        context.context->getSettingsRef().max_threads,
        context.context->getSettingsRef().aggregation_memory_efficient_merge_threads);
    auto final_agg_node
        = PlanNodeBase::createPlanNode(context.context->nextNodeId(), std::move(final_agg), {partial_agg_node}, node->getStatistics());
    return final_agg_node;
}

PlanNodePtr createPartial(const AggregatingStep * step, PlanNodePtr child, NameToNameMap & map, Context & context)
{
    auto symbol_mapper = SymbolMapper::symbolMapper(map);
    auto agg_step = symbol_mapper.map(*step);
    auto mapped_partial = PlanNodeBase::createPlanNode(context.nextNodeId(), agg_step, {std::move(child)});

    if (map.empty())
    {
        return mapped_partial;
    }

    Assignments assignments;
    NameToType name_to_type;
    bool is_identity = true;
    for (const auto & output : agg_step->getOutputStream().header)
    {
        auto input = symbol_mapper.map(output.name);
        assignments.emplace_back(output.name, std::make_shared<ASTIdentifier>(input));
        is_identity &= output.name == input;
        name_to_type[output.name] = output.type;
    }

    if (is_identity)
    {
        return mapped_partial;
    }
    return PlanNodeBase::createPlanNode(
        context.nextNodeId(), std::make_shared<ProjectionStep>(agg_step->getOutputStream(), assignments, name_to_type), {mapped_partial});
}

TransformResult pushPartial(const PlanNodePtr & node, RuleContext & context)
{
    const auto * step = dynamic_cast<const AggregatingStep *>(node->getStep().get());
    auto exchange_node = node->getChildren()[0];
    const auto * exchange_step = dynamic_cast<const ExchangeStep *>(exchange_node->getStep().get());

    PlanNodes partials;
    for (size_t index = 0; index < exchange_step->getInputStreams().size(); ++index)
    {
        NameToNameMap map;
        for (const auto & item : exchange_step->getOutToInputs())
        {
            if (item.first != item.second[index])
            {
                map[item.first] = item.second[index];
            }
        }

        auto projection = createPartial(step, exchange_node->getChildren()[index], map, *context.context);
        partials.emplace_back(projection);
    }

    DataStreams exchange_inputs;
    for (const auto & item : partials)
    {
        exchange_inputs.emplace_back(item->getCurrentDataStream());
    }

    return PlanNodeBase::createPlanNode(
        context.context->nextNodeId(),
        std::make_shared<ExchangeStep>(
            exchange_inputs, exchange_step->getExchangeMode(), exchange_step->getSchema(), exchange_step->needKeepOrder()),
        partials);
}

TransformResult PushPartialAggThroughExchange::transformImpl(PlanNodePtr node, const Captures &, RuleContext & context)
{
    if (!context.context->getSettingsRef().enable_push_partial_agg)
        return {};
    const auto * step = dynamic_cast<const AggregatingStep *>(node->getStep().get());

    for (const auto & agg : step->getAggregates())
    {
        if (BLOCK_AGGS.count(agg.function->getName()))
        {
            return {};
        }
    }
    // TODO if aggregate data is almost pure distinct, then partial aggregate is useless.
    // Consider these two cases
    // 1: if group by keys is or contains primary key (may be need function dependency ...), then don't push partial aggregate.
    // 2: for aggregate on base table, the statistics is accurate enough to perform cost-based push down. while other
    // cases, we can't relay on statistics.
    //    auto group_keys = step->getKeys();
    //    if (!group_keys.empty())
    //    {
    //        PlanNodeStatisticsPtr statistics = CardinalityEstimator::estimate(node->getChildren()[0], context.context);
    //        UInt64 row_count = statistics->getRowCount() == 0 ? 1 : statistics->getRowCount();
    //        UInt64 group_keys_ndv = 1;
    //        for (auto & key : group_keys)
    //        {
    //            SymbolStatisticsPtr symbol_stats = statistics->getSymbolStatistics(key);
    //            group_keys_ndv = group_keys_ndv * symbol_stats->getNdv();
    //        }
    //        double density = (double)group_keys_ndv / row_count;
    //        if (density > context.context.getSettingsRef().enable_partial_aggregate_ratio)
    //        {
    //            return {};
    //        }
    //    }

    if(step->isFinal())
        return split(node, context);
    else
        return pushPartial(node, context);
}

PatternPtr PushPartialAggThroughUnion::getPattern() const
{
    return Patterns::aggregating()
        .matchingStep<AggregatingStep>([](const AggregatingStep & step) { return step.isPartial(); })
        .withSingle(Patterns::unionn()).result();
}

TransformResult PushPartialAggThroughUnion::transformImpl(PlanNodePtr node, const Captures &, RuleContext & context)
{
    const auto * step = dynamic_cast<const AggregatingStep *>(node->getStep().get());
    auto union_node = node->getChildren()[0];
    const auto * union_step = dynamic_cast<const UnionStep *>(union_node->getStep().get());

    PlanNodes partials;
    DataStreams union_inputs;
    for (size_t index = 0; index < union_step->getInputStreams().size(); ++index)
    {
        NameToNameMap map;
        for (const auto & item : union_step->getOutToInputs())
        {
            if (item.first != item.second[index])
            {
                map[item.first] = item.second[index];
            }
        }

        auto projection = createPartial(step, union_node->getChildren()[index], map, *context.context);

        partials.emplace_back(projection);
        union_inputs.emplace_back(projection->getCurrentDataStream());
    }

    NameToNameMap map;
    for (const auto & item : union_step->getOutToInputs())
    {
        map[item.second[0]] = item.first;
    }
    auto mapper = SymbolMapper::symbolMapper(map);

    DataStream output;
    for (const auto & item : union_inputs[0].header)
    {
        output.header.insert(ColumnWithTypeAndName{item.type, mapper.map(item.name)});
    }


    return PlanNodeBase::createPlanNode(
        context.context->nextNodeId(), std::make_shared<UnionStep>(union_inputs, output, union_step->isLocal()), partials);
}

PatternPtr PushPartialSortingThroughExchange::getPattern() const
{
    return Patterns::sorting().withSingle(Patterns::exchange().matchingStep<ExchangeStep>(
        [](const ExchangeStep & step) { return step.getExchangeMode() == ExchangeMode::GATHER; })).result();
}

TransformResult PushPartialSortingThroughExchange::transformImpl(PlanNodePtr node, const Captures &, RuleContext & context)
{
    const auto * step = dynamic_cast<const SortingStep *>(node->getStep().get());
    auto old_exchange_node = node->getChildren()[0];
    const auto * old_exchange_step = dynamic_cast<const ExchangeStep *>(old_exchange_node->getStep().get());

    PlanNodes exchange_children;
    for (size_t index = 0; index < old_exchange_node->getChildren().size(); index++)
    {
        auto exchange_child = old_exchange_node->getChildren()[index];
        if (dynamic_cast<SortingNode *>(exchange_child.get()))
        {
            return {};
        }

        SortDescription new_sort_desc;
        for (const auto & desc : step->getSortDescription())
        {
            auto new_desc = desc;
            new_desc.column_name = old_exchange_step->getOutToInputs().at(desc.column_name).at(index);
            new_sort_desc.emplace_back(new_desc);
        }

        auto before_exchange_sort = std::make_unique<SortingStep>(
            exchange_child->getStep()->getOutputStream(), new_sort_desc, step->getLimit(), true, SortDescription{});
        PlanNodes children{exchange_child};
        auto before_exchange_sort_node
            = PlanNodeBase::createPlanNode(context.context->nextNodeId(), std::move(before_exchange_sort), children, node->getStatistics());
        exchange_children.emplace_back(before_exchange_sort_node);
    }

    auto exchange_step = old_exchange_step->copy(context.context);
    auto exchange_node = PlanNodeBase::createPlanNode(
        context.context->nextNodeId(), std::move(exchange_step), exchange_children, old_exchange_node->getStatistics());

    QueryPlanStepPtr final_sort = step->copy(context.context);
    PlanNodes exchange{exchange_node};
    auto final_sort_node
        = PlanNodeBase::createPlanNode(context.context->nextNodeId(), std::move(final_sort), exchange, node->getStatistics());
    return final_sort_node;
}

static bool isLimitNeeded(const LimitStep & limit, const PlanNodePtr & node)
{
    auto range = PlanNodeCardinality::extractCardinality(*node);
    return range.upperBound > limit.getLimit() + limit.getOffset();
}

PatternPtr PushPartialLimitThroughExchange::getPattern() const
{
    return Patterns::limit().withSingle(Patterns::exchange().matchingStep<ExchangeStep>(
        [](const ExchangeStep & step) { return step.getExchangeMode() == ExchangeMode::GATHER; })).result();
}

TransformResult PushPartialLimitThroughExchange::transformImpl(PlanNodePtr node, const Captures &, RuleContext & context)
{
    const auto * step = dynamic_cast<const LimitStep *>(node->getStep().get());
    auto old_exchange_node = node->getChildren()[0];
    const auto * old_exchange_step = dynamic_cast<const ExchangeStep *>(old_exchange_node->getStep().get());

    PlanNodes exchange_children;
    bool should_apply = false;
    for (const auto & exchange_child : old_exchange_node->getChildren())
    {
        if (isLimitNeeded(*step, exchange_child))
        {
            auto partial_limit = std::make_unique<LimitStep>(
                exchange_child->getStep()->getOutputStream(),
                step->getLimit() + step->getOffset(),
                0,
                false,
                false,
                step->getSortDescription(),
                true);
            auto partial_limit_node = PlanNodeBase::createPlanNode(
                context.context->nextNodeId(), std::move(partial_limit), PlanNodes{exchange_child}, node->getStatistics());
            exchange_children.emplace_back(partial_limit_node);

            should_apply = true;
        }
    }

    if (!should_apply)
        return {};

    auto exchange_step = old_exchange_step->copy(context.context);
    auto exchange_node = PlanNodeBase::createPlanNode(
        context.context->nextNodeId(), std::move(exchange_step), exchange_children, old_exchange_node->getStatistics());

    node->replaceChildren({exchange_node});
    return node;
}

PatternPtr PushPartialDistinctThroughExchange::getPattern() const
{
    return Patterns::distinct()
        .matchingStep<DistinctStep>([](const DistinctStep & step) { return !step.preDistinct(); })
        .withSingle(Patterns::exchange()).result();
}

TransformResult PushPartialDistinctThroughExchange::transformImpl(PlanNodePtr node, const Captures &, RuleContext & context)
{
    const auto * step = dynamic_cast<const DistinctStep *>(node->getStep().get());
    auto old_exchange_node = node->getChildren()[0];
    const auto * old_exchange_step = dynamic_cast<const ExchangeStep *>(old_exchange_node->getStep().get());
    if (dynamic_cast<const DistinctStep *>(old_exchange_node->getChildren()[0]->getStep().get()))
    {
        return {};
    }

    PlanNodes exchange_children;
    for (const auto & exchange_child : old_exchange_node->getChildren())
    {
        auto partial_limit = std::make_unique<DistinctStep>(
            exchange_child->getStep()->getOutputStream(), step->getSetSizeLimits(), step->getLimitHint(), step->getColumns(), true);
        auto partial_limit_node = PlanNodeBase::createPlanNode(
            context.context->nextNodeId(), std::move(partial_limit), PlanNodes{exchange_child}, node->getStatistics());
        exchange_children.emplace_back(partial_limit_node);
    }
    auto exchange_step = old_exchange_step->copy(context.context);
    auto exchange_node = PlanNodeBase::createPlanNode(
        context.context->nextNodeId(), std::move(exchange_step), exchange_children, old_exchange_node->getStatistics());

    node->replaceChildren({exchange_node});
    return node;
}

}
