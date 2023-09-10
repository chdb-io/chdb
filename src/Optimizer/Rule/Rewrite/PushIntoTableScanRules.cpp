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

#include <Optimizer/Rule/Rewrite/PushIntoTableScanRules.h>

#include <Optimizer/ExpressionDeterminism.h>
#include <Optimizer/PredicateUtils.h>
#include <Optimizer/Rule/Patterns.h>
#include <Optimizer/SymbolsExtractor.h>
#include <QueryPlan/SymbolMapper.h>
#include <QueryPlan/TableScanStep.h>
#include <QueryPlan/LimitStep.h>

namespace DB
{

namespace
{
    bool isOptimizerProjectionSupportEnabled(RuleContext & rule_context)
    {
        return rule_context.context->getSettingsRef().optimizer_projection_support;
    }
}

PatternPtr PushQueryInfoFilterIntoTableScan::getPattern() const
{
    return Patterns::filter().withSingle(
        Patterns::tableScan().matchingStep<TableScanStep>([](const auto & step) { return !step.hasQueryInfoFilter(); })).result();
}

TransformResult PushQueryInfoFilterIntoTableScan::transformImpl(PlanNodePtr node, const Captures &, RuleContext & rule_context)
{
    if (isOptimizerProjectionSupportEnabled(rule_context))
        return {};

    auto table_scan = node->getChildren()[0];

    const auto * filter_step = dynamic_cast<const FilterStep *>(node->getStep().get());
    auto filter_conjuncts = PredicateUtils::extractConjuncts(filter_step->getFilter());
    auto copy_table_step = table_scan->getStep()->copy(rule_context.context);

    if (!pushQueryInfoFilter(dynamic_cast<TableScanStep &>(*copy_table_step), filter_conjuncts, rule_context.context))
        return {}; // repeat calls

    table_scan->setStep(copy_table_step);

    auto remaining_filters = removeStorageFilter(filter_conjuncts);
    if (remaining_filters.size() == filter_conjuncts.size())
        return {};

    ConstASTPtr new_predicate = PredicateUtils::combineConjuncts(remaining_filters);
    if (PredicateUtils::isTruePredicate(new_predicate))
        return table_scan;

    auto new_filter_step
        = std::make_shared<FilterStep>(table_scan->getStep()->getOutputStream(), new_predicate, filter_step->removesFilterColumn());
    return PlanNodeBase::createPlanNode(
        rule_context.context->nextNodeId(), std::move(new_filter_step), PlanNodes{table_scan}, node->getStatistics());
}

bool PushQueryInfoFilterIntoTableScan::pushQueryInfoFilter(TableScanStep & table_step, const std::vector<ConstASTPtr> & filter_conjuncts,
                                                           ContextPtr context)
{
    auto pushdown_filters = extractPushDownFilter(filter_conjuncts, context);
    if (!pushdown_filters.empty())
    {
        std::unordered_map<String, String> inv_alias;
        for (auto & item : table_step.getColumnAlias())
            inv_alias.emplace(item.second, item.first);

        auto mapper = SymbolMapper::simpleMapper(inv_alias);

        std::vector<ConstASTPtr> conjuncts;
        for (auto & filter : pushdown_filters)
        {
            bool all_in = true;
            auto symbols = SymbolsExtractor::extract(filter);
            for (const auto & item : symbols)
                all_in &= inv_alias.contains(item);

            if (all_in)
                conjuncts.emplace_back(mapper.map(filter));
        }

        bool applied = table_step.setQueryInfoFilter(conjuncts);
        if (!applied)
            return false;
    }

    return true;
}

std::vector<ConstASTPtr> PushQueryInfoFilterIntoTableScan::extractPushDownFilter(const std::vector<ConstASTPtr> & conjuncts, ContextPtr context)
{
    std::vector<ConstASTPtr> filters;
    for (auto & conjunct : conjuncts)
    {
        if (auto dynamic_filter = DynamicFilters::extractDescription(conjunct))
        {
            auto & description = dynamic_filter.value();
            if (!DynamicFilters::isSupportedForTableScan(description))
                continue;
        }

        if (!ExpressionDeterminism::isDeterministic(conjunct, context))
            continue;

        filters.emplace_back(conjunct);
    }
    return filters;
}

std::vector<ConstASTPtr> PushQueryInfoFilterIntoTableScan::removeStorageFilter(const std::vector<ConstASTPtr> & conjuncts)
{
    std::vector<ConstASTPtr> remove_array_set_check;
    for (const auto & conjunct : conjuncts)
    {
        // Attention !!!
        // arraySetCheck must push into storage, it is not executable in engine.
        if (conjunct->as<ASTFunction>())
        {
            const ASTFunction & fun = conjunct->as<const ASTFunction &>();
            if (fun.name == "arraySetCheck")
            {
                continue;
            }
        }
        remove_array_set_check.emplace_back(conjunct);
    }
    return remove_array_set_check;
}

PatternPtr PushLimitIntoTableScan::getPattern() const
{
    return Patterns::limit()
        .matchingStep<LimitStep>([](auto const & limit_step) { return !limit_step.isAlwaysReadTillEnd(); })
        .withSingle(Patterns::tableScan().matchingStep<TableScanStep>(
            [](const auto & step) { return !step.getPushdownAggregation() && !step.getPushdownFilter() && !step.hasQueryInfoFilter(); })).result();
}

TransformResult PushLimitIntoTableScan::transformImpl(PlanNodePtr node, const Captures &, RuleContext & rule_context)
{
    const auto * limit_step = dynamic_cast<const LimitStep *>(node->getStep().get());
    auto table_scan = node->getChildren()[0];

    auto copy_table_step = table_scan->getStep()->copy(rule_context.context);

    auto * table_step = dynamic_cast<TableScanStep *>(copy_table_step.get());
    bool applied = table_step->setLimit(limit_step->getLimit() + limit_step->getOffset(), rule_context.context);
    if (!applied)
        return {}; // repeat calls

    table_scan->setStep(copy_table_step);
    node->replaceChildren({table_scan});
    return node;
}


PatternPtr PushAggregationIntoTableScan::getPattern() const
{
    return Patterns::aggregating()
               .matchingStep<AggregatingStep>([](auto & step) { return step.isPartial() && !step.isGroupingSet(); })
               .withSingle(Patterns::tableScan()).result();
}

TransformResult PushAggregationIntoTableScan::transformImpl(PlanNodePtr node, const Captures &, RuleContext & rule_context)
{
    if (!isOptimizerProjectionSupportEnabled(rule_context))
        return {};

    auto copy_step = node->getChildren()[0]->getStep()->copy(rule_context.context);
    auto *copy_table_step = dynamic_cast<TableScanStep *>(copy_step.get());

    // TODO: combine aggregates if grouping keys are the same
    chassert(copy_table_step != nullptr);
    // coverity[var_deref_model]
    if (copy_table_step->getPushdownAggregation())
        return {};

    copy_table_step->setPushdownAggregation(node->getStep()->copy(rule_context.context));
    copy_table_step->formatOutputStream();
    return PlanNodeBase::createPlanNode(rule_context.context->nextNodeId(), std::move(copy_step), {}, node->getStatistics());
}

PatternPtr PushProjectionIntoTableScan::getPattern() const
{
    return Patterns::project().withSingle(
               Patterns::tableScan().matchingStep<TableScanStep>(
                   [](const auto & step) { return !step.getPushdownAggregation(); })).result();
}

TransformResult PushProjectionIntoTableScan::transformImpl(PlanNodePtr node, const Captures &, RuleContext & rule_context)
{
    if (!isOptimizerProjectionSupportEnabled(rule_context))
        return {};

    auto copy_step = node->getChildren()[0]->getStep()->copy(rule_context.context);
    auto *copy_table_step = dynamic_cast<TableScanStep *>(copy_step.get());

    // TODO: inline projection
    chassert(copy_table_step != nullptr);
    // coverity[var_deref_model]
    if (copy_table_step->getPushdownProjection())
        return {};

    copy_table_step->setPushdownProjection(node->getStep()->copy(rule_context.context));
    copy_table_step->formatOutputStream();
    return PlanNodeBase::createPlanNode(rule_context.context->nextNodeId(), std::move(copy_step), {}, node->getStatistics());
}

PatternPtr PushFilterIntoTableScan::getPattern() const
{
    return Patterns::filter().withSingle(
               Patterns::tableScan().matchingStep<TableScanStep>(
                   [](const auto & step) { return !step.getPushdownProjection() && !step.getPushdownAggregation(); })).result();
}

TransformResult PushFilterIntoTableScan::transformImpl(PlanNodePtr node, const Captures &, RuleContext & rule_context)
{
    if (!isOptimizerProjectionSupportEnabled(rule_context))
        return {};

    auto copy_step = node->getChildren()[0]->getStep()->copy(rule_context.context);
    auto *copy_table_step = dynamic_cast<TableScanStep *>(copy_step.get());
    chassert(copy_table_step != nullptr);

    // in case of TableScan has already a pushdown filter, combine them into one
    if (const auto * pushdown_filter_step = copy_table_step->getPushdownFilterCast())
    {
        const auto & old_pushdown_filter = pushdown_filter_step->getFilter();
        const auto & filter_step_filter = dynamic_cast<const FilterStep *>(node->getStep().get())->getFilter();

        auto new_pushdown_filter = PredicateUtils::combineConjuncts(ConstASTs{old_pushdown_filter, filter_step_filter});
        copy_table_step->setPushdownFilter(std::make_shared<FilterStep>(pushdown_filter_step->getInputStreams()[0], new_pushdown_filter));
    }
    else
    {
        copy_table_step->setPushdownFilter(node->getStep()->copy(rule_context.context));
    }

    // coverity[var_deref_model]
    copy_table_step->formatOutputStream();
    return PlanNodeBase::createPlanNode(rule_context.context->nextNodeId(), std::move(copy_step), {}, node->getStatistics());
}

}
