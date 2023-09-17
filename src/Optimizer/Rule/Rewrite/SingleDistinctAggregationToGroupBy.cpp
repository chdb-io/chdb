#include <Optimizer/Rule/Patterns.h>
#include <Optimizer/Rule/Rewrite/SingleDistinctAggregationToGroupBy.h>
#include <Processors/QueryPlan/AggregatingStep.h>
#include <Processors/QueryPlan/DistinctStep.h>
#include <AggregateFunctions/AggregateFunctionFactory.h>

namespace DB
{
PatternPtr SingleDistinctAggregationToGroupBy::getPattern() const
{
    return Patterns::aggregating().matchingStep<AggregatingStep>([&](const AggregatingStep & s) {
        if (!s.isNormal())
        {
            return false;
        }
        if (s.getAggregates().size() != 1)
        {
            return false;
        }

        static const std::set<String> distinct_func{"uniqexact", "countdistinct"};

        for (const auto & agg : s.getAggregates())
        {
            if (!agg.mask_column.empty())
                return false;
            if (!distinct_func.contains(Poco::toLower(agg.function->getName())))
                return false;
        }
        return true;
    }).result();
}

/**
 * Implements distinct aggregations with similar inputs by transforming plans of the following shape:
 * <pre>
 * - Aggregation
 *        GROUP BY (k)
 *        F1(DISTINCT s0, s1, ...),
 *        F2(DISTINCT s0, s1, ...),
 *     - X
 * </pre>
 * into
 * <pre>
 * - Aggregation
 *          GROUP BY (k)
 *          F1(x)
 *          F2(x)
 *      - Aggregation
 *             GROUP BY (k, s0, s1, ...)
 *          - X
 * </pre>
 * <p>
 * Assumes s0, s1, ... are symbol references (i.e., complex expressions have been pre-projected)
 */
TransformResult SingleDistinctAggregationToGroupBy::transformImpl(PlanNodePtr node, const Captures &, RuleContext & rule_context)
{
    if (!rule_context.context->getSettingsRef().enable_single_distinct_to_group_by)
    {
        return {};
    }
    auto step_ptr = node->getStep();
    const auto & step = dynamic_cast<const AggregatingStep &>(*step_ptr);

    auto symbols = step.getAggregates()[0].argument_names;
    auto group_by = step.getKeys();

    symbols.insert(symbols.begin(), group_by.begin(), group_by.end());

    AggregateDescriptions aggregate_descriptions;
    auto group_agg_step
        = std::make_shared<AggregatingStep>(node->getChildren()[0]->getStep()->getOutputStream(), symbols, aggregate_descriptions, GroupingSetsParamsList{}, true);
    auto group_agg_node = PlanNodeBase::createPlanNode(rule_context.context->nextNodeId(), std::move(group_agg_step), node->getChildren());

    std::unordered_map<String, DataTypePtr> name_to_type;
    for (const auto & item : group_agg_node->getStep()->getOutputStream().header)
    {
        name_to_type[item.name] = item.type;
    }
    AggregateDescriptions count_aggs;
    for (const auto & agg : step.getAggregates())
    {
        auto count_agg = agg;
        DataTypes arg_types;
        if (count_agg.argument_names.size() > 1)
            return {};
        for (const auto & arg : count_agg.argument_names)
        {
            if (!name_to_type.contains(arg))
            {
                // bug, logic err
                return {};
            }
            arg_types.emplace_back(name_to_type[arg]);
        }
        AggregateFunctionProperties properties;
        count_agg.function = AggregateFunctionFactory::instance().get("count", arg_types, {}, properties);
        count_aggs.emplace_back(count_agg);
    }

    auto count_agg_step = std::make_shared<AggregatingStep>(group_agg_node->getStep()->getOutputStream(), group_by, count_aggs, GroupingSetsParamsList{}, true);
    auto count_agg_node = PlanNodeBase::createPlanNode(rule_context.context->nextNodeId(), std::move(count_agg_step), {group_agg_node});
    return count_agg_node;
}

}
