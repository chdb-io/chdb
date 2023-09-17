#include <Optimizer/Rule/Patterns.h>
#include <Optimizer/Rule/Rewrite/TopNRules.h>
#include <Optimizer/Utils.h>
#include <Processors/QueryPlan/AggregatingStep.h>
#include <Processors/QueryPlan/ProjectionStep.h>
#include <Processors/QueryPlan/SortingStep.h>
#include <Processors/QueryPlan/TopNFilteringStep.h>

namespace DB
{
PatternPtr CreateTopNFilteringForAggregating::getPattern() const
{
    // TopN -> Aggregating -> !TopNFiltering
    return Patterns::topN()
        .withSingle(Patterns::aggregating().withSingle(Patterns::any().matching(
            [](const PlanNodePtr & node, auto &) { return node->getStep()->getType() != IQueryPlanStep::Type::TopNFiltering; })))
        .result();
}

TransformResult CreateTopNFilteringForAggregating::transformImpl(PlanNodePtr node, const Captures &, RuleContext & context)
{
    // TODO: in long term, TopNFilteringStep should be generated according to property and/or cost
    if (!context.context->getSettingsRef().enable_topn_filtering_optimization)
        return {};

    const auto & topn_step = dynamic_cast<const SortingStep &>(*node->getStep());
    auto & aggregate_node = node->getChildren()[0];
    const auto & aggregate_step = dynamic_cast<const AggregatingStep &>(*aggregate_node->getStep());

    NameSet agg_keys{aggregate_step.getKeys().begin(), aggregate_step.getKeys().end()};

    // if topN sort column is from an aggregate function, fail to rewrite
    for (const auto & sort_column_desc : topn_step.getSortDescription())
        if (!agg_keys.count(sort_column_desc.column_name))
            return {};

    const auto & aggregate_child_node = aggregate_node->getChildren()[0];

    auto topn_filter_step = std::make_shared<TopNFilteringStep>(
        aggregate_child_node->getStep()->getOutputStream(), topn_step.getSortDescription(), topn_step.getLimit(), TopNModel::DENSE_RANK);
    auto topn_filter_node = PlanNodeBase::createPlanNode(context.context->nextNodeId(), topn_filter_step, PlanNodes{aggregate_child_node});

    aggregate_node->replaceChildren(PlanNodes{topn_filter_node});
    return TransformResult{node};
}

PatternPtr PushTopNThroughProjection::getPattern() const
{
    return Patterns::topN().withSingle(Patterns::project()).result();
}

TransformResult PushTopNThroughProjection::transformImpl(PlanNodePtr node, const Captures &, RuleContext & rule_ctx)
{
    auto projection = node->getChildren()[0];
    auto source = projection->getChildren()[0];

    const auto * topn_step = dynamic_cast<const SortingStep *>(node->getStep().get());
    const auto * project_step = dynamic_cast<const ProjectionStep *>(projection->getStep().get());

    if (!project_step || Utils::canChangeOutputRows(*project_step, rule_ctx.context))
        return {};

    auto symbols_exist_in_source = Utils::extractIdentities(*project_step);
    auto new_sort = topn_step->getSortDescription();

    for (auto & sort_column_desc : new_sort)
    {
        if (!symbols_exist_in_source.count(sort_column_desc.column_name))
            return {};
        sort_column_desc.column_name = symbols_exist_in_source.at(sort_column_desc.column_name);
    }

    auto new_topn = PlanNodeBase::createPlanNode(
        node->getId(),
        std::make_shared<SortingStep>(
            source->getStep()->getOutputStream(),
            new_sort,
            topn_step->getLimit(),
            topn_step->isPartial(), 
            topn_step->getPrefixDescription()),
        {source});

    projection->replaceChildren({new_topn});
    projection->setStatistics(node->getStatistics());
    return TransformResult{projection};
}

PatternPtr PushTopNFilteringThroughProjection::getPattern() const
{
    return Patterns::topNFiltering().withSingle(Patterns::project()).result();
}

TransformResult PushTopNFilteringThroughProjection::transformImpl(PlanNodePtr node, const Captures &, RuleContext &)
{
    auto projection = node->getChildren()[0];
    auto source = projection->getChildren()[0];

    const auto * topn_filter_step = dynamic_cast<const TopNFilteringStep *>(node->getStep().get());
    const auto * project_step = dynamic_cast<const ProjectionStep *>(projection->getStep().get());

    auto symbols_exist_in_source = Utils::extractIdentities(*project_step);
    auto new_sort = topn_filter_step->getSortDescription();

    for (auto & sort_column_desc : new_sort)
    {
        if (!symbols_exist_in_source.count(sort_column_desc.column_name))
            return {};
        sort_column_desc.column_name = symbols_exist_in_source.at(sort_column_desc.column_name);
    }

    auto new_topn_filter = PlanNodeBase::createPlanNode(
        node->getId(),
        std::make_shared<TopNFilteringStep>(
            source->getStep()->getOutputStream(), new_sort, topn_filter_step->getSize(), topn_filter_step->getModel()),
        {source});

    projection->replaceChildren({new_topn_filter});
    projection->setStatistics(node->getStatistics());
    return TransformResult{projection};
}

}
