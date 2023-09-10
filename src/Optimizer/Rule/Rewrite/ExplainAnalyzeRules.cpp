#include <Optimizer/Rule/Rewrite/ExplainAnalyzeRules.h>

#include <Optimizer/CardinalityEstimate/CardinalityEstimator.h>
#include <Optimizer/Rule/Patterns.h>
#include <Optimizer/SymbolsExtractor.h>
#include <QueryPlan/SymbolMapper.h>
#include <Storages/MergeTree/MergeTreeData.h>
#include <Storages/MergeTree/MergeTreeWhereOptimizer.h>

namespace DB
{

PatternPtr ExplainAnalyze::getPattern() const
{
    return Patterns::explainAnalyze().matchingStep<ExplainAnalyzeStep>([](const auto & step) { return !step.hasPlan(); }).result();
}

TransformResult ExplainAnalyze::transformImpl(PlanNodePtr node, const Captures &, RuleContext & rule_context)
{
    auto original_query_plan_ptr = std::make_shared<QueryPlan>(node->getChildren()[0], rule_context.cte_info, rule_context.context->getPlanNodeIdAllocator());
    CardinalityEstimator::estimate(*original_query_plan_ptr, rule_context.context, true);

    const auto & explain_step = dynamic_cast<const ExplainAnalyzeStep &>(*node->getStep());
    auto new_explain_analyze_step = std::make_shared<ExplainAnalyzeStep>(
        explain_step.getInputStreams()[0],
        explain_step.getKind(),
        rule_context.context,
        original_query_plan_ptr
    );

    return PlanNodeBase::createPlanNode(
        rule_context.context->nextNodeId(),
        new_explain_analyze_step,
        node->getChildren(),
        node->getStatistics()
    );
}

}
