#include <Optimizer/Rule/Rewrite/PushDownApplyRules.h>

#include <Core/ColumnsWithTypeAndName.h>
#include <Core/Names.h>
#include <Core/SortDescription.h>
#include <Optimizer/PlanNodeCardinality.h>
#include <Optimizer/Rule/Pattern.h>
#include <Optimizer/Rule/Patterns.h>
#include <Processors/QueryPlan/IQueryPlanStep.h>
#include <QueryPlan/ApplyStep.h>
#include <QueryPlan/JoinStep.h>
#include <QueryPlan/LimitStep.h>
#include <QueryPlan/PlanNode.h>
#include <QueryPlan/QueryPlan.h>
#include <QueryPlan/WindowStep.h>

namespace DB
{
PatternPtr PushDownApplyThroughJoin::getPattern() const
{
    return Patterns::apply()
        .matchingStep<ApplyStep>([](const ApplyStep & step) { return !step.getOuterColumns().empty() && step.getCorrelation().empty(); })
        .with(Patterns::join(), Patterns::any())
        .result();
}

TransformResult PushDownApplyThroughJoin::transformImpl(PlanNodePtr node, const Captures &, RuleContext & context)
{
    auto * apply = dynamic_cast<ApplyStep *>((node->getStep().get()));
    const auto & outer_columns = apply->getOuterColumns();
    auto join = node->getChildren()[0];
    size_t index = 0;
    for (const auto & child : join->getChildren())
    {
        auto out_names = child->getOutputNames();
        if (std::includes(out_names.begin(), out_names.end(), outer_columns.begin(), outer_columns.end()))
        {
            DataStreams input{child->getStep()->getOutputStream(), node->getChildren()[1]->getStep()->getOutputStream()};

            auto apply_step = std::make_shared<ApplyStep>(
                input,
                apply->getCorrelation(),
                apply->getApplyType(),
                apply->getSubqueryType(),
                apply->getAssignment(),
                apply->getOuterColumns());
            PlanNodes children{child, node->getChildren()[1]};
            auto apply_node
                = ApplyNode::createPlanNode(context.context->nextNodeId(), std::move(apply_step), children, node->getStatistics());
            PlanNodes join_children = join->getChildren();
            join_children[index] = apply_node;

            DataStreams inputs;
            ColumnsWithTypeAndName output;
            for (const auto & item : join_children)
            {
                auto stream = item->getCurrentDataStream();
                inputs.emplace_back(stream);
                output.insert(output.end(), stream.header.begin(), stream.header.end());
            }

            auto step = dynamic_cast<JoinStep *>(join->getStep().get());
            auto new_step = std::make_shared<JoinStep>(
                inputs,
                DataStream{Block{output}},
                step->getKind(),
                step->getStrictness(),
                step->getMaxStreams(),
                step->getKeepLeftReadInOrder(),
                step->getLeftKeys(),
                step->getRightKeys(),
                step->getFilter(),
                step->isHasUsing(),
                step->getRequireRightKeys(),
                step->getAsofInequality(),
                step->getDistributionType(),
                step->getJoinAlgorithm(),
                step->isMagic(),
                step->isOrdered(),
                step->getHints());

            return PlanNodeBase::createPlanNode(join->getId(), new_step, join_children);
        }
        ++index;
    }
    return {};
}

}
