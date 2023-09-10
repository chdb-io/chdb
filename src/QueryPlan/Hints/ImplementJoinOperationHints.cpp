#include <QueryPlan/Hints/ImplementJoinOperationHints.h>
#include <QueryPlan/Hints/BroadcastJoin.h>
#include <QueryPlan/Hints/RepartitionJoin.h>
#include <Optimizer/CostModel/CostCalculator.h>
#include <Optimizer/Rule/Implementation/SetJoinDistribution.h>
#include <Optimizer/CardinalityEstimate/CardinalityEstimator.h>
#include <Common/logger_useful.h>


namespace DB
{

void ImplementJoinOperationHints::rewrite(QueryPlan & plan, ContextMutablePtr context) const
{
    JoinOperationHintsVisitor visitor{context, plan.getCTEInfo(), plan.getPlanNode()};
    Void v;
    VisitorUtil::accept(*plan.getPlanNode(), visitor, v);
}

void JoinOperationHintsVisitor::setStepOptions(JoinStepPtr & step, DistributionType distribution_type, bool isOrdered)
{
    step->setDistributionType(distribution_type);
    step->setOrdered(isOrdered);
}

void JoinOperationHintsVisitor::visitJoinNode(JoinNode & node, Void & v)
{
    visitPlanNode(node, v);

    TableScanHintVisitor visitor;
    TableScanContext left_scan_context;
    TableScanContext right_scan_context;

    VisitorUtil::accept(node.getChildren()[0], visitor, left_scan_context);
    VisitorUtil::accept(node.getChildren()[1], visitor, right_scan_context);

    if (left_scan_context.hint_list.empty() && right_scan_context.hint_list.empty())
        return;

    PlanHintPtr left_hint;
    // If there are multiple JOIN_OPERATION hints, only the first one needs to be considered
    for (auto & hint : left_scan_context.hint_list)
    {
        if (hint->getType() == HintCategory::DISTRIBUTION_TYPE)
        {
            left_hint = hint;
            break;
        }
    }

    PlanHintPtr right_hint;
    // If there are multiple JOIN_OPERATION hints, only the first one needs to be considered
    for (auto & hint : right_scan_context.hint_list)
    {
        if (hint->getType() == HintCategory::DISTRIBUTION_TYPE)
        {
            right_hint = hint;
            break;
        }
    }

    if (left_hint == nullptr && right_hint == nullptr)
        return ;

    const auto & step = *node.getStep();
    if (step.mustRepartition())
        return;

    if (step.mustReplicate())
        return;

    auto left_broadcast_hint = std::dynamic_pointer_cast<BroadcastJoin>(left_hint);
    auto right_broadcast_hint = std::dynamic_pointer_cast<BroadcastJoin>(right_hint);
    auto left_repartition_hint = std::dynamic_pointer_cast<RepartitionJoin>(left_hint);
    auto right_repartition_hint = std::dynamic_pointer_cast<RepartitionJoin>(right_hint);

    // If the nodes on the left and right have different types of hints return；
    if ((left_broadcast_hint && right_broadcast_hint)
        || (left_repartition_hint && right_broadcast_hint)
        || (right_broadcast_hint && right_repartition_hint))
        return;

    if (left_broadcast_hint && supportSwap(step))
    {
        //Change the Join order
        DataStreams streams = {step.getInputStreams()[1], step.getInputStreams()[0]};
        auto new_join_step = std::make_shared<JoinStep>(
            streams,
            step.getOutputStream(),
            step.getKind(),
            step.getStrictness(),
            step.getMaxStreams(),
            step.getKeepLeftReadInOrder(),
            step.getRightKeys(),
            step.getLeftKeys(),
            step.getFilter(),
            step.isHasUsing(),
            step.getRequireRightKeys(),
            step.getAsofInequality(),
            DistributionType::BROADCAST,
            step.getJoinAlgorithm(),
            step.isMagic(),
            true,
            step.getHints());

        if (step.getKind() == JoinKind::Left)
            new_join_step->setKind(JoinKind::Right);
        else if (step.getKind() == JoinKind::Right)
            new_join_step->setKind(JoinKind::Left);

        setStepOptions(new_join_step, DistributionType::BROADCAST, true);
        node.setStep(new_join_step);
        node.replaceChildren(PlanNodes{node.getChildren()[1], node.getChildren()[0]});
        LOG_WARNING(&Poco::Logger::get("ImplementJoinOperationHints"), "BROADCAST_JOIN({})", left_broadcast_hint->getOptions().back());
    }
    else if (right_broadcast_hint)
    {
        auto broadcast_step = std::dynamic_pointer_cast<JoinStep>(step.copy(context));
        setStepOptions(broadcast_step, DistributionType::BROADCAST, true);
        node.setStep(broadcast_step);
        LOG_WARNING(&Poco::Logger::get("ImplementJoinOperationHints"), "BROADCAST_JOIN({})", right_broadcast_hint->getOptions().back());
    }
    else if (left_repartition_hint || right_repartition_hint)
    {
        auto left_stats = CardinalityEstimator::estimate(*node.getChildren()[0], cte_info, context);
        auto right_stats = CardinalityEstimator::estimate(*node.getChildren()[1], cte_info, context);

        if((left_stats.has_value() && right_stats.has_value())
            && (right_stats.value()->getRowCount() > left_stats.value()->getRowCount())
            && supportSwap(step))
        {
            //Change the Join order
            DataStreams streams = {step.getInputStreams()[1], step.getInputStreams()[0]};
            auto new_join_step = std::make_shared<JoinStep>(
                streams,
                step.getOutputStream(),
                step.getKind(),
                step.getStrictness(),
                step.getMaxStreams(),
                step.getKeepLeftReadInOrder(),
                step.getRightKeys(),
                step.getLeftKeys(),
                step.getFilter(),
                step.isHasUsing(),
                step.getRequireRightKeys(),
                step.getAsofInequality(),
                DistributionType::BROADCAST,
                step.getJoinAlgorithm(),
                step.isMagic(),
                true,
                step.getHints());

            if (step.getKind() == JoinKind::Left)
                new_join_step->setKind(JoinKind::Right);
            else if (step.getKind() == JoinKind::Right)
                new_join_step->setKind(JoinKind::Left);

            setStepOptions(new_join_step, DistributionType::REPARTITION, true);
            node.setStep(new_join_step);
            node.replaceChildren(PlanNodes{node.getChildren()[1], node.getChildren()[0]});
        }
        else
        {
            auto repartition_step = std::dynamic_pointer_cast<JoinStep>(step.copy(context));
            setStepOptions(repartition_step, DistributionType::REPARTITION, true);
            node.setStep(repartition_step);
        }

        if (left_repartition_hint)
            LOG_WARNING(&Poco::Logger::get("ImplementJoinOperationHints"), "REPARTITION_JOIN({})", left_repartition_hint->getOptions().back());
        else
            LOG_WARNING(&Poco::Logger::get("ImplementJoinOperationHints"), "REPARTITION_JOIN({})", right_repartition_hint->getOptions().back());
    }
}

void JoinOperationHintsVisitor::visitPlanNode(PlanNodeBase & node, Void & v)
{
    if (!node.getChildren().empty())
        for (auto & child : node.getChildren())
            VisitorUtil::accept(*child, *this, v);
}

void JoinOperationHintsVisitor::visitCTERefNode(CTERefNode & node, Void & v)
{
    CTEId cte_id = node.getStep()->getId();
    post_order_cte_helper.accept(cte_id, *this, v);
}


void TableScanHintVisitor::visitTableScanNode(TableScanNode & node, TableScanContext & context)
{
    context.hint_list = node.getStep()->getHints();
}

void TableScanHintVisitor::visitProjectionNode(ProjectionNode & node, TableScanContext & context)
{
    VisitorUtil::accept(*node.getChildren()[0], *this, context);
}
void TableScanHintVisitor::visitFilterNode(FilterNode & node, TableScanContext & context)
{
    VisitorUtil::accept(*node.getChildren()[0], *this, context);
}

void TableScanHintVisitor::visitPlanNode(PlanNodeBase & , TableScanContext & context)
{
    context.hint_list.clear();
}

}
