#include <QueryPlan/Hints/HintsPropagator.h>

#include <QueryPlan/IQueryPlanStep.h>
#include <QueryPlan/JoinStep.h>
#include <QueryPlan/PlanNode.h>
#include <QueryPlan/Hints/Leading.h>
#include <QueryPlan/Hints/IPlanHint.h>

namespace DB
{

void HintsPropagator::rewrite(QueryPlan & plan, ContextMutablePtr context) const
{
    HintsVisitor visitor{context, plan.getCTEInfo(), plan.getPlanNode()};
    HintsVisitorContext hints_context;
    VisitorUtil::accept(plan.getPlanNode(), visitor, hints_context);
}

void HintsVisitor::attachPlanHints(PlanNodeBase & node, HintOptions & hint_options)
{
    PlanHints plan_hints;
    // This order is for better handling of hints override
    for (auto it = hints_list.rbegin(); it != hints_list.rend(); ++it)
    {
        for (auto & hint : *it)
        {
            if (hint->canAttach(node, hint_options))
                plan_hints.emplace_back(hint);
        }
    }
    if (!plan_hints.empty())
    {
        auto new_step = node.getStep()->copy(context);
        new_step->setHints(plan_hints);
        node.setStep(new_step);
    }
}

void HintsVisitor::processNodeWithHints(PlanNodeBase & node, HintsVisitorContext & hints_context, HintOptions & hint_options)
{
    auto step_hints = node.getStep()->getHints();
    // check leading hint
    for (auto it = step_hints.begin(); it != step_hints.end(); ++it)
    {
        if (auto leading_hint = std::dynamic_pointer_cast<Leading>(*it))
        {
            if (!leading_hint->isVaildLeading(node))
            {
                step_hints.erase(it);
                break;
            }
        }
    }

    bool has_hints = !step_hints.empty();
    if (has_hints)
        hints_list.emplace_back(step_hints);

    // process child node
    for (const auto & item : node.getChildren())
        VisitorUtil::accept(*item, *this, hints_context);

    // process current node
    attachPlanHints(node, hint_options);

    if (has_hints)
        hints_list.pop_back();
}

void HintsVisitor::visitProjectionNode(ProjectionNode & node, HintsVisitorContext & hints_context)
{
    HintOptions hint_options;
    processNodeWithHints(node, hints_context, hint_options);
}

void HintsVisitor::visitJoinNode(JoinNode & node, HintsVisitorContext & hints_context)
{
    auto step_hints = node.getStep()->getHints();
    bool has_hints = !step_hints.empty();
    if (has_hints)
        hints_list.emplace_back(step_hints);

    for (const auto & item : node.getChildren())
    {
        HintsVisitorContext child_hints_context;
        VisitorUtil::accept(*item, *this, child_hints_context);
        hints_context.name_list.insert(hints_context.name_list.end(), child_hints_context.name_list.begin(), child_hints_context.name_list.end());
    }

    HintOptions hint_options;
    hint_options.table_name_list = hints_context.name_list;
    attachPlanHints(node, hint_options);
    if (has_hints)
        hints_list.pop_back();
}

void HintsVisitor::visitFilterNode(FilterNode & node, HintsVisitorContext & hints_context)
{
    HintOptions hint_options;
    processNodeWithHints(node, hints_context, hint_options);
}

void HintsVisitor::visitTableScanNode(TableScanNode & node, HintsVisitorContext & hints_context)
{
    hints_context.name_list.clear();

    auto table_scan_step =  node.getStep();
    if (!table_scan_step->getTableAlias().empty())
        hints_context.name_list.emplace_back(table_scan_step->getTableAlias());
    else
        hints_context.name_list.emplace_back(table_scan_step->getTable());

    HintOptions hint_options;
    hint_options.table_name_list = hints_context.name_list;

    processNodeWithHints(node, hints_context, hint_options);
}

void HintsVisitor::visitPlanNode(PlanNodeBase & node, HintsVisitorContext & hints_context)
{
    hints_context.name_list.clear();
    for (const auto & item : node.getChildren())
    {
        VisitorUtil::accept(*item, *this, hints_context);
        hints_context.name_list.clear();
    }
}

void HintsVisitor::visitCTERefNode(CTERefNode & node, HintsVisitorContext &)
{
    CTEId cte_id = node.getStep()->getId();
    auto hints_list_tmp = hints_list;
    hints_list.clear();
    HintsVisitorContext hints_context;
    post_order_cte_helper.accept(cte_id, *this, hints_context);
    hints_list = hints_list_tmp;
}

}
