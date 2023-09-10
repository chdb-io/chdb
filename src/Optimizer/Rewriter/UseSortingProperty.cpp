#include <Optimizer/Rewriter/UseSortingProperty.h>

#include <Optimizer/Property/Property.h>
#include <Optimizer/Property/PropertyDeriver.h>
#include <Optimizer/Property/PropertyMatcher.h>
#include <QueryPlan/CTERefStep.h>
#include <QueryPlan/PlanNode.h>
#include <QueryPlan/SymbolMapper.h>

namespace DB
{
void SortingOrderedSource::rewrite(QueryPlan & plan, ContextMutablePtr context) const
{
    SortingOrderedSource::Rewriter rewriter{context, plan.getCTEInfo()};
    Void require;
    auto result = VisitorUtil::accept(plan.getPlanNode(), rewriter, require);

    PushSortingInfoRewriter push_rewriter{context, plan.getCTEInfo(), plan.getPlanNode()};
    SortInfo sort_info;
    auto plan_node = VisitorUtil::accept(result.plan, push_rewriter, sort_info);
    plan.update(plan_node);
}

PlanAndProp SortingOrderedSource::Rewriter::visitPlanNode(PlanNodeBase & node, Void &)
{
    PlanNodes children;
    DataStreams inputs;
    Void require;
    PropertySet input_properties;
    for (const auto & child : node.getChildren())
    {
        auto result = VisitorUtil::accept(child, *this, require);
        children.emplace_back(result.plan);
        inputs.push_back(result.plan->getStep()->getOutputStream());
        input_properties.emplace_back(result.property);
    }

    node.getStep()->setInputStreams(inputs);

    node.replaceChildren(children);
    Property prop = PropertyDeriver::deriveProperty(node.getStep(), input_properties, context);
    return {node.shared_from_this(), prop};
}

PlanAndProp SortingOrderedSource::Rewriter::visitSortingNode(SortingNode & node, Void & v)
{
    auto result = VisitorUtil::accept(node.getChildren()[0], *this, v);

    auto step = node.getStep();
    auto prefix_sorting = PropertyMatcher::matchSorting(*context, step->getSortDescription(), result.property.getSorting());
    step->setPrefixDescription(prefix_sorting.toSortDesc());
    Property prop = PropertyDeriver::deriveProperty(node.getStep(), {result.property}, context);
    return {node.shared_from_this(), prop};
}

PlanAndProp SortingOrderedSource::Rewriter::visitAggregatingNode(AggregatingNode & node, Void & v)
{
    auto result = VisitorUtil::accept(node.getChildren()[0], *this, v);

    if (context->getSettingsRef().optimize_aggregation_in_order)
    {
        auto step = node.getStep();

        SortDescription order_descr;
        order_descr.reserve(step->getKeys().size());
        for (const auto & name : step->getKeys())
        {
            order_descr.emplace_back(name, 1, 1);
        }


        auto prefix_sorting = PropertyMatcher::matchSorting(*context, order_descr, result.property.getSorting());
        step->setGroupBySortDescription(prefix_sorting.toSortDesc());
    }
    Property prop = PropertyDeriver::deriveProperty(node.getStep(), {result.property}, context);
    return {node.shared_from_this(), prop};
}

PlanAndProp SortingOrderedSource::Rewriter::visitWindowNode(WindowNode & node, Void & v)
{
    auto result = VisitorUtil::accept(node.getChildren()[0], *this, v);

#if 0 
    if (context->getSettingsRef().optimize_read_in_window_order)
    {
        auto step = node.getStep();

        SortDescription order_descr;
        const auto & partition_by = step->getWindowDescription().partition_by;
        order_descr.insert(order_descr.end(), partition_by.begin(), partition_by.end());

        const auto & order_by = step->getWindowDescription().order_by;
        order_descr.insert(order_descr.end(), order_by.begin(), order_by.end());

        auto prefix_sorting = PropertyMatcher::matchSorting(*context, order_descr, result.property.getSorting());

        step->setPrefixDescription(prefix_sorting.toSortDesc());
    }
#endif
    Property prop = PropertyDeriver::deriveProperty(node.getStep(), {result.property}, context);
    return {node.shared_from_this(), prop};
}

PlanAndProp SortingOrderedSource::Rewriter::visitCTERefNode(CTERefNode & node, Void & v)
{
    const auto * step = node.getStep().get();

    auto cte_plan = cte_helper.accept(step->getId(), *this, v);
    return {node.shared_from_this(), Property{}};
}

PlanNodePtr PushSortingInfoRewriter::visitSortingNode(SortingNode & node, SortInfo &)
{
    auto prefix_desc = node.getStep()->getPrefixDescription();
    SortInfo s{prefix_desc, node.getStep()->getLimit()};
    return visitPlanNode(node, s);
}

PlanNodePtr PushSortingInfoRewriter::visitAggregatingNode(AggregatingNode & node, SortInfo &)
{

    auto prefix_desc = node.getStep()->getGroupBySortDescription();
    SortInfo s{prefix_desc, 0};
    return visitPlanNode(node, s);
}

PlanNodePtr PushSortingInfoRewriter::visitWindowNode(WindowNode & node, SortInfo &)
{

    auto prefix_desc = node.getStep()->getPrefixDescription();
    SortInfo s{prefix_desc, 0};
    return visitPlanNode(node, s);
}

PlanNodePtr PushSortingInfoRewriter::visitTableScanNode(TableScanNode & node, SortInfo & s)
{
    node.getStep()->setReadOrder(s.sort_desc);
    return node.shared_from_this();
}

}
