#include <Optimizer/JoinOrderUtils.h>

#include <Optimizer/Property/Property.h>
#include <Optimizer/Property/PropertyDeriver.h>
#include <Optimizer/Property/PropertyMatcher.h>
#include <Optimizer/SymbolsExtractor.h>
#include <Processors/QueryPlan/CTERefStep.h>
#include <Processors/QueryPlan/PlanNode.h>
#include <Processors/QueryPlan/SymbolMapper.h>
#include <Processors/QueryPlan/TableScanStep.h>

namespace DB
{
String JoinOrderUtils::getJoinOrder(QueryPlan & plan)
{
    JoinOrderExtractor rewriter{plan.getCTEInfo()};
    Void require;
    return VisitorUtil::accept(plan.getPlanNode(), rewriter, require);
}

String JoinOrderExtractor::visitPlanNode(PlanNodeBase & node, Void &)
{
    Names children;
    Void require;
    PropertySet input_properties;
    for (const auto & child : node.getChildren())
    {
        auto result = VisitorUtil::accept(child, *this, require);
        children.emplace_back(result);
    }
    if (children.size() == 1)
        return children[0];

    return fmt::format("[{}]", boost::algorithm::join(children, ", "));
}

String JoinOrderExtractor::visitJoinNode(JoinNode & node, Void & v)
{
    auto left = VisitorUtil::accept(node.getChildren()[0], *this, v);
    auto right = VisitorUtil::accept(node.getChildren()[1], *this, v);

    return fmt::format("({}, {})", left, right);
}


String JoinOrderExtractor::visitTableScanNode(TableScanNode & node, Void &)
{
    return node.getStep()->getTable();
}

String JoinOrderExtractor::visitCTERefNode(CTERefNode & node, Void & v)
{
    const auto * step = node.getStep().get();

    auto cte_order = cte_helper.accept(step->getId(), *this, v);

    return cte_order;
}


}
