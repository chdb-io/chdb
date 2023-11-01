#pragma once

#include <Core/SortDescription.h>
#include <Interpreters/Context.h>
#include <Processors/QueryPlan/CTEInfo.h>
#include <Processors/QueryPlan/SimplePlanRewriter.h>
#include <Processors/QueryPlan/SimplePlanVisitor.h>

namespace DB
{
class JoinOrderUtils
{
public:
    static String getJoinOrder(QueryPlan & plan);
};

struct PlanAndProp
{
    PlanNodePtr plan;
    Property property;
};

class JoinOrderExtractor : public PlanNodeVisitor<String, Void>
{
public:
    JoinOrderExtractor(CTEInfo & cte_info_) : cte_helper(cte_info_) { }
    String visitPlanNode(PlanNodeBase &, Void &) override;
    String visitJoinNode(JoinNode &, Void &) override;
    String visitCTERefNode(CTERefNode & node, Void &) override;
    String visitTableScanNode(TableScanNode & node, Void &) override;

private:
    SimpleCTEVisitHelper<String> cte_helper;
};



}
