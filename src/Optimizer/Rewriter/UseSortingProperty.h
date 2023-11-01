#pragma once

#include <Core/SortDescription.h>
#include <Interpreters/Context.h>
#include <Optimizer/Equivalences.h>
#include <Optimizer/Rewriter/Rewriter.h>
#include <Processors/QueryPlan/CTEInfo.h>
#include <Processors/QueryPlan/SimplePlanRewriter.h>
#include <Processors/QueryPlan/SimplePlanVisitor.h>

namespace DB
{
class SortingOrderedSource : public Rewriter
{
public:
    void rewrite(QueryPlan & plan, ContextMutablePtr context) const override;
    String name() const override { return "SortingOrderedSource"; }

private:
    class Rewriter;
};

struct PlanAndProp
{
    PlanNodePtr plan;
    Property property;
};

class SortingOrderedSource::Rewriter : public PlanNodeVisitor<PlanAndProp, Void>
{
public:
    Rewriter(ContextMutablePtr context_, CTEInfo & cte_info_) : context(context_), cte_helper(cte_info_) { }
    PlanAndProp visitPlanNode(PlanNodeBase &, Void &) override;
    PlanAndProp visitSortingNode(SortingNode &, Void &) override;
    PlanAndProp visitAggregatingNode(AggregatingNode &, Void &) override;
    PlanAndProp visitWindowNode(WindowNode &, Void &) override;
    PlanAndProp visitCTERefNode(CTERefNode & node, Void &) override;

private:
    ContextMutablePtr context;
    SimpleCTEVisitHelper<PlanAndProp> cte_helper;
};


struct SortInfo
{
    SortDescription sort_desc;
    size_t limit;
};

class PushSortingInfoRewriter : public SimplePlanRewriter<SortInfo>
{
public:
    PushSortingInfoRewriter(ContextMutablePtr context_, CTEInfo & cte_info_, PlanNodePtr & root)
        : SimplePlanRewriter(context_, cte_info_), post_order_cte_helper(cte_info_, root)
    {
    }
    PlanNodePtr visitSortingNode(SortingNode &, SortInfo &) override;
    PlanNodePtr visitAggregatingNode(AggregatingNode &, SortInfo &) override;
    PlanNodePtr visitWindowNode(WindowNode &, SortInfo &) override;
    PlanNodePtr visitTableScanNode(TableScanNode &, SortInfo &) override;

private:
    CTEPostorderVisitHelper post_order_cte_helper;
};

}
