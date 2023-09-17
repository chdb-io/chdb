#pragma once
#include <Optimizer/Rewriter/Rewriter.h>
#include <Processors/QueryPlan/Assignment.h>
#include <Processors/QueryPlan/PlanNode.h>
#include <Processors/QueryPlan/JoinStep.h>
#include <Optimizer/PredicateUtils.h>
#include <Processors/QueryPlan/PlanVisitor.h>
#include <Processors/QueryPlan/SimplePlanRewriter.h>


namespace DB
{
using JoinStepPtr = std::shared_ptr<JoinStep>;

class ImplementJoinOperationHints : public Rewriter
{
public:
    void rewrite(QueryPlan & plan, ContextMutablePtr context) const override;
    String name() const override { return "ImplementJoinOperationHints"; }
};

class JoinOperationHintsVisitor : public PlanNodeVisitor<void, Void>
{
public:
    explicit JoinOperationHintsVisitor(ContextMutablePtr & context_, CTEInfo & cte_info_, PlanNodePtr & root)
        : context(context_), post_order_cte_helper(cte_info_, root), cte_info(cte_info_)
    {}

    static bool supportSwap(const JoinStep & s) { return s.supportSwap() && PredicateUtils::isTruePredicate(s.getFilter());}

private:

    void visitJoinNode(JoinNode & node, Void &) override;
    void visitPlanNode(PlanNodeBase & node, Void &)  override;
    void visitCTERefNode(CTERefNode & node, Void &) override;

    static void setStepOptions(JoinStepPtr & step, DistributionType distribution_type, bool isOrdered = false);

    ContextMutablePtr & context;
    CTEPostorderVisitHelper post_order_cte_helper;
    CTEInfo & cte_info;
};

struct TableScanContext
{
    PlanHints hint_list = {};
};
class TableScanHintVisitor : public PlanNodeVisitor<void, TableScanContext>
{
private:
    void visitProjectionNode(ProjectionNode & node, TableScanContext & table_names) override;
    void visitFilterNode(FilterNode & node, TableScanContext & table_names) override;
    void visitTableScanNode(TableScanNode & node, TableScanContext & table_names) override;
    void visitPlanNode(PlanNodeBase & node, TableScanContext & table_names)  override;

};

}
