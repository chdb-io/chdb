#pragma once
#include <Processors/QueryPlan/Hints/IPlanHint.h>
#include <Processors/QueryPlan/Hints/PlanHintFactory.h>
#include <Parsers/IAST.h>
#include <Processors/QueryPlan/PlanNode.h>
#include <Processors/QueryPlan/PlanVisitor.h>


namespace DB
{

class Leading : public IPlanHint
{
public:
    static constexpr auto name = "LEADING";

    static PlanHintPtr create(const SqlHint & sql_hint, const ContextMutablePtr & context);

    Leading(const SqlHint & sql_hint, const ContextMutablePtr & )
    {
        Strings options = sql_hint.getOptions();
        if (options.size() > 2 && options.front() == "(" && options.back() == ")")
            transformJoinPairsToRPN(options);
        else
            transformTableListToRPN(options);
    }

    String getName() const override { return name; }
    HintCategory getType() const override { return HintCategory::JOIN_ORDER; }
    Strings getRPNList() const  { return RPN_table_list; }
    Strings getOptions() const override { return table_list;}
    String getJoinOrderString() const;

    bool checkOptions(Strings & table_name_list) const;

    bool isVaildLeading(PlanNodeBase & node);

    bool canAttach(PlanNodeBase & node, HintOptions & hint_options) const override;
private:

    // leading hint table list
    Strings table_list;

    // Reverse Polish notation(RPN) for join order，Comma represent two tables join
    Strings RPN_table_list;

    void transformJoinPairsToRPN(const Strings & options);
    void transformTableListToRPN(const Strings & options);
};

//get the table name list of a node
class TableNamesVisitor : public PlanNodeVisitor<void, Strings>
{
private:
    void visitTableScanNode(TableScanNode & node, Strings & table_names) override;
    void visitPlanNode(PlanNodeBase & node, Strings & table_names)  override;

};


}
