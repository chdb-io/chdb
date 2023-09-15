#include <QueryPlan/Hints/DistributionTypeHint.h>
#include <QueryPlan/Hints/PlanHintFactory.h>
#include <Parsers/IAST.h>
#include <QueryPlan/PlanNode.h>


namespace DB
{

bool DistributionTypeHint::checkOptions(Strings & table_name_list) const
{
    if (table_name_list.size() != 1)
        return false;
    if (table_name_list[0] == options[0])
        return true;

    return false;
}

bool DistributionTypeHint::canAttach(PlanNodeBase & node, HintOptions & hint_options) const
{
    auto & table_name_list = hint_options.table_name_list;
    if (node.getStep()->getType() != IQueryPlanStep::Type::TableScan)
        return false;

    if (!checkOptions(table_name_list))
        return false;

    return true;
}

}
