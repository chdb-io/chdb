#pragma once
#include <QueryPlan/Hints/IPlanHint.h>
#include <QueryPlan/Hints/PlanHintFactory.h>
#include <Parsers/IAST.h>
#include <QueryPlan/PlanNode.h>
#include <QueryPlan/Hints/DistributionTypeHint.h>


namespace DB
{

class BroadcastJoin : public DistributionTypeHint
{
public:
    static constexpr auto name = "BROADCAST_JOIN";

    static PlanHintPtr create(const SqlHint & sql_hint, const ContextMutablePtr & context)
    {
        if (sql_hint.getOptions().size() != 1)
            return {};
        return std::make_shared<BroadcastJoin>(sql_hint, context);
    }

    BroadcastJoin(const SqlHint & sql_hint, const ContextMutablePtr & )
        : DistributionTypeHint(sql_hint.getOptions())
    {}

    String getName() const override { return name; }

};

}
