#pragma once
#include <Processors/QueryPlan/Hints/IPlanHint.h>
#include <Processors/QueryPlan/Hints/PlanHintFactory.h>
#include <Parsers/IAST.h>
#include <Processors/QueryPlan/PlanNode.h>
#include <Processors/QueryPlan/Hints/DistributionTypeHint.h>


namespace DB
{

class RepartitionJoin : public DistributionTypeHint
{
public:
    static constexpr auto name = "REPARTITION_JOIN";

    static PlanHintPtr create(const SqlHint & sql_hint, const ContextMutablePtr & context)
    {
        if (sql_hint.getOptions().size() != 1)
            return {};
        return std::make_shared<RepartitionJoin>(sql_hint, context);
    }

    RepartitionJoin(const SqlHint & sql_hint, const ContextMutablePtr & )
        : DistributionTypeHint(sql_hint.getOptions())
    {}

    String getName() const override { return name; }

};


}
