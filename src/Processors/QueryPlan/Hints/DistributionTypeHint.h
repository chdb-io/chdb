#pragma once
#include <Processors/QueryPlan/Hints/IPlanHint.h>
#include <Processors/QueryPlan/Hints/PlanHintFactory.h>
#include <Parsers/IAST.h>
#include <Processors/QueryPlan/PlanNode.h>


namespace DB
{

class DistributionTypeHint : public IPlanHint
{
public:

    DistributionTypeHint(const Strings & options_)
        : options(options_)
    {}

    HintCategory getType() const override { return HintCategory::DISTRIBUTION_TYPE; }
    Strings getOptions() const override { return options; }

    bool checkOptions(Strings & table_name_list) const;
    bool canAttach(PlanNodeBase & node, HintOptions  & hint_options) const override;
protected:
    Strings options;
};

}
