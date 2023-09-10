#pragma once

#include <Optimizer/Rule/Rule.h>

namespace DB
{

/**
 *          Apply                 join
 *         /     \               /    \
 *      join    subquery  ==>  A     Apply
 *     /   \                        /   \
 *    A     B                      B     subquery
 *
 */

class PushDownApplyThroughJoin: public Rule
{
public:
    RuleType getType() const override { return RuleType::PUSH_DOWN_APPLY_THROUGH_JOIN; }
    String getName() const override { return "PUSH_DOWN_APPLY_THROUGH_JOIN"; }

    PatternPtr getPattern() const override;

    TransformResult transformImpl(PlanNodePtr node, const Captures & captures, RuleContext & context) override;
};

}

