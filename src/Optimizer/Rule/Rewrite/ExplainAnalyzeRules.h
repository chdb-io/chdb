#pragma once

#include <Optimizer/Rule/Rule.h>

namespace DB
{

class ExplainAnalyze : public Rule
{
public:
    RuleType getType() const override { return RuleType::EXPLAIN_ANALYZE; }
    String getName() const override { return "EXPLAIN_ANALYZE"; }

    PatternPtr getPattern() const override;

    TransformResult transformImpl(PlanNodePtr node, const Captures & captures, RuleContext & context) override;
};

}
