#pragma once
#include <Optimizer/Rule/Rule.h>

namespace DB
{

class CreateTopNFilteringForAggregating: public Rule
{
public:
    RuleType getType() const override { return RuleType::CREATE_TOPN_FILTERING_FOR_AGGREGATING; }
    String getName() const override { return "CREATE_TOPN_FILTERING_FOR_AGGREGATING"; }

    PatternPtr getPattern() const override;

    TransformResult transformImpl(PlanNodePtr node, const Captures & captures, RuleContext & context) override;
};

class PushTopNThroughProjection: public Rule
{
public:
    RuleType getType() const override { return RuleType::PUSH_TOPN_THROUGH_PROJECTION; }
    String getName() const override { return "PUSH_TOPN_THROUGH_PROJECTION"; }

    PatternPtr getPattern() const override;

    TransformResult transformImpl(PlanNodePtr node, const Captures & captures, RuleContext & context) override;
};

class PushTopNFilteringThroughProjection: public Rule
{
public:
    RuleType getType() const override { return RuleType::PUSH_TOPN_FILTERING_THROUGH_PROJECTION; }
    String getName() const override { return "PUSH_TOPN_FILTERING_THROUGH_PROJECTION"; }

    PatternPtr getPattern() const override;

    TransformResult transformImpl(PlanNodePtr node, const Captures & captures, RuleContext & context) override;
};

}
