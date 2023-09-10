#pragma once
#include <Optimizer/Rule/Rule.h>

namespace DB
{

/**
 * Implements distinct aggregations with different inputs by transforming plans of the following shape:
 * <pre>
 * - Aggregation
 *        GROUP BY (k)
 *        F1(DISTINCT a0, a1, ...)
 *        F2(DISTINCT b0, b1, ...)
 *        F3(c0, c1, ...)
 *     - X
 * </pre>
 * into
 * <pre>
 * - Aggregation
 *        GROUP BY (k)
 *        F1(a0, a1, ...) mask ($0)
 *        F2(b0, b1, ...) mask ($1)
 *        F3(c0, c1, ...)
 *     - MarkDistinct (k, a0, a1, ...) -> $0
 *          - MarkDistinct (k, b0, b1, ...) -> $1
 *              - X
 * </pre>
 */
class MultipleDistinctAggregationToMarkDistinct : public Rule
{
public:
    RuleType getType() const override { return RuleType::MULTIPLE_DISTINCT_AGG_TO_MARKDISTINCT; }
    String getName() const override { return "MULTIPLE_DISTINCT_AGG_TO_MARKDISTINCT"; }
    PatternPtr getPattern() const override;
    bool isEnabled(ContextPtr context) override { return context->getSettingsRef().enable_mark_distinct_optimzation; }

protected:
    TransformResult transformImpl(PlanNodePtr node, const Captures & captures, RuleContext & context) override;

private:
    static const std::set<String> distinct_func;
    static const std::unordered_map<String, String> distinct_func_normal_func;
    static bool hasNoDistinctWithFilterOrMask(const AggregatingStep & step);
    static bool hasMultipleDistincts(const AggregatingStep & step);
    static bool hasMixedDistinctAndNonDistincts(const AggregatingStep & step);
};

}
