#include <Optimizer/Rule/Patterns.h>
#include <Processors/QueryPlan/SortingStep.h>

namespace DB::Patterns
{

PatternBuilder topN()
{
    auto result = typeOf(IQueryPlanStep::Type::Sorting);
    result.matchingStep<SortingStep>([&](const SortingStep & s) { return s.getLimit() != 0; });
    return result;
}
PatternBuilder & PatternBuilder::capturedAs(const Capture & capture)
{
    return capturedAs(
        capture, [](const PlanNodePtr & node) { return std::any{node}; }, "`this`");
}

PatternBuilder & PatternBuilder::capturedAs(const Capture & capture, const PatternProperty & property)
{
    return capturedAs(capture, property, "unknown");
}

PatternBuilder & PatternBuilder::capturedAs(const Capture & capture, const PatternProperty & property, const std::string & name)
{
    current = std::make_unique<CapturePattern>(name, property, capture, std::move(current));
    return *this;
}

PatternBuilder & PatternBuilder::matching(const PatternPredicate & predicate)
{
    return matching(predicate, "unknown");
}

PatternBuilder & PatternBuilder::matching(const PatternPredicate & predicate, const std::string & name)
{
    current = std::make_unique<FilterPattern>(name, predicate, std::move(current));
    return *this;
}

PatternBuilder & PatternBuilder::matchingCapture(const std::function<bool(const Captures &)> & capture_predicate)
{
    return matchingCapture(capture_predicate, "unknown");
}

PatternBuilder & PatternBuilder::matchingCapture(const std::function<bool(const Captures &)> & capture_predicate, const std::string & name)
{
    return matching(std::bind(capture_predicate, std::placeholders::_2), name); // NOLINT(modernize-avoid-bind)
}

PatternBuilder & PatternBuilder::withEmpty()
{
    current = std::make_unique<WithPattern>(PatternQuantifier::EMPTY, nullptr, std::move(current));
    return *this;
}

PatternBuilder & PatternBuilder::withSingle(PatternPtr sub_pattern)
{
    current = std::make_unique<WithPattern>(PatternQuantifier::SINGLE, std::move(sub_pattern), std::move(current));
    return *this;
}

PatternBuilder & PatternBuilder::withAny(PatternPtr sub_pattern)
{
    current = std::make_unique<WithPattern>(PatternQuantifier::ANY, std::move(sub_pattern), std::move(current));
    return *this;
}

PatternBuilder & PatternBuilder::withAll(PatternPtr sub_pattern)
{
    current = std::make_unique<WithPattern>(PatternQuantifier::ALL, std::move(sub_pattern), std::move(current));
    return *this;
}

PatternBuilder & PatternBuilder::with(PatternPtrs sub_patterns)
{
    current = std::make_unique<WithPattern>(PatternQuantifier::ALL, std::move(sub_patterns), std::move(current));
    return *this;
}

}
