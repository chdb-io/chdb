#include <Optimizer/CardinalityEstimate/LimitEstimator.h>
#include <Optimizer/CardinalityEstimate/SortingEstimator.h>

namespace DB
{
PlanNodeStatisticsPtr SortingEstimator::estimate(PlanNodeStatisticsPtr & child_stats, const SortingStep & step)
{
    if (step.getLimit() > 0)
    {
        size_t limit = step.getLimit();
        return LimitEstimator::getLimitStatistics(child_stats, limit);
    }
    return child_stats;
}

PlanNodeStatisticsPtr SortingEstimator::estimate(PlanNodeStatisticsPtr & child_stats, const PartialSortingStep & step)
{
    if (step.getLimit() > 0)
    {
        size_t limit = step.getLimit();
        return LimitEstimator::getLimitStatistics(child_stats, limit);
    }
    return child_stats;
}

PlanNodeStatisticsPtr SortingEstimator::estimate(PlanNodeStatisticsPtr & child_stats, const MergeSortingStep & step)
{
    if (step.getLimit() > 0)
    {
        size_t limit = step.getLimit();
        return LimitEstimator::getLimitStatistics(child_stats, limit);
    }
    return child_stats;
}

PlanNodeStatisticsPtr SortingEstimator::estimate(PlanNodeStatisticsPtr & child_stats, const MergingSortedStep & step)
{
    if (step.getLimit() > 0)
    {
        size_t limit = step.getLimit();
        return LimitEstimator::getLimitStatistics(child_stats, limit);
    }
    return child_stats;
}

}
