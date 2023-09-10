#pragma once
#include <Optimizer/CardinalityEstimate/PlanNodeStatistics.h>
#include <QueryPlan/MergeSortingStep.h>
#include <QueryPlan/MergingSortedStep.h>
#include <QueryPlan/PartialSortingStep.h>
#include <QueryPlan/SortingStep.h>

namespace DB
{
class SortingEstimator
{
public:
    static PlanNodeStatisticsPtr estimate(PlanNodeStatisticsPtr & child_stats, const SortingStep &);
    static PlanNodeStatisticsPtr estimate(PlanNodeStatisticsPtr & child_stats, const PartialSortingStep &);
    static PlanNodeStatisticsPtr estimate(PlanNodeStatisticsPtr & child_stats, const MergeSortingStep &);
    static PlanNodeStatisticsPtr estimate(PlanNodeStatisticsPtr & child_stats, const MergingSortedStep &);

};

}
