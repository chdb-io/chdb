#pragma once
#include <Optimizer/CardinalityEstimate/PlanNodeStatistics.h>
#include <Processors/QueryPlan/MergeSortingStep.h>
#include <Processors/QueryPlan/MergingSortedStep.h>
#include <Processors/QueryPlan/PartialSortingStep.h>
#include <Processors/QueryPlan/SortingStep.h>

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
