/*
 * Copyright (2022) Bytedance Ltd. and/or its affiliates
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <QueryPlan/PlanVisitor.h>
#include <Optimizer/CostModel/PlanNodeCost.h>
// #include <Interpreters/DistributedStages/PlanSegment.h>
#include <Interpreters/ProcessorProfile.h>

#include <Poco/JSON/Object.h>

namespace DB
{
using PlanCostMap = std::unordered_map<PlanNodeId, double>;

class PlanPrinter
{
public:
    PlanPrinter() = delete;

    static String textLogicalPlan(
        QueryPlan & plan,
        ContextMutablePtr context,
        bool print_stats,
        bool verbose,
        PlanCostMap costs = {},
        const StepAggregatedOperatorProfiles & profiles = {});
    static String jsonLogicalPlan(QueryPlan & plan, bool print_stats, bool verbose, const PlanNodeCost & plan_cost = {});
    static String textDistributedPlan(
        PlanSegmentDescriptions & segments_desc,
        bool print_stats,
        bool verbose,
        const std::unordered_map<PlanNodeId, double> & costs = {},
        const StepAggregatedOperatorProfiles & profiles = {},
        const QueryPlan & query_plan = {});
    static void getPlanNodes(const PlanNodePtr & parent, std::unordered_map<PlanNodeId, PlanNodePtr> & id_to_node);
    static std::unordered_map<PlanNodeId, PlanNodePtr>  getPlanNodeMap(const QueryPlan & query_plan);
    static void getRemoteSegmentId(const QueryPlan::Node * node, std::unordered_map<PlanNodeId, size_t> & exchange_to_segment);

    class TextPrinter;
private:
    class JsonPrinter;
};

class TextPrinterIntent
{
public:
    static constexpr auto VERTICAL_LINE = "│  ";
    static constexpr auto INTERMEDIATE_PREFIX = "├─ ";
    static constexpr auto LAST_PREFIX = "└─ ";
    static constexpr auto EMPTY_PREFIX = "   ";

    TextPrinterIntent() = default;
    explicit TextPrinterIntent(size_t prefix, bool has_children_)
        : current_lines_prefix(std::string(prefix, ' '))
        , next_lines_prefix(std::string(prefix, ' '))
        , hasChildren(has_children_)
    {
    }

    TextPrinterIntent forChild(bool last, bool has_children_) const;
    String print() const { return current_lines_prefix; }
    String detailIntent() const;

private:
    TextPrinterIntent(String current_lines_prefix_, String next_lines_prefix_, bool hasChildren);

    String current_lines_prefix;
    String next_lines_prefix;
    bool hasChildren{true};
};

class PlanPrinter::TextPrinter
{
public:
    TextPrinter(bool print_stats_, bool verbose_, const std::unordered_map<PlanNodeId, double> & costs_, bool is_distributed_ = false, const std::unordered_map<PlanNodeId, size_t> & exchange_to_segment_ = {})
        : print_stats(print_stats_), verbose(verbose_), costs(costs_), is_distributed(is_distributed_), exchange_to_segment(exchange_to_segment_)
    {
    }
    static String printOutputColumns(PlanNodeBase & plan_node, const TextPrinterIntent & intent = {});
    String printLogicalPlan(PlanNodeBase & plan, const TextPrinterIntent & intent = {}, const StepAggregatedOperatorProfiles & profiles = {});

    static String prettyNum(size_t num);
    static String prettyBytes(size_t bytes);
private:
    static String prettySeconds(size_t seconds);
    static String printPrefix(PlanNodeBase & plan);
    String printSuffix(PlanNodeBase & plan);
    static String printQError(PlanNodeBase & plan, const TextPrinterIntent & intent, const StepAggregatedOperatorProfiles & profiles);
    String printDetail(QueryPlanStepPtr plan, const TextPrinterIntent & intent) const;
    String printStatistics(const PlanNodeBase & plan, const TextPrinterIntent & intent = {}) const;
    static String printOperatorProfiles(PlanNodeBase & plan, const TextPrinterIntent & intent = {}, const StepAggregatedOperatorProfiles & profiles = {}) ;

    const bool print_stats;
    const bool verbose;
    const std::unordered_map<PlanNodeId, double> & costs;
    bool is_distributed;
    const std::unordered_map<PlanNodeId, size_t> & exchange_to_segment;
};

class PlanPrinter::JsonPrinter
{
public:
    explicit JsonPrinter(bool print_stats_) : print_stats(print_stats_) { }
    Poco::JSON::Object::Ptr printLogicalPlan(PlanNodeBase & plan);

private:
    static void detail(Poco::JSON::Object::Ptr & json, QueryPlanStepPtr plan);

    const bool print_stats;
};

}
