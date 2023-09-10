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

#include <QueryPlan/PlanPrinter.h>

#include <Analyzers/ASTEquals.h>
#include <Optimizer/PlanNodeSearcher.h>
#include <Optimizer/PredicateConst.h>
#include <Optimizer/PredicateUtils.h>
#include <Optimizer/Utils.h>
#include <Optimizer/OptimizerMetrics.h>
#include <AggregateFunctions/AggregateFunctionNull.h>
#include <Poco/JSON/Object.h>
#include <Parsers/formatAST.h>
#include <QueryPlan/QueryPlan.h>
#include <QueryPlan/GraphvizPrinter.h>
#include <Interpreters/convertFieldToType.h>

#include <utility>

namespace DB
{
namespace
{
template <class V>
String join(const V & v, const String & sep, const String & prefix = {}, const String & suffix = {})
{
    std::stringstream out;
    out << prefix;
    if (!v.empty())
    {
        auto it = v.begin();
        out << *it;
        for (++it; it != v.end(); ++it)
            out << sep << *it;
    }
    out << suffix;
    return out.str();
}
}

String PlanPrinter::textLogicalPlan(
    QueryPlan & plan,
    ContextMutablePtr context,
    bool print_stats,
    bool verbose,
    PlanCostMap costs,
    const StepAggregatedOperatorProfiles & profiles)
{
    TextPrinter printer{print_stats, verbose, costs};
    bool has_children = !plan.getPlanNode()->getChildren().empty();
    auto output = printer.printLogicalPlan(*plan.getPlanNode(), TextPrinterIntent{0, has_children}, profiles);

    for (auto & item : plan.getCTEInfo().getCTEs())
    {
        output += "CTEDef [" + std::to_string(item.first) + "]\n";
        output += printer.printLogicalPlan(*item.second, TextPrinterIntent{3, !item.second->getChildren().empty()}, profiles);
    }

    auto magic_sets = PlanNodeSearcher::searchFrom(plan)
                          .where([](auto & node) {
                              return node.getStep()->getType() == IQueryPlanStep::Type::Join
                                  && dynamic_cast<const JoinStep &>(*node.getStep()).isMagic();
                          })
                          .count();

    if (magic_sets > 0)
        output += "note: Magic Set is applied for " + std::to_string(magic_sets) + " parts.\n";

    auto filter_nodes = PlanNodeSearcher::searchFrom(plan)
                            .where([](auto & node) { return node.getStep()->getType() == IQueryPlanStep::Type::Filter; })
                            .findAll();

    size_t dynamic_filters = 0;
    for (auto & filter : filter_nodes)
    {
        const auto * filter_step = dynamic_cast<const FilterStep *>(filter->getStep().get());
        auto filters = DynamicFilters::extractDynamicFilters(filter_step->getFilter());
        dynamic_filters += filters.first.size();
    }

    if (dynamic_filters > 0)
        output += "note: Dynamic Filter is applied for " + std::to_string(dynamic_filters) + " times.\n";

    auto cte_nodes = PlanNodeSearcher::searchFrom(plan)
                         .where([](auto & node) { return node.getStep()->getType() == IQueryPlanStep::Type::CTERef; })
                         .count();

    if (cte_nodes > 0)
        output += "note: CTE(Common Table Expression) is applied for " + std::to_string(cte_nodes) + " times.\n";

    auto & optimizer_metrics = context->getOptimizerMetrics();
    if (optimizer_metrics && !optimizer_metrics->getUsedMaterializedViews().empty())
    {
        output += "note: Materialized Views is applied for " + std::to_string(optimizer_metrics->getUsedMaterializedViews().size())
            + " times: ";
        const auto & views = optimizer_metrics->getUsedMaterializedViews();
        auto it = views.begin();
        output += it->getDatabaseName() + "." + it->getTableName();
        for (++it; it != views.end(); ++it)
            output += ", " + it->getDatabaseName() + "." + it->getTableName();
        output += ".";
    }

    return output;
}

String PlanPrinter::jsonLogicalPlan(QueryPlan & plan, bool print_stats, bool, const PlanNodeCost & plan_cost)
{
    std::ostringstream os;
    JsonPrinter printer{print_stats};
    Poco::JSON::Object::Ptr json = new Poco::JSON::Object(true);
    if (!plan.getPlanNode()->getStatistics() && plan_cost.getCost() == 0)
    {
        json = printer.printLogicalPlan(*plan.getPlanNode());
    }
    else
    {
        json->set("total_cost", plan_cost.getCost());
        json->set("cpu_cost_value", plan_cost.getCpuValue());
        json->set("net_cost_value", plan_cost.getNetValue());
        json->set("men_cost_value", plan_cost.getMenValue());
        json->set("plan", printer.printLogicalPlan(*plan.getPlanNode()));
    }
    json->stringify(os, 2);
    return os.str();
}

String PlanPrinter::textDistributedPlan(
    PlanSegmentDescriptions & segments_desc,
    bool print_stats,
    bool verbose,
    const std::unordered_map<PlanNodeId, double> & costs,
    const StepAggregatedOperatorProfiles & profiles,
    const QueryPlan & query_plan
)
{
    auto id_to_node = getPlanNodeMap(query_plan);
    for (auto & segment_desc : segments_desc)
    {
        if (segment_desc->segment_id == 0)
        {
            segment_desc->plan_node = query_plan.getPlanNodeRoot();
            continue;
        }

        if (segment_desc->root_id == 0)
            continue;

        PlanNodePtr plan_node;
        if (id_to_node.contains(segment_desc->root_id))
            plan_node = id_to_node.at(segment_desc->root_id);
        else if (segment_desc->root_child_id != 0)
            plan_node = id_to_node.at(segment_desc->root_child_id);
        else
            continue;

        segment_desc->plan_node = plan_node;
    }

    auto f = [](ExchangeMode mode) {
        switch (mode)
        {
            case ExchangeMode::LOCAL_NO_NEED_REPARTITION:
                return "LOCAL_NO_NEED_REPARTITION";
            case ExchangeMode::LOCAL_MAY_NEED_REPARTITION:
                return "LOCAL_MAY_NEED_REPARTITION";
            case ExchangeMode::BROADCAST:
                return "BROADCAST";
            case ExchangeMode::REPARTITION:
                return "REPARTITION";
            case ExchangeMode::GATHER:
                return "GATHER";
            default:
                return "UNKNOWN";
        }
    };

    std::ostringstream os;

    auto cmp = [](const PlanSegmentDescriptionPtr & s1, const PlanSegmentDescriptionPtr & s2) { return s1->segment_id < s2->segment_id; };
    std::sort(segments_desc.begin(), segments_desc.end(), cmp);

    for (auto & segment_ptr : segments_desc)
    {
        String state_partition;
        if (segment_ptr->segment_id == 0)
            state_partition = "SINGLE";
        else if (segment_ptr->is_source)
            state_partition = "SOURCE";
        else
            state_partition = "HASH";

        size_t segment_id = segment_ptr->segment_id;
        os << "Segment[" << segment_id << "] ["<< state_partition << "]\n";

        ExchangeMode mode = segment_ptr->mode;
        String exchange = (segment_id == 0) ? "Output" : f(mode);
        os << "   Output Exchange: " << exchange;
        if (exchange == "REPARTITION") // print shuffle keys
        {
            os << "[";
            bool first = true;
            for (auto & key : segment_ptr->shuffle_keys)
            {
                if (first)
                {
                    os << key;
                }
                os << ", " << key;
            }
            os << "]";
        }
        os << "\n";

        os << "   Parallel Size: " << segment_ptr->parallel;
        os << ", Cluster Name: " << (segment_ptr->cluster_name.empty() ?"server" : segment_ptr->cluster_name);
        os << ", Exchange Parallel Size: " << segment_ptr->exchange_parallel_size;
        os << ", Exchange Output Parallel Size: " << segment_ptr->exchange_parallel_size << "\n";

        if (!segment_ptr->plan_node)
            continue;

        auto analyze_node = PlanNodeSearcher::searchFrom(segment_ptr->plan_node)
                                .where([](auto & node) { return node.getStep()->getType() == IQueryPlanStep::Type::ExplainAnalyze; })
                                .findFirst();
        if (analyze_node)
        {
            os << TextPrinter::printOutputColumns(*analyze_node.value()->getChildren()[0], TextPrinterIntent{3, false});
            TextPrinter printer{print_stats, verbose, costs, true, segment_ptr->exchange_to_segment};
            bool has_children = !analyze_node.value()->getChildren().empty();
            if ((analyze_node.value()->getStep()->getType() == IQueryPlanStep::Type::CTERef || analyze_node.value()->getStep()->getType() == IQueryPlanStep::Type::Exchange))
                has_children = false;

            auto output = printer.printLogicalPlan(*analyze_node.value(), TextPrinterIntent{6, has_children}, profiles);
            os << output;
        }
        else
        {
            auto plan_root = segment_ptr->plan_node;
            os << TextPrinter::printOutputColumns(*segment_ptr->plan_node, TextPrinterIntent{3, false});
            TextPrinter printer{print_stats, verbose, costs, true, segment_ptr->exchange_to_segment};
            bool has_children = !plan_root->getChildren().empty();
            if ((plan_root->getStep()->getType() == IQueryPlanStep::Type::CTERef || plan_root->getStep()->getType() == IQueryPlanStep::Type::Exchange))
                has_children = false;

            auto output = printer.printLogicalPlan(*segment_ptr->plan_node, TextPrinterIntent{6, has_children}, profiles);
            os << output;
        }

        os << "\n";
    }

    return os.str();
}

void PlanPrinter::getRemoteSegmentId(const QueryPlan::Node * node, std::unordered_map<PlanNodeId, size_t> & exchange_to_segment)
{
    auto * step = dynamic_cast<RemoteExchangeSourceStep *>(node->step.get());
    if (step)
        exchange_to_segment[node->id] = step->getInput()[0]->getPlanSegmentId();

    for (const auto & child : node->children)
        getRemoteSegmentId(child, exchange_to_segment);
}

std::unordered_map<PlanNodeId, PlanNodePtr> PlanPrinter::getPlanNodeMap(const QueryPlan & query_plan)
{
    std::unordered_map<PlanNodeId, PlanNodePtr> id_to_node;
    const auto & plan =  query_plan.getPlanNodeRoot();
    if (!plan)
        return id_to_node;

    id_to_node[plan->getId()] = plan;
    getPlanNodes(plan, id_to_node);

    for (const auto & cte : query_plan.getCTEInfo().getCTEs())
    {
        id_to_node[cte.second->getId()] = cte.second;
        getPlanNodes(cte.second, id_to_node);
    }

    return id_to_node;
}

void PlanPrinter::getPlanNodes(const PlanNodePtr & parent, std::unordered_map<PlanNodeId, PlanNodePtr> & id_to_node)
{
    for (const auto & child : parent->getChildren())
    {
        id_to_node[child->getId()] = child;
        if (!child->getChildren().empty())
            getPlanNodes(child, id_to_node);
    }
}

String PlanPrinter::TextPrinter::printOutputColumns(PlanNodeBase & plan_node, const TextPrinterIntent & intent)
{
    auto header = plan_node.getStep()->getOutputStream().header;

    String res;
    size_t line_feed_limit = 120;
    res += intent.print() + "Output Columns: [";

    std::vector<std::string> output_columns;
    for (auto & it : header)
    {
        output_columns.push_back(it.name);
    }
    sort(output_columns.begin(), output_columns.end());

    bool first = true;
    for (auto & column_name : output_columns) {
        if (res.length() > line_feed_limit) {
            res += "\n";
            res += intent.print() + String(17, ' ');
            line_feed_limit += 120;
            first = true;
        }
        if (first) {
            res += column_name;
            first = false;
        }
        else {
            res += ", ";
            res += column_name;
        }
    }
    res += "]\n";
    return res;
}

TextPrinterIntent TextPrinterIntent::forChild(bool last, bool hasChildren_) const
{
    return TextPrinterIntent{
        next_lines_prefix + (last ? LAST_PREFIX : INTERMEDIATE_PREFIX),
        next_lines_prefix + (last ? EMPTY_PREFIX : VERTICAL_LINE),
        hasChildren_};
}

TextPrinterIntent::TextPrinterIntent(String current_lines_prefix_, String next_lines_prefix_, bool hasChildren_)
    : current_lines_prefix(std::move(current_lines_prefix_)), next_lines_prefix(std::move(next_lines_prefix_)), hasChildren(hasChildren_)
{
}

String TextPrinterIntent::detailIntent() const
{
    return "\n" + next_lines_prefix + (hasChildren ? VERTICAL_LINE : EMPTY_PREFIX) + EMPTY_PREFIX;
}

String PlanPrinter::TextPrinter::printLogicalPlan(PlanNodeBase & plan, const TextPrinterIntent & intent, const StepAggregatedOperatorProfiles & profiles) // NOLINT(misc-no-recursion)
{
    std::stringstream out;

    auto step = plan.getStep();
    if (step->getType() == IQueryPlanStep::Type::ExplainAnalyze)
        return printLogicalPlan(*plan.getChildren()[0], intent, profiles);

    if (profiles.empty())
    {
        if (print_stats)
            out << intent.print() << printPrefix(plan) << step->getName() << printSuffix(plan) << " " << printStatistics(plan, intent)
                << printDetail(plan.getStep(), intent) << "\n";
        else
            out << intent.print() << printPrefix(plan) << step->getName() << printSuffix(plan) << printDetail(plan.getStep(), intent) << "\n";
    }
    else
    {
        out << intent.print() << printPrefix(plan) << step->getName() << printSuffix(plan)
            << intent.detailIntent() << printStatistics(plan, intent)
            << printOperatorProfiles(plan, intent, profiles)
            << printQError(plan, intent, profiles)
            << printDetail(plan.getStep(), intent) << "\n";
    }

    if ((step->getType() == IQueryPlanStep::Type::CTERef || step->getType() == IQueryPlanStep::Type::Exchange) && is_distributed)
        return out.str();

    for (auto it = plan.getChildren().begin(); it != plan.getChildren().end();)
    {
        auto child = *it++;
        bool last = it == plan.getChildren().end();
        bool has_children = !child->getChildren().empty();
        if ((child->getStep()->getType() == IQueryPlanStep::Type::CTERef || child->getStep()->getType() == IQueryPlanStep::Type::Exchange) && is_distributed)
            has_children = false;

        out << printLogicalPlan(*child, intent.forChild(last, has_children), profiles);
    }

    return out.str();
}

String PlanPrinter::TextPrinter::printStatistics(const PlanNodeBase & plan, const TextPrinterIntent &) const
{
    if (!print_stats)
        return "";
    std::stringstream out;
    const auto & stats = plan.getStatistics();
    out << "Est. " << (stats ? std::to_string(stats.value()->getRowCount()) : "?") << " rows";
    if (costs.contains(plan.getId()))
        out << ", cost " << std::scientific << costs.at(plan.getId());
    return out.str();
}

String PlanPrinter::TextPrinter::printOperatorProfiles(PlanNodeBase & plan, const TextPrinterIntent & intent, const StepAggregatedOperatorProfiles & profiles)
{
    size_t step_id = plan.getId();
    if (profiles.count(step_id))
    {
        const auto & profile = profiles.at(step_id);
        std::stringstream out;
        out << intent.detailIntent() << "Act. Output " << prettyNum(profile->output_rows) << " rows (" << prettyBytes(profile->output_bytes) << ")";
        out << ", Output wait Time: " << prettySeconds(profile->max_output_wait_elapsed_us);
        out << ", Wall Time: " << prettySeconds(profile->max_elapsed_us);

        int num = 1;
        if (!plan.getChildren().empty() && profile->inputs_profile.contains(plan.getChildren()[0]->getId()))
        {
            for (auto & child : plan.getChildren())
            {
                auto input_profile = profile->inputs_profile[child->getId()];
                if (num == 1)
                    out << intent.detailIntent() << "Input. " ;
                else
                    out << intent.detailIntent() << "       ";

                if (plan.getChildren().size() > 1)
                    out << "source [" << num << "] : ";

                out <<  prettyNum(input_profile.input_rows) << " rows (" << prettyBytes(input_profile.input_bytes) << ")";
                out << ", Input wait Time: " << prettySeconds(input_profile.input_wait_elapsed_us);
                ++num;
            }
        }
        else
        {
            for (auto & [id, input_metrics] : profile->inputs_profile)
            {
                if (num == 1)
                    out << intent.detailIntent() << "Input. " ;
                else
                    out << intent.detailIntent() << "       ";

                if (plan.getChildren().size() > 1)
                    out << "source [" << num << "] : ";

                out << "Input wait Time: " << prettySeconds(input_metrics.input_wait_elapsed_us);
                ++num;
            }
        }

        return out.str();
    }
    return "";
}

String PlanPrinter::TextPrinter::prettyNum(size_t num)
{
    std::vector<std::string> suffixes{ "", "K", "M", "B", "T" };
    size_t idx = 0;
    auto count = static_cast<double>(num);
    while (count >= 1000 && idx < suffixes.size() - 1)
    {
        idx++;
        count /= static_cast<double>(1000);
    }

    std::stringstream out;
    if (idx == 0)
        out << static_cast<int>(count);
    else
        out << std::fixed << std::setprecision(1) << count << suffixes[idx];
    return out.str();
}

String PlanPrinter::TextPrinter::prettySeconds(size_t us)
{
    std::vector<std::string> suffixes{ " us", " ms", " s" };
    size_t idx = 0;
    auto count = static_cast<double>(us);
    while (count >= 1000 && idx < suffixes.size() - 1)
    {
        idx++;
        count /= static_cast<double>(1000);
    }

    std::stringstream out;
    out << std::fixed << std::setprecision(1) << count << suffixes[idx];
    return out.str();
}

String PlanPrinter::TextPrinter::prettyBytes(size_t bytes)
{
    std::vector<std::string> suffixes{ " Bytes", " KB", " MB", " GB", " TB" };
    size_t idx = 0;
    auto count = static_cast<double>(bytes);
    while (count >= 1024 && idx < suffixes.size() - 1)
    {
        idx++;
        count /= static_cast<double>(1024);
    }

    std::stringstream out;
    out << std::fixed << std::setprecision(1) << count << suffixes[idx];
    return out.str();
}

String PlanPrinter::TextPrinter::printQError(PlanNodeBase & plan, const TextPrinterIntent & intent, const StepAggregatedOperatorProfiles & profiles)
{
    const auto & stats = plan.getStatistics();
    std::stringstream out;

    size_t step_id = plan.getId();
    if (profiles.count(step_id))
    {
        const auto& profile = profiles.at(step_id);
        out << intent.detailIntent();
        if (plan.getChildren().size() > 1)
        {
            size_t max_input_rows = 0;
            for (auto & p : plan.getChildren())
            {
                if (profiles.count(p->getId()) == 0)
                    continue;
                max_input_rows = std::max(max_input_rows, profiles.at(p->getId())->output_rows);
            }
            if (max_input_rows == 0)
                out << "Filtered: 0.0%";
            else
            {
                double max_rows = static_cast<double>(max_input_rows);
                out << "Filtered: " << std::fixed << std::setprecision(1)
                    << (max_rows - static_cast<double>(profile->output_rows)) * static_cast<double>(100) / max_rows << "%";
            }

        }
        else if (plan.getChildren().size() == 1)
        {
            if (profiles.count(plan.getChildren()[0]->getId()) == 0)
                out << "Filtered: 0.0%";
            else
            {
                auto child_input_rows = static_cast<double>(profiles.at(plan.getChildren()[0]->getId())->output_rows);
                out << "Filtered: " << std::fixed << std::setprecision(1)
                    << (child_input_rows - static_cast<double>(profile->output_rows)) * static_cast<double>(100) / child_input_rows << "%";
            }
        }
        else
        {
            out << "Filtered: 0.0%";
        }

        if (stats && profile->output_rows != 0)
        {
            if (profile->output_rows > stats.value()->getRowCount())
                out << ", QError: " << std::fixed << std::setprecision(1) << static_cast<double>(profile->output_rows) / static_cast<double>(stats.value()->getRowCount());
            else
                out << ", QError: " << std::fixed << std::setprecision(1) << static_cast<double>(stats.value()->getRowCount()) / static_cast<double>(profile->output_rows);
        }
        return out.str();
    }
    return "";
}

String PlanPrinter::TextPrinter::printPrefix(PlanNodeBase & plan)
{
    if (plan.getStep()->getType() == IQueryPlanStep::Type::Exchange)
    {
        const auto * exchange = dynamic_cast<const ExchangeStep *>(plan.getStep().get());
        auto f = [](ExchangeMode mode) {
            switch (mode)
            {
                case ExchangeMode::LOCAL_NO_NEED_REPARTITION:
                case ExchangeMode::LOCAL_MAY_NEED_REPARTITION:
                    return "Local ";
                case ExchangeMode::BROADCAST:
                    return "Broadcast ";
                case ExchangeMode::REPARTITION:
                    return "Repartition ";
                case ExchangeMode::GATHER:
                    return "Gather ";
                default:
                    return "";
            }
        };
        return f(exchange->getExchangeMode());
    }

    if (plan.getStep()->getType() == IQueryPlanStep::Type::Join)
    {
        const auto * join = dynamic_cast<const JoinStep *>(plan.getStep().get());
        auto f = [](ASTTableJoin::Kind kind) {
            switch (kind)
            {
                case ASTTableJoin::Kind::Inner:
                    return "Inner ";
                case ASTTableJoin::Kind::Left:
                    return "Left ";
                case ASTTableJoin::Kind::Right:
                    return "Right ";
                case ASTTableJoin::Kind::Full:
                    return "Full ";
                case ASTTableJoin::Kind::Cross:
                    return "Cross ";
                default:
                    return "";
            }
        };

        return f(join->getKind());
    }
    return "";
}

String PlanPrinter::TextPrinter::printSuffix(PlanNodeBase & plan)
{
    std::stringstream out;
    Int64 segment_id = -1;
    if (is_distributed && exchange_to_segment.contains(plan.getId()))
        segment_id = exchange_to_segment.at(plan.getId());

    if (plan.getStep()->getType() == IQueryPlanStep::Type::TableScan)
    {
        const auto * table_scan = dynamic_cast<const TableScanStep *>(plan.getStep().get());
        out << " " << table_scan->getDatabase() << "." << table_scan->getOriginalTable();
    }
    else if (plan.getStep()->getType() == IQueryPlanStep::Type::Exchange && segment_id != -1)
    {
        out << " segment[" << exchange_to_segment.at(plan.getId()) << "]";
    }
    else if (plan.getStep()->getType() == IQueryPlanStep::Type::CTERef)
    {
        const auto *cte = dynamic_cast<const CTERefStep *>(plan.getStep().get());
        out << "[" << cte->getId() << "]" ;
        if (segment_id != -1)
            out << " <--" << " segment[" << exchange_to_segment.at(plan.getId()) << "]";
    }
    return out.str();
}

String PlanPrinter::TextPrinter::printDetail(QueryPlanStepPtr plan, const TextPrinterIntent & intent) const
{
    std::stringstream out;
    if (verbose && plan->getType() == IQueryPlanStep::Type::Join)
    {
        const auto * join = dynamic_cast<const JoinStep *>(plan.get());
        out << intent.detailIntent() << "Condition: ";
        if (!join->getLeftKeys().empty())
            out << join->getLeftKeys()[0] << " == " << join->getRightKeys()[0];
        for (size_t i = 1; i < join->getLeftKeys().size(); i++)
            out << ", " << join->getLeftKeys()[i] << " == " << join->getRightKeys()[i];

        if (!ASTEquality::compareTree(join->getFilter(), PredicateConst::TRUE_VALUE))
        {
            out << intent.detailIntent() << "Filter: ";
            out << serializeAST(*join->getFilter());
        }
    }

    if (verbose && plan->getType() == IQueryPlanStep::Type::Sorting)
    {
        const auto *sort = dynamic_cast<const SortingStep *>(plan.get());
        std::vector<String> sort_columns;
        for (const auto & desc : sort->getSortDescription())
            sort_columns.emplace_back(
                desc.column_name + (desc.direction == -1 ? " desc" : " asc") + (desc.nulls_direction == -1 ? " nulls_last" : ""));
        out << intent.detailIntent() << "Order by: " << join(sort_columns, ", ", "{", "}");
        if (sort->getLimit())
            out << intent.detailIntent() << "Limit: " << sort->getLimit();
    }


    if (verbose && plan->getType() == IQueryPlanStep::Type::Limit)
    {
        const auto * limit = dynamic_cast<const LimitStep *>(plan.get());
        out << intent.detailIntent();
        if (limit->getLimit())
            out << "Limit: " << limit->getLimit();
        if (limit->getOffset())
            out << "Offset: " << limit->getOffset();
    }

    if (verbose && plan->getType() == IQueryPlanStep::Type::Aggregating)
    {
        const auto * agg = dynamic_cast<const AggregatingStep *>(plan.get());
        auto keys = agg->getKeys();
        std::sort(keys.begin(), keys.end());
        out << intent.detailIntent() << "Group by: " << join(keys, ", ", "{", "}");

        std::vector<String> aggregates;
        for (const auto & desc : agg->getAggregates())
        {
            std::stringstream ss;
            String func_name = desc.function->getName();
            auto type_name = String(typeid(desc.function.get()).name());
            if (type_name.find("AggregateFunctionNull"))
                func_name = String("AggNull(").append(std::move(func_name)).append(")");
            ss << desc.column_name << ":=" << func_name << join(desc.argument_names, "," ,"(", ")");
            aggregates.emplace_back(ss.str());
        }
        if (!aggregates.empty())
            out << intent.detailIntent() << "Aggregates: " << join(aggregates, ", ");
    }

    if (verbose && plan->getType() == IQueryPlanStep::Type::Exchange)
    {
        const auto * exchange = dynamic_cast<const ExchangeStep *>(plan.get());
        if (exchange->getExchangeMode() == ExchangeMode::REPARTITION)
        {
            auto keys = exchange->getSchema().getPartitioningColumns();
            std::sort(keys.begin(), keys.end());
            out << intent.detailIntent() << "Partition by: " << join(keys, ", ", "{", "}");
        }
    }

    if (verbose && plan->getType() == IQueryPlanStep::Type::Filter)
    {
        const auto * filter = dynamic_cast<const FilterStep *>(plan.get());
        auto filters = DynamicFilters::extractDynamicFilters(filter->getFilter());
        if (!filters.second.empty())
            out << intent.detailIntent() << "Condition: " << serializeAST(*PredicateUtils::combineConjuncts(filters.second));
        if (!filters.first.empty())
        {
            std::vector<std::string> dynamic_filters;
            for (auto & item : filters.first)
                dynamic_filters.emplace_back(DynamicFilters::toString(DynamicFilters::extractDescription(item).value()));
            std::sort(dynamic_filters.begin(), dynamic_filters.end());
            out << intent.detailIntent() << "Dynamic Filters: " << join(dynamic_filters, ",", "{", "}");
        }
    }

    if (verbose && plan->getType() == IQueryPlanStep::Type::Projection)
    {
        const auto * projection = dynamic_cast<const ProjectionStep *>(plan.get());

        std::vector<String> identities;
        std::vector<String> assignments;

        for (const auto & assignment : projection->getAssignments())
            if (Utils::isIdentity(assignment))
                identities.emplace_back(assignment.first);
            else
                assignments.emplace_back(assignment.first + ":=" + serializeAST(*assignment.second));

        std::sort(assignments.begin(), assignments.end());
        if (!identities.empty())
        {
            std::stringstream ss;
            std::sort(identities.begin(), identities.end());
            ss << join(identities, ", ", "[", "]");
            assignments.insert(assignments.begin(), ss.str());
        }

        out << intent.detailIntent() << "Expressions: " << join(assignments, ", ");

        if (!projection->getDynamicFilters().empty())
        {
            std::vector<std::string> dynamic_filters;
            for (const auto & item : projection->getDynamicFilters())
                dynamic_filters.emplace_back(item.first);
            std::sort(dynamic_filters.begin(), dynamic_filters.end());
            out << intent.detailIntent() << "Dynamic Filters Builder: " << join(dynamic_filters, ",", "{", "}");
        }
    }

    if (verbose && plan->getType() == IQueryPlanStep::Type::TableScan)
    {
        const auto * table_scan = dynamic_cast<const TableScanStep *>(plan.get());
        std::vector<String> identities;
        std::vector<String> assignments;
        for (const auto & name_with_alias : table_scan->getColumnAlias())
            if (name_with_alias.second == name_with_alias.first)
                identities.emplace_back(name_with_alias.second);
            else
                assignments.emplace_back(name_with_alias.second + ":=" + name_with_alias.first);

        auto query_info = table_scan->getQueryInfo();
        auto *query = query_info.query->as<ASTSelectQuery>();
        if (query->getWhere() || query->getPrewhere() || query->implicitWhere() || query->getLimitLength())
        {
            out << intent.detailIntent();
            if (query->getWhere())
            {
                auto filters = DynamicFilters::extractDynamicFilters(query->getWhere());
                if (!filters.second.empty())
                {
                    out << "Condition : " << serializeAST(*PredicateUtils::combineConjuncts(filters.second));
                    out << ".";
                }
                if (!filters.first.empty())
                {
                    std::vector<std::string> dynamic_filters;
                    for (auto & item : filters.first)
                        dynamic_filters.emplace_back(DynamicFilters::toString(DynamicFilters::extractDescription(item).value()));
                    std::sort(dynamic_filters.begin(), dynamic_filters.end());
                    out << "Dynamic Filters : " << join(dynamic_filters, ", ", "{", "}");
                    out << ".";
                }
            }

            if (query->getPrewhere())
            {
                out << "Prewhere : ";
                out << serializeAST(*query->getPrewhere());
                out << ".";
            }
            if (query->implicitWhere())
            {
                out << "Implicit Filter : ";
                out << serializeAST(*query->implicitWhere());
                out << ".";
            }

            if (query->getLimitLength())
            {
                out << "Limit : ";
                Field converted = convertFieldToType(query->refLimitLength()->as<ASTLiteral>()->value, DataTypeUInt64());
                out << converted.safeGet<UInt64>();
                out << ".";
            }
        }

        std::sort(assignments.begin(), assignments.end());
        if (!identities.empty())
        {
            std::stringstream ss;
            std::sort(identities.begin(), identities.end());
            ss << join(identities, ", ", "[", "]");
            assignments.insert(assignments.begin(), ss.str());
        }

        out << intent.detailIntent() << "Outputs: " << join(assignments, ", ");

        if (table_scan->getPushdownFilter())
            out << printDetail(table_scan->getPushdownFilter(), intent);

        if (table_scan->getPushdownProjection())
            out << printDetail(table_scan->getPushdownProjection(), intent);

        if (table_scan->getPushdownAggregation())
            out << printDetail(table_scan->getPushdownAggregation(), intent);
    }

    if (verbose && plan->getType() == IQueryPlanStep::Type::TopNFiltering)
    {
        auto topn_filter = dynamic_cast<const TopNFilteringStep *>(plan.get());
        std::vector<String> sort_columns;
        for (auto & desc : topn_filter->getSortDescription())
            sort_columns.emplace_back(
                desc.column_name + (desc.direction == -1 ? " desc" : " asc") + (desc.nulls_direction == -1 ? " nulls_last" : ""));
        out << intent.detailIntent() << "Order by: " << join(sort_columns, ", ", "{", "}");
        out << intent.detailIntent() << "Size: " << topn_filter->getSize();
    }
    return out.str();
}

Poco::JSON::Object::Ptr PlanPrinter::JsonPrinter::printLogicalPlan(PlanNodeBase & plan) // NOLINT(misc-no-recursion)
{
    Poco::JSON::Object::Ptr json = new Poco::JSON::Object(true);
    json->set("id", plan.getId());
    json->set("type", IQueryPlanStep::toString(plan.getStep()->getType()));
    detail(json, plan.getStep());

    if (print_stats)
    {
        const auto & statistics = plan.getStatistics();
        if (statistics)
            json->set("statistics", statistics.value()->toJson());
    }

    if (!plan.getChildren().empty())
    {
        Poco::JSON::Array children;
        for (auto & child : plan.getChildren())
            children.add(printLogicalPlan(*child));

        json->set("children", children);
    }

    return json;
}

void PlanPrinter::JsonPrinter::detail(Poco::JSON::Object::Ptr & json, QueryPlanStepPtr plan)
{
    if (plan->getType() == IQueryPlanStep::Type::Join)
    {
        const auto *join = dynamic_cast<const JoinStep *>(plan.get());
        Poco::JSON::Array left_keys;
        Poco::JSON::Array right_keys;

        for (size_t i = 0; i < join->getLeftKeys().size(); i++)
        {
            left_keys.add(join->getLeftKeys()[i]);
            right_keys.add(join->getRightKeys()[i]);
        }
        json->set("leftKeys", left_keys);
        json->set("rightKeys", right_keys);

        json->set("filter", serializeAST(*join->getFilter()));

        auto f = [](ASTTableJoin::Kind kind) {
            switch (kind)
            {
                case ASTTableJoin::Kind::Inner:
                    return "inner";
                case ASTTableJoin::Kind::Left:
                    return "left";
                case ASTTableJoin::Kind::Right:
                    return "right";
                case ASTTableJoin::Kind::Full:
                    return "full";
                case ASTTableJoin::Kind::Cross:
                    return "cross";
                default:
                    return "";
            }
        };

        json->set("kind", f(join->getKind()));
    }

    if (plan->getType() == IQueryPlanStep::Type::Aggregating)
    {
        const auto * agg = dynamic_cast<const AggregatingStep *>(plan.get());
        Poco::JSON::Array keys;
        for (const auto & item : agg->getKeys())
            keys.add(item);
        json->set("groupKeys", keys);
    }

    if (plan->getType() == IQueryPlanStep::Type::Exchange)
    {
        const auto * exchange = dynamic_cast<const ExchangeStep *>(plan.get());
        Poco::JSON::Array keys;
        for (const auto & item : (exchange->getSchema().getPartitioningColumns()))
            keys.add(item);
        json->set("partitionKeys", keys);

        auto f = [](ExchangeMode mode) {
            switch (mode)
            {
                case ExchangeMode::LOCAL_NO_NEED_REPARTITION:
                case ExchangeMode::LOCAL_MAY_NEED_REPARTITION:
                    return "local";
                case ExchangeMode::BROADCAST:
                    return "broadcast";
                case ExchangeMode::REPARTITION:
                    return "repartition";
                case ExchangeMode::GATHER:
                    return "gather";
                default:
                    return "";
            }
        };
        json->set("mode", f(exchange->getExchangeMode()));
    }

    if (plan->getType() == IQueryPlanStep::Type::Filter)
    {
        const auto * filter = dynamic_cast<const FilterStep *>(plan.get());
        auto filters = DynamicFilters::extractDynamicFilters(filter->getFilter());
        json->set("filter", serializeAST(*PredicateUtils::combineConjuncts(filters.second)));
        if (!filters.first.empty())
            json->set("dynamicFilter", serializeAST(*PredicateUtils::combineConjuncts(filters.first)));
    }

    if (plan->getType() == IQueryPlanStep::Type::TableScan)
    {
        const auto * table_scan = dynamic_cast<const TableScanStep *>(plan.get());
        json->set("database", table_scan->getDatabase());
        json->set("table", table_scan->getTable());

        if (table_scan->getPushdownFilter())
        {
            Poco::JSON::Object::Ptr sub_json = new Poco::JSON::Object(true);
            detail(sub_json, table_scan->getPushdownFilter());
            json->set("pushdownFilter", sub_json);
        }

        if (table_scan->getPushdownProjection())
        {
            Poco::JSON::Object::Ptr sub_json = new Poco::JSON::Object(true);
            detail(sub_json, table_scan->getPushdownProjection());
            json->set("pushdownProjection", sub_json);
        }

        if (table_scan->getPushdownAggregation())
        {
            Poco::JSON::Object::Ptr sub_json = new Poco::JSON::Object(true);
            detail(sub_json, table_scan->getPushdownAggregation());
            json->set("pushdownAggregation", sub_json);
        }
    }
}

}
