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

#include <Processors/QueryPlan/TableScanStep.h>

#include <Formats/FormatSettings.h>
// #include <Functions/FunctionsInBloomFilter.h>
#include <Interpreters/ActionsVisitor.h>
#include <Interpreters/InterpreterSelectQuery.h>
#include <Interpreters/TableJoin.h>
#include <Interpreters/convertFieldToType.h>
#include <Interpreters/evaluateConstantExpression.h>
#include <Interpreters/misc.h>
// #include <MergeTreeCommon/CnchBucketTableCommon.h>
// #include <Optimizer/DynamicFilters.h>
#include <Optimizer/PredicateUtils.h>
#include <Optimizer/SymbolTransformMap.h>
#include <Optimizer/Rule/Rewrite/PushIntoTableScanRules.h>
#include <Parsers/ASTFunction.h>
#include <Parsers/ASTIdentifier.h>
#include <Parsers/ASTSelectQuery.h>
#include <Parsers/ASTTablesInSelectQuery.h>
#include <Parsers/ASTTableColumnReference.h>
#include <QueryPipeline/QueryPipelineBuilder.h>
#include <Processors/Sources/NullSource.h>
#include <Processors/Transforms/ExpressionTransform.h>
#include <Processors/QueryPlan/PlanSerDerHelper.h>
#include <Processors/QueryPlan/QueryPlan.h>
#include <Processors/QueryPlan/Optimizations/QueryPlanOptimizationSettings.h>
#include <Storages/IStorage_fwd.h>
#include <Storages/MergeTree/MergeTreeWhereOptimizer.h>
#include <Storages/VirtualColumnUtils.h>
// #include <Storages/StorageCloudMergeTree.h>
// #include <Storages/StorageCnchMergeTree.h>
// #include <Storages/StorageCnchHive.h>
#include <Storages/StorageDistributed.h>
#include <Storages/StorageReplicatedMergeTree.h>
#include <Storages/VirtualColumnUtils.h>
#include <Common/FieldVisitorToString.h>
#include "Interpreters/DatabaseCatalog.h"
#include <Common/Stopwatch.h>

#include <fmt/format.h>

namespace DB
{
namespace ErrorCodes
{
    extern const int ILLEGAL_PREWHERE;
    extern const int INVALID_LIMIT_EXPRESSION;
    extern const int PROJECTION_SELECTION_ERROR;
}

namespace _scan_execute_impl
{
    /// interfaces
    // parts within a same group have the same columns & available projections
    struct PartGroup
    {
        MergeTreeData::DataPartsVector parts;

        bool hasProjection(const String & projection_name) const
        {
            return parts.front()->hasProjection(projection_name);
        }

        const NamesAndTypesList & getColumns() const
        {
            return parts.front()->getColumns();
        }

        String toString() const
        {
            std::stringstream ss;
            ss << '{';
            for (const auto & part: parts)
                ss << part->getNameWithState() << "";
            ss << '}';
            return ss.str();
        }

        size_t partsNum() const
        {
            return parts.size();
        }
    };

    using PartGroups = std::vector<PartGroup>;

    struct ExecutePlanElement
    {
        PartGroup part_group;
        MergeTreeDataSelectAnalysisResultPtr read_analysis;

        // if not use projection, below members are not empty
        MergeTreeData::DataPartsVector parts;

        // if use projection, below members are not empty
        ProjectionDescriptionRawPtr projection_desc = nullptr;
        Names projection_required_columns;
        QueryPlanStepPtr rewritten_projection_step;
        QueryPlanStepPtr rewritten_filter_step;
        ActionsDAGPtr prewhere_actions;

        ExecutePlanElement(PartGroup part_group_,
                           MergeTreeDataSelectAnalysisResultPtr read_analysis_,
                           MergeTreeData::DataPartsVector parts_)
            : part_group(std::move(part_group_)),
            read_analysis(std::move(read_analysis_)),
            parts(std::move(parts_))
        {}

        ExecutePlanElement(PartGroup part_group_,
                           MergeTreeDataSelectAnalysisResultPtr read_analysis_,
                           ProjectionDescriptionRawPtr projection_desc_,
                           Names required_columns_,
                           QueryPlanStepPtr projection_,
                           QueryPlanStepPtr filter_,
                           ActionsDAGPtr prewhere_actions_)
            : part_group(std::move(part_group_)),
            read_analysis(std::move(read_analysis_)),
            projection_desc(projection_desc_),
            projection_required_columns(std::move(required_columns_)),
            rewritten_projection_step(std::move(projection_)),
            rewritten_filter_step(std::move(filter_)),
            prewhere_actions(std::move(prewhere_actions_))
        {}

        String toString() const;
    };

    String ExecutePlanElement::toString() const
    {
        std::ostringstream os;
        if (!read_analysis->error())
            os << "Read Marks: " << read_analysis->marks() << std::endl;

        if (projection_desc)
        {
            os << "Used Projection: " << projection_desc->name << std::endl;
            os << "Projection Type: " << (projection_desc->type == ProjectionDescription::Type::Aggregate ? "aggregate" : "normal" )<< std::endl;

            if (rewritten_filter_step)
            {
                auto filter_dump = serializeAST(*(dynamic_cast<FilterStep &>(*rewritten_filter_step).getFilter()));
                os << "Rewritten Filter: " << filter_dump << std::endl;
            }

            if (rewritten_projection_step)
            {
                os << "Column Mapping: " << std::endl;
                auto assignment_dump = dynamic_cast<ProjectionStep &>(*rewritten_projection_step).getAssignments().dump();
                os << assignment_dump;
            }
        }

        return os.str();
    }

    class ExecutePlan: public std::vector<ExecutePlanElement>
    {};

    // a struct used in projection selection with any necessary intermediate states
    struct ProjectionMatchContext;
    using ProjectionMatchContexts = std::vector<ProjectionMatchContext>;

    /// implementations
    struct ProjectionMatchContext
    {
        const ProjectionDescription & projection_desc;
        NamesAndTypesList missing_columns;
        SymbolTranslationMap column_translation;
        NameToType column_types;
        Assignments rewritten_assignments;
        NameToType rewritten_types;
        ASTPtr rewritten_filter;
        NameSet required_column_set;
        MergeTreeDataSelectAnalysisResultPtr read_analysis;

        ProjectionMatchContext(const ProjectionDescription & projection_desc_,
                               const PartGroup & part_group,
                               StoragePtr storage,
                               const NamesAndTypesList & table_columns);

        ExecutePlanElement buildExecutePlanElement(PartGroup part_group, const SelectQueryInfo & query_info) const;

        Names requiredColumns() const
        {
            Names result {required_column_set.begin(), required_column_set.end()};
            return result;
        }
    };

    ProjectionMatchContext::ProjectionMatchContext(const ProjectionDescription & projection_desc_,
                                                   const PartGroup & part_group,
                                                   StoragePtr storage,
                                                   const NamesAndTypesList & table_columns)
        : projection_desc(projection_desc_)
        , column_translation(storage.get())
    {
        const auto & data_part_columns = part_group.getColumns();

        for (const auto & column: table_columns)
            if (!data_part_columns.contains(column.name))
                missing_columns.push_back(column);

        // for (size_t i = 0; i < projection_desc.column_names.size(); ++i)
        // {
        //     // clone AST first since `addTranslation` will modify the AST
        //     column_translation.addTranslation(projection_desc.column_asts[i]->clone(), projection_desc.column_names[i]);
        //     column_types.emplace(projection_desc.column_names[i], projection_desc.data_types[i]);
        // }
        size_t i = 0;
        for (const auto & column : projection_desc.metadata->getColumns())
        {
            column_translation.addTranslation(projection_desc.column_asts[i]->clone(), column.name);
            column_types.emplace(column.name, column.type);
            ++i;
        }
    }

    ExecutePlanElement ProjectionMatchContext::buildExecutePlanElement(PartGroup part_group, const SelectQueryInfo & query_info) const
    {
        QueryPlanStepPtr rewritten_filter_step = nullptr;
        QueryPlanStepPtr rewritten_projection_step = nullptr;
        auto require_columns = requiredColumns();

        NamesAndTypes names_and_types;
        for (const auto & column: require_columns)
            names_and_types.emplace_back(column, column_types.at(column));

        DataStream input_stream {.header = Block{names_and_types}};

        if (rewritten_filter)
            rewritten_filter_step = std::make_shared<FilterStep>(input_stream, rewritten_filter);

        rewritten_projection_step = std::make_shared<ProjectionStep>(rewritten_filter ? rewritten_filter_step->getOutputStream(): input_stream,
                                                                     rewritten_assignments, rewritten_types);

        ActionsDAGPtr prewhere_actions;

        if (query_info.prewhere_info)
        {
            throw Exception(ErrorCodes::ILLEGAL_PREWHERE, "Prewhere is not supported in projection selection");
            // auto & prewhere_info = query_info.prewhere_info;
            // prewhere_actions = prewhere_info->prewhere_actions->clone();
            // prewhere_actions->foldActionsByProjection(
            //     required_column_set, projection_desc.sample_block_for_keys, prewhere_info->prewhere_column_name, false);

            // // This is hacky. Since `query_info.query` is not the original query, `prewhere_actions` may prune off
            // // columns mistakenly. Here, we add used required columns back.
            // // For example. projection desc: SELECT toDate(at), count() GROUP BY toDate(at)
            // // query: SELECT toDate(at), count() PREWHERE toDate(at) = '2022-01-01' GROUP BY toDate(at)
            // auto & index = prewhere_actions->getIndex();
            // for (const auto & input: prewhere_actions->getInputs())
            //     if (std::find(index.begin(), index.end(), input) == index.end())
            //         index.emplace_back(input);
        }

        return ExecutePlanElement {
            std::move(part_group),
            read_analysis,
            &projection_desc,
            std::move(require_columns),
            rewritten_projection_step,
            rewritten_filter_step,
            prewhere_actions
        };
    }

    struct PartSchemaKey
    {
        // use ordered set to ensure hash function stable
        std::set<std::string_view> columns;
        std::set<std::string_view> projections;

        PartSchemaKey(MergeTreeData::DataPartPtr part);

        bool operator==(const PartSchemaKey & other) const
        {
            return columns == other.columns && projections == other.projections;
        }

        struct Hash
        {
            size_t operator()(const PartSchemaKey & key) const
            {
                size_t res = 17;

                for (const auto & column: key.columns)
                    res = res * 31 + std::hash<std::string_view>()(column);

                for (const auto & projection: key.projections)
                    res = res * 31 + std::hash<std::string_view>()(projection);

                return res;
            }
        };
    };


    PartSchemaKey::PartSchemaKey(MergeTreeData::DataPartPtr part)
    {
        std::vector<std::string_view> columns_vector;
        std::vector<std::string_view> projections_vector;

        for (const auto & column_entry: part->getColumns())
            columns_vector.emplace_back(column_entry.name);
        for (const auto & projection_entry: part->getProjectionParts())
            projections_vector.emplace_back(projection_entry.first);

        columns.insert(columns_vector.begin(), columns_vector.end());
        projections.insert(projections_vector.begin(), projections_vector.end());
    }
}

using namespace _scan_execute_impl;

class TableScanExecutor
{
public:
    TableScanExecutor(TableScanStep & step, ContextPtr context_);
    ExecutePlan buildExecutePlan();

private:
    bool match(ProjectionMatchContext & candidate) const;
    MergeTreeDataSelectAnalysisResultPtr estimateReadMarks(const PartGroup & part_group) const;
    void estimateReadMarksForProjection(const PartGroup & part_group, ProjectionMatchContext & candidate) const;
    void prunePartsByIndex(MergeTreeData::DataPartsVector & parts) const;
    PartGroups groupPartsBySchema(const MergeTreeData::DataPartsVector & parts);
    ASTPtr rewriteExpr(ASTPtr expr, ProjectionMatchContext & candidate) const;

    struct NameWithAST
    {
        String name;
        ASTPtr flatten_ast;
    };

    StoragePtr storage;
    StorageMetadataPtr storage_metadata;
    const MergeTreeData & merge_tree_data;
    MergeTreeDataSelectExecutor merge_tree_reader;
    const SelectQueryInfo & select_query_info;
    ContextPtr context;
    Poco::Logger * log;

    bool has_aggregate;
    std::optional<SymbolTransformMap> query_lineage;
    Names query_required_columns;
    NameToType column_types_before_agg;
    std::vector<NameWithAST> aggregate_keys;
    std::vector<NameWithAST> aggregate_descs;
    ASTPtr flatten_filter;
    std::shared_ptr<PartitionIdToMaxBlock> max_added_blocks;
};

TableScanExecutor::TableScanExecutor(TableScanStep & step, ContextPtr context_)
    : storage(step.getStorage())
    , storage_metadata(storage->getInMemoryMetadataPtr())
    // , merge_tree_data(dynamic_cast<const MergeTreeMetaBase &>(*storage))
    , merge_tree_data(dynamic_cast<const MergeTreeData &>(*storage))
    , merge_tree_reader(merge_tree_data)
    , select_query_info(step.getQueryInfo())
    , context(std::move(context_))
    , log(&Poco::Logger::get("TableScanExecutor"))
{
    if (storage_metadata->projections.empty())
        return;

    has_aggregate = step.getPushdownAggregation() != nullptr;
    query_required_columns = step.getColumnNames();
    query_lineage = [&]()
    {
        PlanNodeId node_id = 0;
        PlanNodePtr node;
        QueryPlanStepPtr table_scan_without_pushdown_steps = std::make_shared<TableScanStep>(
            context,
            step.getStorageID(),
            step.getColumnAlias(),
            step.getQueryInfo(),
            step.getMaxBlockSize());
        node = PlanNodeBase::createPlanNode(node_id++, table_scan_without_pushdown_steps);

        if (const auto & filter = step.getPushdownFilter())
            node = PlanNodeBase::createPlanNode(node_id++, filter, {node});

        if (const auto &  projection = step.getPushdownProjection())
            node = PlanNodeBase::createPlanNode(node_id++, projection, {node});

        if (const auto &  aggregation = step.getPushdownAggregation())
            node = PlanNodeBase::createPlanNode(node_id++, aggregation, {node});

        return SymbolTransformMap::buildFrom(*node);
    }();

    if (!query_lineage)
        return;

    if (has_aggregate)
    {
        const auto * query_aggregate = step.getPushdownAggregationCast();
        column_types_before_agg = query_aggregate->getInputStreams()[0].header.getNamesToTypes();

        for (const auto & origin_grouping_key: query_aggregate->getKeys())
            aggregate_keys.emplace_back(NameWithAST{origin_grouping_key, query_lineage->inlineReferences(origin_grouping_key)});

        for (const auto & query_aggregate_desc: query_aggregate->getAggregates())
            aggregate_descs.emplace_back(NameWithAST{query_aggregate_desc.column_name,
                                                     query_lineage->inlineReferences(query_aggregate_desc.column_name)});
    }

    if (const auto * query_filter_step = step.getPushdownFilterCast())
    {
        const auto & query_filter = query_filter_step->getFilter();
        flatten_filter = query_lineage->inlineReferences(query_filter);
    }

    const auto & settings = context->getSettingsRef();
    if (settings.select_sequential_consistency)
    {
        if (const StorageReplicatedMergeTree * replicated = dynamic_cast<const StorageReplicatedMergeTree *>(storage.get()))
            max_added_blocks = std::make_shared<PartitionIdToMaxBlock>(replicated->getMaxAddedBlocks());
    }
}

ExecutePlan TableScanExecutor::buildExecutePlan()
{
    if (storage_metadata->projections.empty())
        return {};

    PartGroups part_groups;
    {
        MergeTreeData::DataPartStateVector out_states{MergeTreeData::DataPartState::Active};
        auto parts = merge_tree_data.getAllDataPartsVector(&out_states);
        parts.erase(std::remove_if(parts.begin(), parts.end(), [](auto & part) { return part->info.isFakeDropRangePart(); }), parts.end());

        LOG_DEBUG(log, "Num of parts before part pruning: {}", std::to_string(parts.size()));

        prunePartsByIndex(parts);

        LOG_DEBUG(log, "Num of parts after part pruning: {}", std::to_string(parts.size()));

        if (parts.size() > 100000)
            throw Exception(ErrorCodes::PROJECTION_SELECTION_ERROR, "Projection selection error: too many parts before part grouping");

        part_groups = groupPartsBySchema(parts);

        LOG_DEBUG(log, "Num of part groups before part grouping: {}", std::to_string(part_groups.size()));

        if (part_groups.size() > 100)
            throw Exception(ErrorCodes::PROJECTION_SELECTION_ERROR, "Projection selection error: Too many part groups before projection selection");
    }

    ExecutePlan execute_plan;
    auto table_columns = storage_metadata->getColumns().getAllPhysical();

    for (auto & part_group: part_groups)
    {
        ProjectionMatchContexts projection_candidates;

        // collect qualified projection candidate
        for (const auto & projection_desc: storage_metadata->projections)
            if (part_group.hasProjection(projection_desc.name))
            {
                projection_candidates.emplace_back(projection_desc, part_group, storage, table_columns);
                if (!match(projection_candidates.back()))
                    projection_candidates.pop_back();
            }

        // select best projection by marks to read
        ProjectionMatchContext * selected_candidate = nullptr;
        size_t min_sum_marks = std::numeric_limits<size_t>::max();

        auto normal_read_result = estimateReadMarks(part_group);

        // Add 1 to base sum_marks so that we prefer projections even when they have equal number of marks to read.
        // NOTE: It is not clear if we need it. E.g. projections do not support skip index for now.
        if (!normal_read_result->error())
            min_sum_marks = normal_read_result->marks() + 1;

        for (auto & candidate: projection_candidates)
        {
            estimateReadMarksForProjection(part_group, candidate);
            if (!candidate.read_analysis->error() && candidate.read_analysis->marks() < min_sum_marks)
            {
                selected_candidate = &candidate;
                min_sum_marks = candidate.read_analysis->marks();
            }
        }

        if (selected_candidate)
            execute_plan.push_back(selected_candidate->buildExecutePlanElement(std::move(part_group), select_query_info));
        else
            execute_plan.push_back(ExecutePlanElement(std::move(part_group), std::move(normal_read_result), std::move(part_group.parts)));
    }

    if (log->debug())
    {
        size_t total_parts = 0;
        size_t total_marks = 0;
        size_t normal_parts = 0;
        size_t projection_parts = 0;
        size_t marks_of_normal_parts = 0;
        size_t marks_of_projection_parts = 0;
        for (const auto & e: execute_plan)
        {
            size_t parts = e.part_group.partsNum();
            size_t marks = e.read_analysis->marks();

            total_parts += parts;
            total_marks += marks;
            if (e.projection_desc)
            {
                projection_parts += parts;
                marks_of_projection_parts += marks;
            }
            else
            {
                normal_parts += parts;
                marks_of_normal_parts += marks;
            }
        }
        String str = fmt::format(
            "data input pipeline with projection(summary): total {} parts, total {} marks, "
            "{} normal parts, {} projection parts, {} marks of normal parts, {} marks of projection parts",
            total_parts, total_marks, normal_parts, projection_parts, marks_of_normal_parts, marks_of_projection_parts);
        LOG_DEBUG(log, "{}", str);

        str = "data input pipeline with projection(detail): ";
        for (const auto & e: execute_plan)
        {
            str.push_back('\n');
            str.append(e.toString());
        }
        LOG_DEBUG(log, "{}", str);
    }

    return execute_plan;
}

bool TableScanExecutor::match(ProjectionMatchContext & candidate) const
{
    auto & projection_desc = candidate.projection_desc;

    if (!has_aggregate && projection_desc.type == ProjectionDescription::Type::Aggregate)
        return false;

    // quick filter by required columns, proceed to match if query required columns belong to
    // projection required columns.
    //
    // e.g., given
    //   query: select toYear(date), sum(amount) from t1 group by toYear(date);
    //   projection: select toYear(date), key, sum(amount) from t1 group by toYear(date), key;
    // it produce:
    //   query required columns: date, amount
    //   projection require columns: date, key, amount
    NameSet projection_required_columns {projection_desc.required_columns.begin(), projection_desc.required_columns.end()};
    for (const auto & query_column: query_required_columns)
        if (!projection_required_columns.contains(query_column) && !candidate.missing_columns.contains(query_column))
            return false;

    // translate query column based expression to projection column based expression,
    // this is implemented by below steps:
    //   1. flatten query column based expression to table column based expression,
    //      by query column lineage information
    //   2. translate table column based expression to projection column,
    //      by projection column translation information
    //   3. if there are table columns unable to translate, either it's a missing key,
    //      rewrite to a literal with the default value of the column type; or this
    //      projection can not serve this query
    //
    // e.g. given
    //   query plan: Aggregating(key: $3, function: sum($2)) -> Projection($3 := toYear($1) + 1) -> TableScan($1 := date, $2 := amount)
    //   projection: select toYear(date), key, sum(amount) from t1 group by toYear(date), key;
    // the procedure of rewriting grouping key(i.e. `$3`) is as below:
    //   1. flatten `$3` => toYear(`$1`) + 1 => toYear(`date`) + 1
    //   2. translate toYear(`date`) + 1 => `toYear(date)` + 1
    if (projection_desc.type == ProjectionDescription::Type::Aggregate)
    {
        // match & rewrite grouping keys
        for (const auto & aggregate_key: aggregate_keys)
        {
            auto rewritten_expr = rewriteExpr(aggregate_key.flatten_ast, candidate);
            if (!rewritten_expr)
                return false;
            candidate.rewritten_assignments.emplace_back(aggregate_key.name, rewritten_expr);
            candidate.rewritten_types.emplace(aggregate_key.name, column_types_before_agg.at(aggregate_key.name));
        }

        // match & rewrite aggregates
        for (const auto & aggregate_desc: aggregate_descs)
        {
            auto projection_agg_column_opt = candidate.column_translation.tryGetTranslation(aggregate_desc.flatten_ast);
            // TODO @wangtao: support if aggregate argument are missing columns
            if (!projection_agg_column_opt)
                return false;
            auto & projection_agg_column = *projection_agg_column_opt;
            candidate.rewritten_assignments.emplace_back(aggregate_desc.name, std::make_shared<ASTIdentifier>(projection_agg_column));
            candidate.rewritten_types.emplace(aggregate_desc.name, candidate.column_types.at(projection_agg_column));
            candidate.required_column_set.emplace(projection_agg_column);
        }

        // match & rewrite where
        if (flatten_filter)
        {
            auto rewritten_filter = rewriteExpr(flatten_filter, candidate);
            if (!rewritten_filter)
                return false;
            candidate.rewritten_filter = std::move(rewritten_filter);
        }

        return true;
    }
    else
    {
        // to prevent a query both use normal projection & aggregate projection
        if (has_aggregate)
            return false;

        return false;
    }
}

MergeTreeDataSelectAnalysisResultPtr TableScanExecutor::estimateReadMarks(const PartGroup & part_group) const
{
    return merge_tree_reader.estimateNumMarksToRead(part_group.parts,
                                                    select_query_info.prewhere_info,
                                                    query_required_columns,
                                                    storage_metadata,
                                                    storage_metadata,
                                                    select_query_info,
                                                    ActionDAGNodes{},
                                                    context,
                                                    context->getSettingsRef().max_threads,
                                                    max_added_blocks);
}

void TableScanExecutor::estimateReadMarksForProjection(const PartGroup & part_group, ProjectionMatchContext & candidate) const
{
    MergeTreeData::DataPartsVector projection_parts;

    for (const auto & part: part_group.parts)
    {
        const auto & projections = part->getProjectionParts();
        auto it = projections.find(candidate.projection_desc.name);
        if (it == projections.end())
            throw Exception(ErrorCodes::LOGICAL_ERROR, "projection not found in a part group, does part grouping perform correctly?");
        projection_parts.push_back(it->second);
    }

    candidate.read_analysis = merge_tree_reader.estimateNumMarksToRead(projection_parts,
                                                                       select_query_info.prewhere_info,
                                                                       candidate.requiredColumns(),
                                                                       storage_metadata,
                                                                       candidate.projection_desc.metadata,
                                                                       select_query_info,
                                                                       ActionDAGNodes{},
                                                                       context,
                                                                       context->getSettingsRef().max_threads,
                                                                       max_added_blocks);

}

void TableScanExecutor::prunePartsByIndex(MergeTreeData::DataPartsVector & parts) const
{
    if (parts.empty())
        return;

    auto part_values = MergeTreeDataSelectExecutor::filterPartsByVirtualColumns(
        merge_tree_data,
        parts,
        select_query_info.query,
        context);

    if (part_values && part_values->empty())
    {
        parts.clear();
        return;
    }

    ReadFromMergeTree::AnalysisResult result;
    MergeTreeDataSelectExecutor::filterPartsByPartition(
        parts,
        part_values,
        storage_metadata,
        merge_tree_data,
        select_query_info,
        context,
        max_added_blocks.get(),
        log,
        result.index_stats);
}

PartGroups TableScanExecutor::groupPartsBySchema(const MergeTreeData::DataPartsVector & parts)
{
    std::unordered_map<PartSchemaKey, MergeTreeData::DataPartsVector, PartSchemaKey::Hash> parts_by_schema;

    for (const auto & part: parts)
    {
        PartSchemaKey key(part);
        parts_by_schema[std::move(key)].emplace_back(part);
    }

    PartGroups groups;

    for (auto & it : parts_by_schema)
        groups.emplace_back(PartGroup{std::move(it.second)});

    return groups;
}

ASTPtr TableScanExecutor::rewriteExpr(ASTPtr expr, ProjectionMatchContext & candidate) const
{
    if (auto projection_column = candidate.column_translation.tryGetTranslation(expr))
    {
        candidate.required_column_set.emplace(*projection_column);
        return std::make_shared<ASTIdentifier>(*projection_column);
    }

    if (auto * col_ref = expr->as<ASTTableColumnReference>())
    {
        if (auto pair = candidate.missing_columns.tryGetByName(col_ref->column_name))
            return std::make_shared<ASTLiteral>(pair->type->getDefault());

        return nullptr;
    }
    else if (auto * func = expr->as<ASTFunction>(); func && func->arguments && !func->arguments->children.empty())
    {
        ASTs rewritten_arguments;

        for (const auto & argument: func->arguments->children)
        {
            rewritten_arguments.push_back(rewriteExpr(argument, candidate));
            if (rewritten_arguments.back() == nullptr)
                return nullptr;
        }

        return makeASTFunction(func->name, rewritten_arguments);
    }
    else
        return expr;
}

void TableScanStep::makeSetsForIndex(const ASTPtr & node, ContextPtr context, PreparedSets & prepared_sets) const
{
    auto settings = context->getSettingsRef();
    SizeLimits size_limits_for_set(settings.max_rows_in_set, settings.max_bytes_in_set, settings.set_overflow_mode);

    if (!node || !storage || !storage->supportsIndexForIn())
        return;

    for (auto & child : node->children)
    {
        /// Don't descend into lambda functions
        const auto * func = child->as<ASTFunction>();
        if (func && func->name == "lambda")
            continue;

        makeSetsForIndex(child, context, prepared_sets);
    }

    const auto * func = node->as<ASTFunction>();
    auto metadata_snapshot = storage->getInMemoryMetadataPtr();
    if (func && functionIsInOrGlobalInOperator(func->name))
    {
        const IAST & args = *func->arguments;
        const ASTPtr & left_in_operand = args.children.at(0);

        if (storage && storage->mayBenefitFromIndexForIn(left_in_operand, context, metadata_snapshot))
        {
            const ASTPtr & arg = args.children.at(1);
            if (arg->as<ASTSubquery>() || arg->as<ASTTableIdentifier>())
            {
                // if (settings.use_index_for_in_with_subqueries)
                //     tryMakeSetForIndexFromSubquery(arg, query_options);
            }
            else
            {
                auto input = storage->getInMemoryMetadataPtr()->getColumns().getAll();
                Names output;
                output.emplace_back(left_in_operand->getColumnName());
                auto temp_actions = createExpressionActions(context, input, output, left_in_operand);
                if (temp_actions->tryFindInIndex(left_in_operand->getColumnName()))
                {
                    makeExplicitSet(func, *temp_actions, true, context, size_limits_for_set, prepared_sets);
                }
            }
        }
    }
}

TableScanStep::TableScanStep(
    ContextPtr /*context*/,
    StoragePtr storage_,
    const NamesWithAliases & column_alias_,
    const SelectQueryInfo & query_info_,
    size_t max_block_size_,
    String alias_,
    PlanHints hints_,
    QueryPlanStepPtr aggregation_,
    QueryPlanStepPtr projection_,
    QueryPlanStepPtr filter_)
    : ISourceStep(DataStream{}, hints_)
    , storage(storage_)
    , storage_id(storage->getStorageID())
    , column_alias(column_alias_)
    , query_info(query_info_)
    , max_block_size(max_block_size_)
    , alias(alias_)
    , pushdown_aggregation(std::move(aggregation_))
    , pushdown_projection(std::move(projection_))
    , pushdown_filter(std::move(filter_))
{
    log = &Poco::Logger::get("TableScanStep");

    for (auto & item : column_alias)
    {
        column_names.emplace_back(item.first);
    }

    // order sensitive
    Names require_column_list = column_names;
    NameSet require_columns{column_names.begin(), column_names.end()};

    //    QueryPlan tmp_query_plan;
    //    storage->read(tmp_query_plan, require_column_list, storage->getInMemoryMetadataPtr(), query_info, context, processing_stage, max_block_size, max_streams, true);

    auto all_columns = storage->getInMemoryMetadataPtr()->getColumns().getAllPhysical();
    auto virtual_col = storage->getVirtuals();
    all_columns.insert(all_columns.end(), virtual_col.begin(), virtual_col.end());
    for (const auto & item : all_columns)
        if (std::find(require_column_list.begin(), require_column_list.end(), item.name) == require_column_list.end()
            && require_columns.contains(item.name))
            require_column_list.push_back(item.name);

    auto header = storage->getInMemoryMetadataPtr()->getSampleBlockForColumns(require_column_list, storage->getVirtuals());

    // init query_info.syntax_analyzer
    auto tree_rewriter_result
            = std::make_shared<TreeRewriterResult>(header.getNamesAndTypesList(), storage, storage->getInMemoryMetadataPtr());
    tree_rewriter_result->required_source_columns = header.getNamesAndTypesList();
    tree_rewriter_result->analyzed_join = std::make_shared<TableJoin>();
    query_info.syntax_analyzer_result = tree_rewriter_result;

    NameToNameMap name_to_name_map;
    for (auto & item : column_alias)
    {
        name_to_name_map[item.first] = item.second;
    }

    const auto select_expression_list = query_info.query->as<ASTSelectQuery>()->select();
    select_expression_list->children.clear();

    for (const auto & item : header)
    {
        select_expression_list->children.emplace_back(std::make_shared<ASTIdentifier>(item.name));
        if (name_to_name_map.contains(item.name))
        {
            table_output_stream.header.insert(ColumnWithTypeAndName{item.type, name_to_name_map[item.name]});
        }
    }

    formatOutputStream();
}

TableScanStep::TableScanStep(
    ContextPtr context,
    StorageID storage_id_,
    const NamesWithAliases & column_alias_,
    const SelectQueryInfo & query_info_,
    size_t max_block_size_,
    String alias_,
    PlanHints hints_,
    QueryPlanStepPtr aggregation_,
    QueryPlanStepPtr projection_,
    QueryPlanStepPtr filter_)
    : TableScanStep(
        context,
        DatabaseCatalog::instance().getTable(storage_id_, context),
        column_alias_,
        query_info_,
        max_block_size_,
        alias_,
        hints_,
        aggregation_,
        projection_,
        filter_)
{
}

void TableScanStep::formatOutputStream()
{
    if (pushdown_aggregation != nullptr)
        *output_stream = pushdown_aggregation->getOutputStream();
    else if (pushdown_projection != nullptr)
        *output_stream = pushdown_projection->getOutputStream();
    else if (pushdown_filter != nullptr)
        *output_stream = pushdown_filter->getOutputStream();
    else
        *output_stream = table_output_stream;
}

void TableScanStep::optimizeWhereIntoPrewhre(ContextPtr context)
{
    auto & query = *query_info.query->as<ASTSelectQuery>();
    if (storage && query.where() && !query.prewhere())
    {
        /// PREWHERE optimization: transfer some condition from WHERE to PREWHERE if enabled and viable
        if (const auto & column_sizes = storage->getColumnSizes(); !column_sizes.empty())
        {
            /// Extract column compressed sizes.
            std::unordered_map<std::string, UInt64> column_compressed_sizes;
            for (const auto & [name, sizes] : column_sizes)
                column_compressed_sizes[name] = sizes.data_compressed;

            MergeTreeWhereOptimizer{
                query_info, context, std::move(column_compressed_sizes), storage->getInMemoryMetadataPtr(), column_names, log};
        }
    }
}

SelectQueryInfo TableScanStep::fillQueryInfo(ContextPtr context)
{
    auto * query = query_info.query->as<ASTSelectQuery>();
    SelectQueryInfo copy_query_info = query_info;
    makeSetsForIndex(query->where(), context, copy_query_info.sets);
    makeSetsForIndex(query->prewhere(), context, copy_query_info.sets);
    return copy_query_info;
}

ASTs cloneChildrenReplacement(ASTs ast_children_replacement)
{
    ASTs cloned_replacement;
    cloned_replacement.reserve(ast_children_replacement.size());
    for (const auto & ast_child : ast_children_replacement)
    {
        cloned_replacement.push_back(ast_child->clone());
    }
    return cloned_replacement;
}

void TableScanStep::rewriteInForBucketTable(ContextPtr context) const
{
    const auto * cloud_merge_tree = dynamic_cast<StorageCloudMergeTree *>(storage.get());
    if (!cloud_merge_tree)
        return;

    auto metadata_snapshot = cloud_merge_tree->getInMemoryMetadataPtr();
    const bool isBucketTableAndNeedOptimise = context->getSettingsRef().optimize_skip_unused_shards && cloud_merge_tree->isBucketTable()
        && metadata_snapshot->getColumnsForClusterByKey().size() == 1 && !cloud_merge_tree->getRequiredBucketNumbers().empty();
    if (!isBucketTableAndNeedOptimise)
        return;

    auto query = query_info.query->as<ASTSelectQuery>();

    // NOTE: Have try-catch for every rewrite_in in order to find the WHERE clause that caused the error
    RewriteInQueryVisitor::Data data;
    LOG_TRACE(log, "Before rewriteInForBucketTable:\n: {}", query->dumpTree());
    if (query->where())
    {
        auto ast_children_replacement = cloud_merge_tree->convertBucketNumbersToAstLiterals(query->where(), context);
        if (!ast_children_replacement.empty())
        {
            data.ast_children_replacement = cloneChildrenReplacement(ast_children_replacement);
            ASTPtr & where = query->refWhere();
            RewriteInQueryVisitor(data).visit(where);
        }
    }
    if (query->prewhere())
    {
        auto ast_children_replacement = cloud_merge_tree->convertBucketNumbersToAstLiterals(query->prewhere(), context);
        if (!ast_children_replacement.empty())
        {
            data.ast_children_replacement = cloneChildrenReplacement(ast_children_replacement);
            ASTPtr & pre_where = query->refPrewhere();
            RewriteInQueryVisitor(data).visit(pre_where);
        }
    }
    LOG_TRACE(log, "After rewriteInForBucketTable:\n: {}", query->dumpTree());
}

void TableScanStep::initializePipeline(QueryPipelineBuilder & pipeline, const BuildQueryPipelineSettings & build_context)
{
    Stopwatch stage_watch, total_watch;
    total_watch.start();
    storage = DatabaseCatalog::instance().getTable(storage_id, build_context.context);

    if (auto * filter_step = getPushdownFilterCast())
    {
        // set where SelectQueryInfo
        auto conjuncts =  PredicateUtils::extractConjuncts(filter_step->getFilter());
        PushQueryInfoFilterIntoTableScan::pushQueryInfoFilter(*this, conjuncts, build_context.context);

        // remove storage filter
        auto conjuncts_wo_storage_filter = PushQueryInfoFilterIntoTableScan::removeStorageFilter(conjuncts);
        if (conjuncts_wo_storage_filter.size() != conjuncts.size())
        {
            auto new_predicate = PredicateUtils::combineConjuncts(conjuncts_wo_storage_filter);
            if (PredicateUtils::isTruePredicate(new_predicate))
                pushdown_filter = nullptr;
            else
                filter_step->setFilter(std::move(new_predicate));
        }
    }

    bool use_optimizer_projection_selection = build_context.context->getSettingsRef().optimizer_projection_support && storage.get()
        && (!pushdown_projection || !getPushdownProjectionCast()->hasDynamicFilters());

    rewriteInForBucketTable(build_context.context);
    auto * query = query_info.query->as<ASTSelectQuery>();
    if (auto where = query->getWhere())
        // query->setExpression(ASTSelectQuery::Expression::WHERE, rewriteDynamicFilter(where, pipeline, build_context));
        query->setExpression(ASTSelectQuery::Expression::WHERE, std::move(where));
    if (auto prewhere = query->getPrewhere())
        // query->setExpression(ASTSelectQuery::Expression::PREWHERE, rewriteDynamicFilter(prewhere, pipeline, build_context));
        query->setExpression(ASTSelectQuery::Expression::PREWHERE, std::move(prewhere));

    // todo support _shard_num rewrite
    // if ()
    // {
    //     VirtualColumnUtils::rewriteEntityInAst(query_info.query, "_shard_num", shard_info.shard_num, "toUInt32");
    // }

    /**
     * reconstuct query level info based on query
     */
    SelectQueryOptions options;
    options.distributedStages();
    if (use_optimizer_projection_selection)
        options.ignoreProjections();

    stage_watch.start();
    auto interpreter = std::make_shared<InterpreterSelectQuery>(query_info.query, build_context.context, options);
    interpreter->execute();
    query_info = interpreter->getQueryInfo();
    query_info = fillQueryInfo(build_context.context);
    LOG_DEBUG(log, "init pipeline stage run time: make up query info, {} ms", stage_watch.elapsedMillisecondsAsDouble());

    // always do filter underneath, as WHERE filter won't reuse PREWHERE result in optimizer mode
    if (query_info.prewhere_info)
        query_info.prewhere_info->need_filter = true;

    ExecutePlan execute_plan;

    stage_watch.restart();
    if (use_optimizer_projection_selection)
        execute_plan = TableScanExecutor(*this, build_context.context).buildExecutePlan();
    LOG_DEBUG(log, "init pipeline stage run time: projection match, {} ms", stage_watch.elapsedMillisecondsAsDouble());

    size_t max_streams = build_context.context->getSettingsRef().max_threads;
    if (max_block_size < build_context.context->getSettingsRef().max_block_size)
        max_streams = 1; // single block single stream.

    if (max_streams > 1 && !storage->isRemote())
        max_streams *= build_context.context->getSettingsRef().max_streams_to_max_threads_ratio;

    if (execute_plan.empty())
    {
        auto pipe = storage->read(
            interpreter->getRequiredColumns(), storage->getInMemoryMetadataPtr(), query_info, build_context.context, QueryProcessingStage::Enum::FetchColumns, max_block_size, max_streams);

        QueryPlanStepPtr step;
        if (pipe.empty())
        {
            auto header = storage->getInMemoryMetadataPtr()->getSampleBlockForColumns(column_names, storage->getVirtuals(), storage_id);
            auto null_pipe = InterpreterSelectQuery::generateNullSourcePipe(header, query_info);
            auto read_from_pipe = std::make_shared<ReadFromPreparedSource>(std::move(null_pipe));
            read_from_pipe->setStepDescription("Read from NullSource");
            step = read_from_pipe;
        }
        else
            step = std::make_shared<ReadFromStorageStep>(std::move(pipe), step_description);


        if (auto * source = dynamic_cast<ISourceStep *>(step.get()))
            source->initializePipeline(pipeline, build_context);

        aliasColumns(pipeline, build_context);

        //    rewriteInForBucketTable(build_context.context);
        //
        //    ASTPtr new_prewhere = rewriteDynamicFilter(prewhere, build_context);
        //
        //    if (auto query_prewhere = new_query_info.query->as<ASTSelectQuery>()->getPrewhere())
        //        new_query_info.query->as<ASTSelectQuery>()->refPrewhere() = rewriteDynamicFilter(query_prewhere, build_context);
        //
        //    if (auto query_where = new_query_info.query->as<ASTSelectQuery>()->getWhere())
        //        new_query_info.query->as<ASTSelectQuery>()->refWhere() = rewriteDynamicFilter(query_where, build_context);
        //
        //    std::tie(new_query_info, new_prewhere) = fillPrewhereInfo(build_context.context, new_query_info, new_prewhere);
        //
        //    // order sensitive
        //    Names require_column_list = column_names;
        //    NameSet require_columns{column_names.begin(), column_names.end()};
        //
        //    if (new_query_info.prewhere_info)
        //    {
        //        if (new_query_info.prewhere_info->alias_actions)
        //        {
        //            auto alias_require = new_query_info.prewhere_info->alias_actions->getRequiredColumns();
        //            require_columns.insert(alias_require.begin(), alias_require.end());
        //        }
        //        if (new_query_info.prewhere_info->prewhere_actions)
        //        {
        //            auto prewhere_require = new_query_info.prewhere_info->prewhere_actions->getRequiredColumns();
        //            require_columns.insert(prewhere_require.begin(), prewhere_require.end());
        //        }
        //    }
        //
        //    auto all_columns = storage->getColumns().getAllPhysical();
        //    auto virtual_col = storage->getVirtuals();
        //    all_columns.insert(all_columns.end(), virtual_col.begin(), virtual_col.end());
        //    for (const auto & item : all_columns)
        //        if (std::find(require_column_list.begin(), require_column_list.end(), item.name) == require_column_list.end()
        //            && require_columns.contains(item.name))
        //            require_column_list.push_back(item.name);
        //
        //
        //    NamesAndTypes new_header;
        //    {
        //        auto header = storage->getSampleBlockForColumns(require_column_list);
        //
        //        if (new_query_info.prewhere_info)
        //        {
        //            if (new_query_info.prewhere_info->alias_actions)
        //                new_query_info.prewhere_info->alias_actions->execute(header);
        //
        //            new_query_info.prewhere_info->prewhere_actions->execute(header);
        //            if (new_query_info.prewhere_info->remove_prewhere_column)
        //                header.erase(new_query_info.prewhere_info->prewhere_column_name);
        //
        //            if (new_query_info.prewhere_info->remove_columns_actions)
        //                new_query_info.prewhere_info->remove_columns_actions->execute(header);
        //
        //            auto check_actions = [](const ExpressionActionsPtr & actions) {
        //                if (actions)
        //                    for (const auto & action : actions->getActions())
        //                        if (action.type == ExpressionAction::Type::JOIN || action.type == ExpressionAction::Type::ARRAY_JOIN)
        //                            throw Exception("PREWHERE cannot contain ARRAY JOIN or JOIN action", ErrorCodes::ILLEGAL_PREWHERE);
        //            };
        //
        //            check_actions(new_query_info.prewhere_info->prewhere_actions);
        //            check_actions(new_query_info.prewhere_info->alias_actions);
        //            check_actions(new_query_info.prewhere_info->remove_columns_actions);
        //        }
        //
        //        NameToNameMap name_to_name_map;
        //        for (auto & item : column_alias)
        //        {
        //            name_to_name_map[item.first] = item.second;
        //        }
        //        for (const auto & item : header)
        //        {
        //            if (name_to_name_map.contains(item.name))
        //            {
        //                new_header.emplace_back(name_to_name_map[item.name], item.type);
        //            }
        //            else
        //            {
        //                // additional columns may be generated. e.g. alias columns in PREWHERE clause:
        //                // DDL: CREATE TABLE test.prewhere_alias(`a` Int32, `b` Int32, `c` ALIAS a + b)
        //                // query: select a, (c + toInt32(1)) * 2 from test.prewhere_alias prewhere (c + toInt32(1)) * 2 = 6;
        //                new_header.emplace_back(item.name, item.type);
        //            }
        //        }
        //    }
        //
        //    size_t max_streams = build_context.max_streams;
        //    if (max_block_size < build_context.context.getSettingsRef().max_block_size)
        //        max_streams = 1; // single block single stream.
        //
        //    if (max_streams > 1 && !storage->isRemote())
        //        max_streams *= build_context.context.getSettingsRef().max_streams_to_max_threads_ratio;
        //
        //    if (dynamic_cast<StorageCloudMergeTree *>(storage.get()))
        //        build_context.context.setNameNode();
        //
        //    // for arraySetCheck function, it requires source_columns.
        //    if (new_query_info.syntax_analyzer_result == nullptr)
        //    {
        //        SyntaxAnalyzerResult result;
        //        result.source_columns = all_columns;
        //        new_query_info.syntax_analyzer_result = std::make_shared<const SyntaxAnalyzerResult>(result);
        //    }
        //
        //    auto streams = storage->read(require_column_list, new_query_info, build_context.context, processing_stage, max_block_size, max_streams);
        //    if (!original_table.empty())
        //    {
        //        auto mem_buffer_table = build_context.context.tryGetTable(database, original_table);
        //        if (mem_buffer_table)
        //        {
        //            auto mem_streams = mem_buffer_table->read(
        //                require_column_list, new_query_info, build_context.context, processing_stage, max_block_size, max_streams);
        //            streams.insert(streams.end(), mem_streams.begin(), mem_streams.end());
        //            LOG_DEBUG(log, "create memory buffer streams for " << database << '.' << original_table << " size " << mem_streams.size());
        //        }
        //    }
        //
        //    if (streams.empty())
        //    {
        //        Block header;
        //        for (const auto & item : new_header)
        //            header.insert(ColumnWithTypeAndName{item.type->createColumn(), item.type, item.name});
        //        pipeline.streams.emplace_back(std::make_shared<NullBlockInputStream>(header));
        //        return;
        //    }
        //
        //    pipeline.streams.insert(pipeline.streams.end(), streams.begin(), streams.end());
        //
        //

        // setQuotaAndLimits(pipeline, options, build_context);

        //
        //    NameToNameMap name_to_name_map;
        //    for (auto & item : column_alias)
        //    {
        //        name_to_name_map[item.second] = item.first;
        //    }
        //    NamesWithAliases outputs;
        //    ASTPtr select = std::make_shared<ASTExpressionList>();
        //    for (const auto & column : new_header)
        //    {
        //        select->children.emplace_back(std::make_shared<ASTIdentifier>(column.name));
        //        if (name_to_name_map.contains(column.name))
        //        {
        //            outputs.emplace_back(name_to_name_map[column.name], column.name);
        //        }
        //    }
        //    auto action
        //        = createExpressionActions(build_context.context, pipeline.firstStream()->getHeader().getNamesAndTypesList(), outputs, select);
        //    pipeline.transform([&](auto & stream) { stream = std::make_shared<ExpressionBlockInputStream>(stream, action); });

        if (auto * filter_step = getPushdownFilterCast())
            filter_step->transformPipeline(pipeline, build_context);
        if (auto * projection_step = getPushdownProjectionCast())
            projection_step->transformPipeline(pipeline, build_context);
        if (auto * aggregate_step = getPushdownAggregationCast())
            aggregate_step->transformPipeline(pipeline, build_context);

        LOG_DEBUG(log, "init pipeline total run time: {} ms", total_watch.elapsedMillisecondsAsDouble());
        return;
    }

    // auto & merge_tree_data = dynamic_cast<MergeTreeMetaBase &>(*storage);
    MergeTreeDataSelectExecutor merge_tree_reader{*storage};
    auto metadata_snapshot = storage->getInMemoryMetadataPtr();
    auto context = build_context.context;
    Pipes pipes;
    std::vector<size_t> plan_element_ids; // num of pipes may be smaller than num of plan elements since MergeTreeDataSelectExecutor
                                          // can infer an empty result for a part group. hence we record a mapping of pipe->plan element
    size_t total_output_ports = 0;

    for (size_t plan_element_id = 0; plan_element_id < execute_plan.size(); ++plan_element_id)
    {
        const auto & plan_element = execute_plan[plan_element_id];
        bool use_projection = plan_element.projection_desc != nullptr;
        QueryPipelinePtr sub_pipeline;

        if (plan_element.read_analysis->marks() == 0)
            continue;

        if (use_projection)
        {
            SelectQueryInfo projection_query_info = query_info;
            if (projection_query_info.prewhere_info)
            {
                projection_query_info.prewhere_info = std::make_shared<PrewhereInfo>(*projection_query_info.prewhere_info);
                projection_query_info.prewhere_info->prewhere_actions = plan_element.prewhere_actions;
            }
            MergeTreeData::DeleteBitmapGetter null_getter = [](auto & /*part*/) { return nullptr; };
            auto read_plan = merge_tree_reader.readFromParts(
                {},
                null_getter,
                plan_element.projection_required_columns,
                metadata_snapshot,
                plan_element.projection_desc->metadata,
                projection_query_info,
                context,
                max_block_size,
                max_streams,
                nullptr,
                plan_element.read_analysis);

            sub_pipeline = read_plan->buildQueryPipeline(QueryPlanOptimizationSettings::fromContext(context),
                                                         BuildQueryPipelineSettings::fromContext(context));

            if (plan_element.rewritten_filter_step)
                dynamic_cast<FilterStep &>(*plan_element.rewritten_filter_step).transformPipeline(*sub_pipeline, build_context);

            if (plan_element.rewritten_projection_step)
                dynamic_cast<ProjectionStep &>(*plan_element.rewritten_projection_step).transformPipeline(*sub_pipeline, build_context);
        }
        else
        {
            // MergeTreeMetaBase::DeleteBitmapGetter delete_bitmap_getter;
            // if (metadata_snapshot->hasUniqueKey())
            // {
            //     /// get a consistent snapshot of delete bitmaps for query,
            //     /// otherwise concurrent upserts that modify part's delete bitmap will cause incorrect query result
            //     auto delete_bitmap_snapshot = merge_tree_data.getLatestDeleteSnapshot(plan_element.parts);
            //     /// move delete_bitmap_snapshot into the closure because delete_bitmap_getter will be used after this function returns
            //     delete_bitmap_getter = [snapshot = std::move(delete_bitmap_snapshot)](const auto & part) -> ImmutableDeleteBitmapPtr
            //     {
            //         if (auto it = snapshot.find(part); it != snapshot.end())
            //             return it->second;
            //         throw Exception(ErrorCodes::LOGICAL_ERROR, "Not found delete bitmap for part " + part->name);
            //     };
            // }
            // else
            // {
            //     delete_bitmap_getter = [](const auto & part) { return part->getDeleteBitmap(); };
            // }
            MergeTreeData::DeleteBitmapGetter null_getter = [](auto & /*part*/) { return nullptr; };
            auto read_plan = merge_tree_reader.readFromParts(
                {},
                null_getter,
                column_names,
                metadata_snapshot,
                metadata_snapshot,
                query_info,
                context,
                max_block_size,
                max_streams,
                nullptr,
                plan_element.read_analysis);

            sub_pipeline = read_plan->buildQueryPipeline(QueryPlanOptimizationSettings::fromContext(context),
                                                         BuildQueryPipelineSettings::fromContext(context));

            aliasColumns(*sub_pipeline, build_context);

            if (pushdown_filter)
                getPushdownFilterCast()->transformPipeline(*sub_pipeline, build_context);

            if (pushdown_projection)
                getPushdownProjectionCast()->transformPipeline(*sub_pipeline, build_context);
        }

        auto pipe = QueryPipeline::getPipe(std::move(*sub_pipeline));
        total_output_ports += pipe.numOutputPorts();
        pipes.emplace_back(std::move(pipe));
        plan_element_ids.emplace_back(plan_element_id);
    }

    if (pushdown_aggregation)
    {
        const auto & settings = context->getSettingsRef();
        const auto & aggregate_step = *getPushdownAggregationCast();
        const auto & aggregate_param = aggregate_step.getParams();
        auto many_data = std::make_shared<ManyAggregatedData>(total_output_ports);
        size_t counter = 0;
        AggregatorListPtr aggregator_list_ptr = std::make_shared<AggregatorList>();

        for (size_t pipe_id = 0; pipe_id < pipes.size(); ++pipe_id)
        {
            auto & pipe = pipes[pipe_id];
            const auto & plan_element = execute_plan[plan_element_ids.at(pipe_id)];
            bool use_projection = plan_element.projection_desc != nullptr;

            const auto & header_before_aggregation = pipe.getHeader();

            ColumnNumbers keys;
            for (const auto & key : aggregate_step.getKeys())
                keys.push_back(header_before_aggregation.getPositionByName(key));

            AggregateDescriptions aggregates = aggregate_step.getAggregates();
            if (!use_projection)
            {
                for (auto & descr : aggregates)
                    if (descr.arguments.empty())
                        for (const auto & name : descr.argument_names)
                            descr.arguments.push_back(header_before_aggregation.getPositionByName(name));
            }

            AggregatingTransformParamsPtr transform_params;

            if (use_projection)
            {
                Aggregator::Params params(
                    header_before_aggregation,
                    keys,
                    aggregates,
                    aggregate_param.overflow_row,
                    settings.max_rows_to_group_by,
                    settings.group_by_overflow_mode,
                    settings.group_by_two_level_threshold,
                    settings.group_by_two_level_threshold_bytes,
                    settings.max_bytes_before_external_group_by,
                    settings.empty_result_for_aggregation_by_empty_set,
                    context->getTemporaryVolume(),
                    settings.max_threads,
                    settings.min_free_disk_space_for_temporary_data,
                    settings.compile_expressions,
                    settings.min_count_to_compile_aggregate_expression,
                    header_before_aggregation); // The source header is also an intermediate header

                transform_params = std::make_shared<AggregatingTransformParams>(std::move(params), aggregator_list_ptr, aggregate_step.isFinal());

                /// This part is hacky.
                /// We want AggregatingTransform to work with aggregate states instead of normal columns.
                /// It is almost the same, just instead of adding new data to aggregation state we merge it with existing.
                ///
                /// It is needed because data in projection:
                /// * is not merged completely (we may have states with the same key in different parts)
                /// * is not split into buckets (so if we just use MergingAggregated, it will use single thread)
                transform_params->only_merge = true;
            }
            else
            {
                Aggregator::Params params(
                    header_before_aggregation,
                    keys,
                    aggregates,
                    aggregate_param.overflow_row,
                    settings.max_rows_to_group_by,
                    settings.group_by_overflow_mode,
                    settings.group_by_two_level_threshold,
                    settings.group_by_two_level_threshold_bytes,
                    settings.max_bytes_before_external_group_by,
                    settings.empty_result_for_aggregation_by_empty_set,
                    context->getTemporaryVolume(),
                    settings.max_threads,
                    settings.min_free_disk_space_for_temporary_data,
                    settings.compile_aggregate_expressions,
                    settings.min_count_to_compile_aggregate_expression);

                transform_params = std::make_shared<AggregatingTransformParams>(std::move(params), aggregator_list_ptr, aggregate_step.isFinal());
            }

            pipe.resize(pipe.numOutputPorts(), true, true);

            auto merge_threads = max_streams;
            auto temporary_data_merge_threads = settings.aggregation_memory_efficient_merge_threads
                ? static_cast<size_t>(settings.aggregation_memory_efficient_merge_threads)
                : static_cast<size_t>(settings.max_threads);

            pipe.addSimpleTransform([&](const Block & header)
            {
                return std::make_shared<AggregatingTransform>(
                    header, transform_params, many_data, counter++, merge_threads, temporary_data_merge_threads);
            });
        }
    }

    Pipe pipe;

    if (!pipes.empty())
    {
        pipe = Pipe::unitePipes(std::move(pipes));
        pipe.resize(1);
    }
    else
    {
        pipe = Pipe(std::make_shared<NullSource>(output_stream->header));
    }

    auto step = std::make_unique<ReadFromStorageStep>(std::move(pipe), "ExecutePlanStep");
    step->initializePipeline(pipeline, build_context);

    // setQuotaAndLimits(pipeline, options, build_context);

    std::ostringstream step_desc;
    bool first = true;
    for (auto & plan_element: execute_plan)
    {
        if (first)
            first = false;
        else
            step_desc << ", ";

        if (plan_element.projection_desc)
            step_desc << plan_element.part_group.partsNum() << " parts from projection `" << plan_element.projection_desc->name << "`";
        else
            step_desc << plan_element.part_group.partsNum() << " parts from raw data";
    }
    setStepDescription(step_desc.str());

    LOG_DEBUG(log, "init pipeline total run time: {} ms, table scan descriptiion: {}", total_watch.elapsedMillisecondsAsDouble(), step_desc.str());
}

void TableScanStep::serialize(WriteBuffer &) const
{
    throw Exception("TableScanStep::serialize is not implemented", ErrorCodes::NOT_IMPLEMENTED);
    // writeBinary(step_description, buffer);
    // serializeBlock(output_stream->header, buffer);

    // storage_id.serialize(buffer);
    // query_info.serialize(buffer);
    // serializeStrings(column_names, buffer);
    // writeBinary(column_alias, buffer);
    // writeBinary(max_block_size, buffer);

    // writeBinary(pushdown_aggregation != nullptr, buffer);
    // if (pushdown_aggregation != nullptr)
    //     serializePlanStep(pushdown_aggregation, buffer);

    // writeBinary(pushdown_projection != nullptr, buffer);
    // if (pushdown_projection != nullptr)
    //     serializePlanStep(pushdown_projection, buffer);

    // writeBinary(pushdown_filter != nullptr, buffer);
    // if (pushdown_filter != nullptr)
    //     serializePlanStep(pushdown_filter, buffer);
}

QueryPlanStepPtr TableScanStep::deserialize(ReadBuffer &, ContextPtr)
{
    throw Exception("TableScanStep::deserialize is not implemented", ErrorCodes::NOT_IMPLEMENTED);
    // String step_description;
    // SelectQueryInfo query_info;

    // readBinary(step_description, buffer);
    // auto header = deserializeBlock(buffer);
    // StorageID storage_id = StorageID::deserialize(buffer, context);
    // query_info.deserialize(buffer);

    // size_t max_block_size;

    // Names column_names = deserializeStrings(buffer);
    // NamesWithAliases columns;
    // readBinary(columns, buffer);
    // readBinary(max_block_size, buffer);

    // bool has_aggregation;
    // readBinary(has_aggregation, buffer);
    // QueryPlanStepPtr aggregation;
    // if (has_aggregation)
    //     aggregation = deserializePlanStep(buffer, context);

    // bool has_projection;
    // readBinary(has_projection, buffer);
    // QueryPlanStepPtr projection;
    // if (has_projection)
    //     projection = deserializePlanStep(buffer, context);

    // bool has_filter;
    // readBinary(has_filter, buffer);
    // QueryPlanStepPtr filter;
    // if (has_filter)
    //     filter = deserializePlanStep(buffer, context);

    // return std::make_unique<TableScanStep>(context, storage_id, columns, query_info, max_block_size,
    //                                        String{}/*alias*/, PlanHints{}, aggregation, projection, filter);
}

std::shared_ptr<IQueryPlanStep> TableScanStep::copy(ContextPtr /*context*/) const
{
    SelectQueryInfo copy_query_info = query_info; // fixme@kaixi: deep copy here
    copy_query_info.query = query_info.query->clone();
    auto new_prewhere = copy_query_info.query->as<ASTSelectQuery &>().prewhere();

    return std::make_unique<TableScanStep>(
        output_stream.value(),
        storage,
        storage_id,
        original_table,
        column_names,
        column_alias,
        copy_query_info,
        max_block_size,
        alias,
        hints,
        pushdown_aggregation,
        pushdown_projection,
        pushdown_filter,
        table_output_stream);
}

std::shared_ptr<IStorage> TableScanStep::getStorage() const
{
    return storage;
}

void TableScanStep::allocate(ContextPtr)
{
    return;
    // original_table = storage_id.table_name;
    // auto * cnch_merge_tree = dynamic_cast<StorageCnchMergeTree *>(storage.get());
    // auto * cnch_hive = dynamic_cast<StorageCnchHive *>(storage.get());

    // if (!cnch_merge_tree && !cnch_hive)
    //     return;

    // if (cnch_merge_tree)
    // {
    //     storage_id.database_name = cnch_merge_tree->getDatabaseName();
    //     auto prepare_res = cnch_merge_tree->prepareReadContext(column_names, cnch_merge_tree->getInMemoryMetadataPtr(),query_info, context);
    //     storage_id.table_name = prepare_res.local_table_name;
    // }
    // else if (cnch_hive)
    // {
    //     size_t max_streams = context->getSettingsRef().max_threads;
    //     if (max_block_size < context->getSettingsRef().max_block_size)
    //         max_streams = 1; // single block single stream.

    //     if (max_streams > 1 && !storage->isRemote())
    //         max_streams *= context->getSettingsRef().max_streams_to_max_threads_ratio;

    //     storage_id.database_name = cnch_hive->getDatabaseName();
    //     auto prepare_res = cnch_hive->prepareReadContext(column_names, cnch_hive->getInMemoryMetadataPtr(),query_info, context, max_streams);
    //     storage_id.table_name = prepare_res.local_table_name;
    // }
    // storage_id.uuid = UUIDHelpers::Nil;
    // if (query_info.query)
    // {
    //     query_info = fillQueryInfo(context);
    //     /// trigger preallocate tables
    //     // if (!cnch->isOnDemandMode())
    //     // cnch->read(column_names, query_info, context, processed_stage, max_block_size, 1);
    //     // cnch->genPlanSegmentQueryAndAllocate(column_names, query_info, context);
    //     auto db_table = getDatabaseAndTable(query_info.query->as<ASTSelectQuery &>(), 0);
    //     if (!db_table->table.empty())
    //     {
    //         if (db_table->table != storage_id.table_name)
    //         {
    //             if (query_info.query)
    //             {
    //                 ASTSelectQuery & select_query = query_info.query->as<ASTSelectQuery &>();
    //                 select_query.replaceDatabaseAndTable(storage_id.database_name, storage_id.table_name);
    //             }
    //         }
    //     }
    // }
}

bool TableScanStep::setQueryInfoFilter(const std::vector<ConstASTPtr> & filters) const
{
    auto filter = PredicateUtils::combineConjuncts(filters);
    if (PredicateUtils::isTruePredicate(filter))
        return false;

    auto * query = query_info.query->as<ASTSelectQuery>();
    auto query_filter = query->getWhere();
    if (!query_filter)
    {
        query->setExpression(ASTSelectQuery::Expression::WHERE, std::move(filter));
        return true;
    }

    auto query_filters = PredicateUtils::extractConjuncts(query_filter);
    size_t original_filters_counts = query_filters.size();
    query_filters.insert(query_filters.end(), filters.begin(), filters.end());
    auto combine_filter = PredicateUtils::combineConjuncts(query_filters);

    if (PredicateUtils::extractConjuncts(combine_filter).size() == original_filters_counts)
        return false;

    query->setExpression(ASTSelectQuery::Expression::WHERE, std::move(combine_filter));
    return true;
}

bool TableScanStep::hasQueryInfoFilter() const
{
    auto * query = query_info.query->as<ASTSelectQuery>();
    return query->where().get();
}

bool TableScanStep::hasLimit() const
{
    auto * query = query_info.query->as<ASTSelectQuery>();
    return query->limitLength().get();
}

static UInt64 getUIntValue(const ASTPtr & node, const ContextPtr & context)
{
    const auto & [field, type] = evaluateConstantExpression(node, context);

    if (!isNativeNumber(type))
        throw Exception(
            "Illegal type " + type->getName() + " of LIMIT expression, must be numeric type", ErrorCodes::INVALID_LIMIT_EXPRESSION);

    Field converted = convertFieldToType(field, DataTypeUInt64());
    if (converted.isNull())
        throw Exception(
            "The value " + applyVisitor(FieldVisitorToString(), field) + " of LIMIT expression is not representable as UInt64",
            ErrorCodes::INVALID_LIMIT_EXPRESSION);

    return converted.safeGet<UInt64>();
}

bool TableScanStep::setLimit(size_t limit, const ContextMutablePtr & context)
{
    auto & query = *query_info.query->as<ASTSelectQuery>();
    auto limit_length = query.getLimitLength();
    if (limit_length)
    {
        if (getUIntValue(limit_length, context) <= limit)
            return false;
    }
    query.setExpression(ASTSelectQuery::Expression::LIMIT_LENGTH, std::make_shared<ASTLiteral>(limit));

    if (!query.distinct && !query.prewhere() && !query.where() && !query.groupBy() && !query.having() && !query.orderBy()
        && !query.limitBy() && query.limitLength() && limit < max_block_size)
    {
        max_block_size = std::max(static_cast<size_t>(1), limit);
    }

    return true;
}

size_t TableScanStep::getMaxBlockSize() const
{
    return max_block_size;
}

// ASTPtr
// TableScanStep::rewriteDynamicFilter(const ASTPtr & filter, QueryPipeline & pipeline, const BuildQueryPipelineSettings & build_context)
// {
//     if (!filter)
//         return nullptr;

//     auto filters = DynamicFilters::extractDynamicFilters(filter);
//     if (filters.first.empty())
//         return filter;

//     std::vector<ConstASTPtr> predicates = std::move(filters.second);
//     for (auto & dynamic_filter : filters.first)
//     {
//         auto description = DynamicFilters::extractDescription(dynamic_filter).value();
//         pipeline.addRuntimeFilterHolder(RuntimeFilterHolder{
//             build_context.distributed_settings.query_id, build_context.distributed_settings.plan_segment_id, description.id});

//         if (description.type == DynamicFilterType::Range)
//         {
//             auto dynamic_filters = DynamicFilters::createDynamicFilterRuntime(
//                 description,
//                 build_context.context->getInitialQueryId(),
//                 build_context.distributed_settings.plan_segment_id,
//                 build_context.context->getSettingsRef().wait_runtime_filter_timeout,
//                 RuntimeFilterManager::getInstance(),
//                 "TableScan");
//             predicates.insert(predicates.end(), dynamic_filters.begin(), dynamic_filters.end());
//         }
//     }

//     return PredicateUtils::combineConjuncts(predicates);
// }

void TableScanStep::aliasColumns(QueryPipelineBuilder & pipeline, const BuildQueryPipelineSettings & build_context)
{
    // simple fix, remove this.
    if (!blocksHaveEqualStructure(pipeline.getHeader(), table_output_stream.header))
    {
        ASTPtr select = std::make_shared<ASTExpressionList>();
        for (const auto & column : column_alias)
        {
            select->children.emplace_back(std::make_shared<ASTIdentifier>(column.first));
        }
        auto actions
            = createExpressionActions(build_context.context, pipeline.getHeader().getNamesAndTypesList(), column_alias, select, true);
        auto expression = std::make_shared<ExpressionActions>(actions, build_context.getActionsSettings());
        pipeline.addSimpleTransform(
            [&](const Block & header) -> ProcessorPtr { return std::make_shared<ExpressionTransform>(header, expression); });
    }
}

// void TableScanStep::setQuotaAndLimits(QueryPipeline & pipeline, const SelectQueryOptions & options, const BuildQueryPipelineSettings & build_context)
// {
//     auto context = build_context.context;
//     const auto & settings = context->getSettingsRef();

//     StreamLocalLimits limits;
//     SizeLimits leaf_limits;
//     std::shared_ptr<const EnabledQuota> quota;

//     /// Set the limits and quota for reading data, the speed and time of the query.
//     if (!options.ignore_limits)
//     {
//         limits = getLimitsForStorage(settings, options);
//         leaf_limits = SizeLimits(settings.max_rows_to_read_leaf, settings.max_bytes_to_read_leaf, settings.read_overflow_mode_leaf);
//     }

//     if (!options.ignore_quota && (options.to_stage == QueryProcessingStage::Complete))
//         quota = context->getQuota();

//     auto table_lock = storage->lockForShare(context->getInitialQueryId(), context->getSettingsRef().lock_acquire_timeout);


//     /// Table lock is stored inside pipeline here.
//     pipeline.setLimits(limits);

//     /**
//       * Leaf size limits should be applied only for local processing of distributed queries.
//       * Such limits allow to control the read stage on leaf nodes and exclude the merging stage.
//       * Consider the case when distributed query needs to read from multiple shards. Then leaf
//       * limits will be applied on the shards only (including the root node) but will be ignored
//       * on the results merging stage.
//       */
//     if (!storage->isRemote())
//         pipeline.setLeafLimits(leaf_limits);

//     if (quota)
//         pipeline.setQuota(quota);

//     /// Order of resources below is important.
//     if (context)
//         pipeline.addInterpreterContext(std::move(context));

//     if (storage)
//         pipeline.addStorageHolder(storage);

//     if (table_lock)
//         pipeline.addTableLock(std::move(table_lock));

//     for (const auto & processor : pipeline.getProcessors())
//         processors.emplace_back(processor);
// }

void TableScanStep::setReadOrder(SortDescription read_order)
{
    if (!read_order.empty())
    {
        query_info.input_order_info = std::make_shared<InputOrderInfo>(read_order, read_order[0].direction);
    }
}

}
