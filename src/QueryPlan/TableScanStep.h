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
#include <Core/QueryProcessingStage.h>
#include <Core/SortDescription.h>
#include <QueryPlan/ISourceStep.h>
#include <QueryPlan/AggregatingStep.h>
#include <QueryPlan/FilterStep.h>
#include <QueryPlan/ProjectionStep.h>
#include <Interpreters/SelectQueryOptions.h>
#include <Storages/IStorage.h>
#include <Storages/SelectQueryInfo.h>


namespace DB
{

class TableScanStep : public ISourceStep
{
public:
    TableScanStep(
        ContextPtr context,
        StoragePtr storage_,
        const NamesWithAliases & column_alias_,
        const SelectQueryInfo & query_info_,
        size_t max_block_size_,
        String alias_ = "",
        PlanHints hints_ = {},
        QueryPlanStepPtr aggregation_ = nullptr,
        QueryPlanStepPtr projection_ = nullptr,
        QueryPlanStepPtr filter_ = nullptr);

    TableScanStep(
        ContextPtr context,
        StorageID storage_id_,
        const NamesWithAliases & column_alias_,
        const SelectQueryInfo & query_info_,
        size_t max_block_size_,
        String alias_ = "",
        PlanHints hints_ = {},
        QueryPlanStepPtr aggregation_ = nullptr,
        QueryPlanStepPtr projection_ = nullptr,
        QueryPlanStepPtr filter_ = nullptr);


    TableScanStep(
        DataStream output,
        StoragePtr storage_,
        StorageID storage_id_,
        String original_table_,
        Names column_names_,
        NamesWithAliases column_alias_,
        SelectQueryInfo query_info_,
        size_t max_block_size_,
        String alias_,
        PlanHints hints_,
        QueryPlanStepPtr aggregation_,
        QueryPlanStepPtr projection_,
        QueryPlanStepPtr filter_,
        DataStream table_output_stream_)
        : ISourceStep(std::move(output), hints_)
        , storage(storage_)
        , storage_id(storage_id_)
        , original_table(std::move(original_table_))
        , column_names(std::move(column_names_))
        , column_alias(std::move(column_alias_))
        , query_info(std::move(query_info_))
        , max_block_size(max_block_size_)
        , pushdown_aggregation(std::move(aggregation_))
        , pushdown_projection(std::move(projection_))
        , pushdown_filter(std::move(filter_))
        , table_output_stream(std::move(table_output_stream_))
        , log(&Poco::Logger::get("TableScanStep"))
        , alias(alias_)
    {
    }

    String getName() const override { return "TableScan"; }
    Type getType() const override { return Type::TableScan; }

    void initializePipeline(QueryPipeline & pipeline, const BuildQueryPipelineSettings &) override;
    void serialize(WriteBuffer & buffer) const override;
    static QueryPlanStepPtr deserialize(ReadBuffer & buf, ContextPtr context);

    const String & getDatabase() const { return storage_id.database_name; }
    const String & getTable() const { return storage_id.table_name; }
    const String & getTableAlias() const { return alias; }
    //    void setTable(const String & table_);
    void setOriginalTable(const String & original_table_) { original_table = original_table_; }
    const String & getOriginalTable() const { return original_table.empty() ? storage_id.table_name : original_table; }
    const Names & getColumnNames() const { return column_names; }
    const NamesWithAliases & getColumnAlias() const { return column_alias; }
    QueryProcessingStage::Enum getProcessedStage() const;
    size_t getMaxBlockSize() const;

    void setPushdownAggregation(QueryPlanStepPtr aggregation_) { pushdown_aggregation = std::move(aggregation_); }
    void setPushdownProjection(QueryPlanStepPtr projection_) { pushdown_projection = std::move(projection_); }
    void setPushdownFilter(QueryPlanStepPtr filter_) { pushdown_filter = std::move(filter_); }
    const QueryPlanStepPtr & getPushdownAggregation() const { return pushdown_aggregation; }
    const QueryPlanStepPtr & getPushdownProjection() const { return pushdown_projection; }
    const QueryPlanStepPtr & getPushdownFilter() const { return pushdown_filter; }
    const AggregatingStep * getPushdownAggregationCast() const { return dynamic_cast<AggregatingStep *>(pushdown_aggregation.get()); }
    const ProjectionStep * getPushdownProjectionCast() const { return dynamic_cast<ProjectionStep *>(pushdown_projection.get()); }
    const FilterStep * getPushdownFilterCast() const { return dynamic_cast<FilterStep *>(pushdown_filter.get()); }
    AggregatingStep * getPushdownAggregationCast() { return dynamic_cast<AggregatingStep *>(pushdown_aggregation.get()); }
    ProjectionStep * getPushdownProjectionCast() { return dynamic_cast<ProjectionStep *>(pushdown_projection.get()); }
    FilterStep * getPushdownFilterCast() { return dynamic_cast<FilterStep *>(pushdown_filter.get()); }
    const DataStream & getTableOutputStream() const { return table_output_stream; }


    void setReadOrder(SortDescription read_order);

    void formatOutputStream();

    bool setQueryInfoFilter(const std::vector<ConstASTPtr> & filters) const;
    bool hasQueryInfoFilter() const;
    bool setLimit(size_t limit, const ContextMutablePtr & context);
    bool hasLimit() const;

    void optimizeWhereIntoPrewhre(ContextPtr context);

    SelectQueryInfo fillQueryInfo(ContextPtr context);
    void fillPrewhereInfo(ContextPtr context);
    void makeSetsForIndex(const ASTPtr & node, ContextPtr context, PreparedSets & prepared_sets) const;

    void allocate(ContextPtr context);
    Int32 getUniqueId() const { return unique_id; }
    void setUniqueId(Int32 unique_id_) { unique_id = unique_id_; }

    std::shared_ptr<IStorage> getStorage() const;
    const SelectQueryInfo & getQueryInfo() const { return query_info; }
    const StorageID & getStorageID() const { return storage_id; }
    std::shared_ptr<IQueryPlanStep> copy(ContextPtr context) const override;

private:
    StoragePtr storage;
    StorageID storage_id;
    String original_table;
    Names column_names;
    NamesWithAliases column_alias;
    SelectQueryInfo query_info;
    size_t max_block_size;

    // Pushdown steps. Now TableScanStep is not like a single step anymore, but more like a sub plan
    // with structure `Partial Aggregate->Projection->Filter->ReadTable`. And we are able to use
    // **clickhouse projection** to optimize its execution.
    // TODO: better to use a new kind of IQueryPlanStep
    QueryPlanStepPtr pushdown_aggregation;
    QueryPlanStepPtr pushdown_projection;
    QueryPlanStepPtr pushdown_filter;
    DataStream table_output_stream;

    // just for cascades, in order to distinguish between the same tables.
    Int32 unique_id{0};
    Poco::Logger * log;
    String alias;

    // Optimises the where clauses for a bucket table by rewriting the IN clause and hence reducing the IN set size
    void rewriteInForBucketTable(ContextPtr context) const;
    static ASTPtr rewriteDynamicFilter(const ASTPtr & filter, QueryPipeline & pipeline, const BuildQueryPipelineSettings & build_context);
    void aliasColumns(QueryPipeline & pipeline, const BuildQueryPipelineSettings &);
    void setQuotaAndLimits(QueryPipeline & pipeline, const SelectQueryOptions & options, const BuildQueryPipelineSettings &);
};

}
