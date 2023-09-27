#pragma once
#include <Core/Block.h>
#include <Core/NameToType.h>
#include <Core/SortDescription.h>
#include <Parsers/IAST_fwd.h>
#include <Processors/QueryPlan/BuildQueryPipelineSettings.h>
#include <Processors/QueryPlan/Hints/IPlanHint.h>
#include <QueryPipeline/QueryPipelineBuilder.h>
#include <Processors/QueryPlan/PlanSerDerHelper.h>


namespace DB
{

class QueryPipeline;
using QueryPipelinePtr = std::unique_ptr<QueryPipeline>;
using QueryPipelines = std::vector<QueryPipelinePtr>;

class QueryPipelineBuilder;
using QueryPipelineBuilderPtr = std::unique_ptr<QueryPipelineBuilder>;
using QueryPipelineBuilders = std::vector<QueryPipelineBuilderPtr>;

class IProcessor;
using ProcessorPtr = std::shared_ptr<IProcessor>;
using Processors = std::vector<ProcessorPtr>;

class ActionsDAG;
using ActionsDAGPtr = std::shared_ptr<ActionsDAG>;
namespace JSONBuilder { class JSONMap; }

/// Description of data stream.
/// Single logical data stream may relate to many ports of pipeline.
class DataStream
{
public:
    Block header;

    /// Tuples with those columns are distinct.
    /// It doesn't mean that columns are distinct separately.
    /// Removing any column from this list brakes this invariant.
    NameSet distinct_columns = {};

    /// QueryPipeline has single port. Totals or extremes ports are not counted.
    bool has_single_port = false;

    /// Sorting scope. Please keep the mutual order (more strong mode should have greater value).
    enum class SortScope
    {
        None   = 0,
        Chunk  = 1, /// Separate chunks are sorted
        Stream = 2, /// Each data steam is sorted
        Global = 3, /// Data is globally sorted
    };

    /// It is not guaranteed that header has columns from sort_description.
    SortDescription sort_description = {};
    SortScope sort_scope = SortScope::None;

    /// Things which may be added:
    /// * limit
    /// * estimated rows number
    /// * memory allocation context

    bool hasEqualPropertiesWith(const DataStream & other) const
    {
        return has_single_port == other.has_single_port
            && sort_description == other.sort_description
            && (sort_description.empty() || sort_scope == other.sort_scope);
    }

    bool hasEqualHeaderWith(const DataStream & other) const
    {
        return blocksHaveEqualStructure(header, other.header);
    }

    NamesAndTypes getNamesAndTypes() const { return header.getNamesAndTypes(); }

    NameToType getNamesToTypes() const { return header.getNamesToTypes(); }
};

using DataStreams = std::vector<DataStream>;

class IQueryPlanStep;
class Context;
using ContextPtr = std::shared_ptr<const Context>;

using PlanHints = std::vector<PlanHintPtr>;

using QueryPlanStepPtr = std::shared_ptr<IQueryPlanStep>;

/// Single step of query plan.
class IQueryPlanStep
{
public:
#define APPLY_STEP_TYPES(M) \
    M(Aggregating) \
    M(Apply) \
    M(ArrayJoin) \
    M(AssignUniqueId) \
    M(CreateSetAndFilterOnTheFly) \
    M(CreatingSet) \
    M(CreatingSets) \
    M(Cube) \
    M(Distinct) \
    M(DelayedCreatingSets) \
    M(EnforceSingleRow) \
    M(Expression) \
    M(Extremes) \
    M(Except) \
    M(ExplainAnalyze) \
    M(Filling) \
    M(FilledJoin) \
    M(Filter) \
    M(FinalSample) \
    M(FinishSorting) \
    M(ISource) \
    M(ITransforming) \
    M(Intersect) \
    M(Join) \
    M(LimitBy) \
    M(Limit) \
    M(MergeSorting) \
    M(Sorting) \
    M(MergingAggregated) \
    M(MergingSorted) \
    M(Offset) \
    M(PartitionTopN) \
    M(PartialSorting) \
    M(PlanSegmentSource) \
    M(Projection) \
    M(ReadFromStorage) \
    M(ReadNothing) \
    M(Rollup) \
    M(TotalsHaving) \
    M(TableScan) \
    M(Union) \
    M(Values) \
    M(Window) \
    M(CTERef) \
    M(TopNFiltering) \
    M(MarkDistinct) \
    M(IntersectOrExcept)

#define ENUM_DEF(ITEM) ITEM,

    enum class Type
    {
        Any = 0,
        APPLY_STEP_TYPES(ENUM_DEF) UNDEFINED,
        ReadFromMergeTree,
        ReadFromCnchHive,
        ReadFromPreparedSource,
        NullSource,
        Tree,
    };

#undef ENUM_DEF

    virtual ~IQueryPlanStep() = default;

    virtual String getName() const = 0;

    virtual Type getType() const = 0;

    /// Add processors from current step to QueryPipeline.
    /// Calling this method, we assume and don't check that:
    ///   * pipelines.size() == getInputStreams.size()
    ///   * header from each pipeline is the same as header from corresponding input_streams
    /// Result pipeline must contain any number of streams with compatible output header is hasOutputStream(),
    ///   or pipeline should be completed otherwise.
    virtual QueryPipelineBuilderPtr updatePipeline(QueryPipelineBuilders pipelines, const BuildQueryPipelineSettings & settings) = 0;
    static ActionsDAGPtr createExpressionActions(
        ContextPtr context, const NamesAndTypesList & source, const Names & output, const ASTPtr & ast, bool add_project = true);
    static ActionsDAGPtr createExpressionActions(
        ContextPtr context, const NamesAndTypesList & source, const NamesWithAliases & output, const ASTPtr & ast, bool add_project = true);
    static void projection(QueryPipeline & pipeline, const Block & target, const BuildQueryPipelineSettings & settings);
    static void aliases(QueryPipeline & pipeline, const Block & target, const BuildQueryPipelineSettings & settings);

    const DataStreams & getInputStreams() const { return input_streams; }
    virtual void setInputStreams(const DataStreams & input_streams_ __attribute__((unused)))
    {
        throw Exception("Not supported setInputStreams.", ErrorCodes::NOT_IMPLEMENTED);
    }

    void addHints(SqlHints & sql_hints, ContextMutablePtr & context);

    const PlanHints & getHints() const { return hints; }

    void setHints(const PlanHints & new_hints) { hints = new_hints; }


    bool hasOutputStream() const { return output_stream.has_value(); }
    const DataStream & getOutputStream() const;

    /// Methods to describe what this step is needed for.
    const std::string & getStepDescription() const { return step_description; }
    void setStepDescription(std::string description) { step_description = std::move(description); }

    struct FormatSettings
    {
        WriteBuffer & out;
        size_t offset = 0;
        const size_t indent = 2;
        const char indent_char = ' ';
        const bool write_header = false;
    };

    /// Get detailed description of step actions. This is shown in EXPLAIN query with options `actions = 1`.
    virtual void describeActions(JSONBuilder::JSONMap & /*map*/) const {}
    virtual void describeActions(FormatSettings & /*settings*/) const {}

    /// Get detailed description of read-from-storage step indexes (if any). Shown in with options `indexes = 1`.
    virtual void describeIndexes(JSONBuilder::JSONMap & /*map*/) const {}
    virtual void describeIndexes(FormatSettings & /*settings*/) const {}

    /// Get description of processors added in current step. Should be called after updatePipeline().
    virtual void describePipeline(FormatSettings & /*settings*/) const {}

    /// Append extra processors for this step.
    void appendExtraProcessors(const Processors & extra_processors);
    virtual void serializeImpl(WriteBuffer & buf) const;
    virtual void serialize(WriteBuffer &) const { throw Exception("Not supported serialize.", ErrorCodes::NOT_IMPLEMENTED); }
    static QueryPlanStepPtr deserialize(ReadBuffer &, ContextPtr)
    {
        throw Exception("Not supported deserialize.", ErrorCodes::NOT_IMPLEMENTED);
    }

    virtual bool isPhysical() const { return true; }
    virtual bool isLogical() const { return true; }

    virtual std::shared_ptr<IQueryPlanStep> copy(ContextPtr) const { throw Exception("Not supported copy.", ErrorCodes::NOT_IMPLEMENTED); }

    size_t hash() const;

    // bool operator==(const IQueryPlanStep & r) const { return serializeToString() == r.serializeToString(); }
    bool operator==(const IQueryPlanStep &) const { throw Exception("Not supported operator==.", ErrorCodes::NOT_IMPLEMENTED); }
    static String toString(Type type);

protected:
    DataStreams input_streams;
    std::optional<DataStream> output_stream;

    /// Text description about what current step does.
    std::string step_description;

    /// This field is used to store added processors from this step.
    /// It is used only for introspection (EXPLAIN PIPELINE).
    Processors processors;
    PlanHints hints;
    static void describePipeline(const Processors & processors, FormatSettings & settings);

private:
    // virtual String serializeToString() const
    // {
    //     WriteBufferFromOwnString buffer;
    //     serialize(buffer);
    //     return buffer.str();
    // }
};
}
