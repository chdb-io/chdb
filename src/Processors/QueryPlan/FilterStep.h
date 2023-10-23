#pragma once
#include <Processors/QueryPlan/ITransformingStep.h>

namespace DB
{

class ActionsDAG;
using ActionsDAGPtr = std::shared_ptr<ActionsDAG>;

/// Implements WHERE, HAVING operations. See FilterTransform.
class FilterStep : public ITransformingStep
{
public:
    FilterStep(
        const DataStream & input_stream_,
        const ActionsDAGPtr & actions_dag_,
        String filter_column_name_,
        bool remove_filter_column_);

    FilterStep(const DataStream & input_stream_, const ConstASTPtr & filter_, bool remove_filter_column_ = true);

    String getName() const override { return "Filter"; }

    Type getType() const override { return Type::Filter; }

    void transformPipeline(QueryPipelineBuilder & pipeline, const BuildQueryPipelineSettings & settings) override;
    void updateInputStream(DataStream input_stream, bool keep_header);

    void describeActions(JSONBuilder::JSONMap & map) const override;
    void describeActions(FormatSettings & settings) const override;

    const ActionsDAGPtr & getExpression() const { return actions_dag; }
    const ConstASTPtr & getFilter() const { return filter; }
    void setFilter(ConstASTPtr new_filter) { filter = std::move(new_filter); }
    const String & getFilterColumnName() const { return filter_column_name; }
    bool removesFilterColumn() const { return remove_filter_column; }

    ActionsDAGPtr createActions(ContextPtr context, ConstASTPtr rewrite_filter) const;

    void serialize(WriteBuffer &) const override
    {
        throw Exception("FilterStep::serializable is not implemented", ErrorCodes::NOT_IMPLEMENTED);
    }
    static QueryPlanStepPtr deserialize(ReadBuffer &, ContextPtr)
    {
        throw Exception("FilterStep::deserialize is not implemented", ErrorCodes::NOT_IMPLEMENTED);
    }
    std::shared_ptr<IQueryPlanStep> copy(ContextPtr ptr) const override;
    void setInputStreams(const DataStreams & input_streams_) override;

private:
    void updateOutputStream() override;

    ActionsDAGPtr actions_dag;
    ConstASTPtr filter;
    String filter_column_name;
    bool remove_filter_column;
};

}
