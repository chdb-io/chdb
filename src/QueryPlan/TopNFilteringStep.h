#pragma once
#include <Core/SortDescription.h>
#include <QueryPlan/ITransformingStep.h>
#include <QueryPlan/TopNModel.h>

namespace DB
{

/// TopNFilteringStep filter out data which cannot be in the top N.
///
/// A possible implementation can be:
/// the operator keeps a state which contains the largest k values ever seen,
/// when a new row come, if it is greater than the low bound of k values, accept this row and update the state;
/// otherwise, reject this row.
/// e.g. input = {1, 3, 5, 2, 4, 6}, output of TopNFiltering(3) = {1, 3, 5, 2}
class TopNFilteringStep : public ITransformingStep
{
public:
    TopNFilteringStep(const DataStream & input_stream_, SortDescription sort_description_, UInt64 size_, TopNModel model_);

    String getName() const override { return "TopNFiltering"; }
    Type getType() const override { return Type::TopNFiltering; }

    void setInputStreams(const DataStreams & input_streams_) override;
    void transformPipeline(QueryPipeline & pipeline, const BuildQueryPipelineSettings &) override;
    void describeActions(JSONBuilder::JSONMap & map) const override;
    void describeActions(FormatSettings & settings) const override;
    void serialize(WriteBuffer &) const override;
    static QueryPlanStepPtr deserialize(ReadBuffer &, ContextPtr context_ = nullptr);
    std::shared_ptr<IQueryPlanStep> copy(ContextPtr ptr) const override;

    const SortDescription & getSortDescription() const { return sort_description; }
    UInt64 getSize() const { return size; }
    TopNModel getModel() const { return model; }
private:
    SortDescription sort_description;
    UInt64 size;
    TopNModel model;
};

}
