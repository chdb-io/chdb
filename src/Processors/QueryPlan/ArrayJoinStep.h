#pragma once
#include <Processors/QueryPlan/ITransformingStep.h>

namespace DB
{

class ArrayJoinAction;
using ArrayJoinActionPtr = std::shared_ptr<ArrayJoinAction>;

class ArrayJoinStep : public ITransformingStep
{
public:
    explicit ArrayJoinStep(const DataStream & input_stream_, ArrayJoinActionPtr array_join_);
    String getName() const override { return "ArrayJoin"; }

    Type getType() const override { return Type::ArrayJoin; }

    void transformPipeline(QueryPipelineBuilder & pipeline, const BuildQueryPipelineSettings &) override;

    void describeActions(JSONBuilder::JSONMap & map) const override;
    void describeActions(FormatSettings & settings) const override;

    const ArrayJoinActionPtr & arrayJoin() const { return array_join; }

    std::shared_ptr<IQueryPlanStep> copy(ContextPtr ptr) const override;
    void setInputStreams(const DataStreams & input_streams_) override;

private:
    void updateOutputStream() override;

    ArrayJoinActionPtr array_join;
};

}
