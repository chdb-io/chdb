#pragma once
#include <Processors/QueryPlan/ITransformingStep.h>
namespace DB
{

/// Calculate extremes. Add special port for extremes.
class ExtremesStep : public ITransformingStep
{
public:
    explicit ExtremesStep(const DataStream & input_stream_);

    String getName() const override { return "Extremes"; }

    Type getType() const override { return Type::Extremes; }

    void transformPipeline(QueryPipelineBuilder & pipeline, const BuildQueryPipelineSettings &) override;

    std::shared_ptr<IQueryPlanStep> copy(ContextPtr ptr) const override;
    void setInputStreams(const DataStreams & input_streams_) override;

private:
    void updateOutputStream() override
    {
        output_stream = createOutputStream(input_streams.front(), input_streams.front().header, getDataStreamTraits());
    }
};

}
