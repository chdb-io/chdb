#include <Processors/QueryPlan/ExtremesStep.h>
#include <QueryPipeline/QueryPipelineBuilder.h>

namespace DB
{

static ITransformingStep::Traits getTraits()
{
    return ITransformingStep::Traits
    {
        {
            .returns_single_stream = false,
            .preserves_number_of_streams = true,
            .preserves_sorting = true,
        },
        {
            .preserves_number_of_rows = true,
        }
    };
}

ExtremesStep::ExtremesStep(const DataStream & input_stream_)
    : ITransformingStep(input_stream_, input_stream_.header, getTraits())
{
}

void ExtremesStep::setInputStreams(const DataStreams & input_streams_)
{
    input_streams = input_streams_;
    output_stream->header = input_streams_[0].header;
}

void ExtremesStep::transformPipeline(QueryPipelineBuilder & pipeline, const BuildQueryPipelineSettings &)
{
    pipeline.addExtremesTransform();
}
std::shared_ptr<IQueryPlanStep> ExtremesStep::copy(ContextPtr) const
{
    return std::make_shared<ExtremesStep>(input_streams[0]);
}
}
