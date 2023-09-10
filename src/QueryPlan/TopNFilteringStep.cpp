#include <QueryPlan/TopNFilteringStep.h>
#include <IO/Operators.h>
#include <Processors/QueryPipeline.h>
#include <Processors/Transforms/TopNFilteringTransform.h>

namespace DB
{

TopNFilteringStep::TopNFilteringStep(const DataStream & input_stream_,
                                     SortDescription sort_description_,
                                     UInt64 size_,
                                     TopNModel model_)
    : ITransformingStep(input_stream_, input_stream_.header, {}),
    sort_description(std::move(sort_description_)),
    size(size_),
    model(model_)
{}

void TopNFilteringStep::setInputStreams(const DataStreams & input_streams_)
{
    input_streams = input_streams_;
    output_stream->header = input_streams_[0].header;
}

void TopNFilteringStep::transformPipeline(QueryPipeline & pipeline, const BuildQueryPipelineSettings &)
{
    pipeline.addSimpleTransform(
        [&](const Block & header) { return std::make_shared<TopNFilteringTransform>(header, sort_description, size, model); });
}

void TopNFilteringStep::describeActions(FormatSettings &) const
{
}

void TopNFilteringStep::describeActions(JSONBuilder::JSONMap &) const
{
}

void TopNFilteringStep::serialize(WriteBuffer & buffer) const
{
    IQueryPlanStep::serializeImpl(buffer);
    serializeSortDescription(sort_description, buffer);
    writeBinary(size, buffer);
    serializeEnum(model, buffer);
}

QueryPlanStepPtr TopNFilteringStep::deserialize(ReadBuffer & buffer, ContextPtr)
{
    String step_description;
    readBinary(step_description, buffer);

    DataStream input_stream;
    input_stream = deserializeDataStream(buffer);

    SortDescription sort_description;
    deserializeSortDescription(sort_description, buffer);

    UInt64 size;
    readBinary(size, buffer);

    TopNModel model;
    deserializeEnum(model, buffer);

    auto step = std::make_shared<TopNFilteringStep>(input_stream, std::move(sort_description), size, model);

    step->setStepDescription(step_description);
    return step;
}

std::shared_ptr<IQueryPlanStep> TopNFilteringStep::copy(ContextPtr) const
{
    return std::make_shared<TopNFilteringStep>(input_streams[0], sort_description, size, model);
}

}
