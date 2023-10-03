#include <Processors/QueryPlan/MarkDistinctStep.h>

#include <DataTypes/DataTypeHelper.h>
#include <IO/Operators.h>
#include <Interpreters/ExpressionActions.h>
// #include <Interpreters/RuntimeFilter/BuildRuntimeFilterTransform.h>
#include <QueryPipeline/QueryPipelineBuilder.h>
#include <Processors/Transforms/MarkDistinctTransform.h>

namespace DB
{
MarkDistinctStep::MarkDistinctStep(const DataStream & input_stream_, String marker_symbol_, std::vector<String> distinct_symbols_)
    : ITransformingStep(input_stream_, MarkDistinctTransform::transformHeader(input_stream_.header, marker_symbol_), {}, true), marker_symbol(marker_symbol_), distinct_symbols(std::move(distinct_symbols_))
{
}

void MarkDistinctStep::setInputStreams(const DataStreams & input_streams_)
{
    input_streams = input_streams_;
    output_stream = input_streams[0];
    output_stream->header.insert(ColumnWithTypeAndName{std::make_shared<DataTypeUInt8>(), marker_symbol});
}

void MarkDistinctStep::transformPipeline(QueryPipelineBuilder & pipeline, const BuildQueryPipelineSettings &)
{
    // TODO Add Local Exchange
    pipeline.resize(1);
    pipeline.addSimpleTransform([&](const Block & header) { return std::make_shared<MarkDistinctTransform>(header, marker_symbol, distinct_symbols); });
}

void MarkDistinctStep::serialize(WriteBuffer & buf) const
{
    IQueryPlanStep::serializeImpl(buf);
    writeBinary(marker_symbol, buf);
    writeBinary(distinct_symbols, buf);
}

QueryPlanStepPtr MarkDistinctStep::deserialize(ReadBuffer &, ContextPtr)
{
    throw Exception("MarkDistinctStep is not supported.", ErrorCodes::NOT_IMPLEMENTED);
    // String step_description;
    // readBinary(step_description, buf);

    // DataStream input_stream = deserializeDataStream(buf);
    // String marker_symbol;
    // readBinary(marker_symbol, buf);

    // std::vector<String> distinct_symbols;
    // readBinary(distinct_symbols, buf);
    // return std::make_shared<MarkDistinctStep>(input_stream, marker_symbol, distinct_symbols);
}

std::shared_ptr<IQueryPlanStep> MarkDistinctStep::copy(ContextPtr) const
{
    return std::make_shared<MarkDistinctStep>(input_streams[0], marker_symbol, distinct_symbols);
}

}
