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

#include <QueryPlan/IntersectStep.h>

#include <Core/Block.h>
#include <Interpreters/Context.h>
#include <Interpreters/ExpressionActions.h>
#include <QueryPipeline/QueryPipeline.h>
#include <Processors/ResizeProcessor.h>
#include <Processors/Sources/NullSource.h>
#include <Processors/Transforms/ExpressionTransform.h>
// #include <Processors/Transforms/IntersectOrExceptTransform.h>

namespace DB
{
IntersectStep::IntersectStep(
    DataStreams input_streams_,
    DataStream output_stream_,
    std::unordered_map<String, std::vector<String>> output_to_inputs_,
    bool distinct_)
    : SetOperationStep(input_streams_, output_stream_, output_to_inputs_), distinct(distinct_)
{
}

QueryPipelinePtr IntersectStep::updatePipeline(QueryPipelineBuilders pipelines, const BuildQueryPipelineSettings & context)
{
    throw Exception("intersect step is not implemented", ErrorCodes::NOT_IMPLEMENTED);
    // auto pipeline = std::make_unique<QueryPipeline>();
    // QueryPipelineProcessorsCollector collector(*pipeline, this);

    // if (pipelines.empty())
    // {
    //     pipeline->init(Pipe(std::make_shared<NullSource>(output_stream->header)));
    //     processors = collector.detachProcessors();
    //     return pipeline;
    // }

    // for (auto & cur_pipeline : pipelines)
    // {
    //     /// Just in case.
    //     if (!isCompatibleHeader(cur_pipeline->getHeader(), getOutputStream().header))
    //     {
    //         auto converting_dag = ActionsDAG::makeConvertingActions(
    //             cur_pipeline->getHeader().getColumnsWithTypeAndName(),
    //             getOutputStream().header.getColumnsWithTypeAndName(),
    //             ActionsDAG::MatchColumnsMode::Position);

    //         auto converting_actions = std::make_shared<ExpressionActions>(std::move(converting_dag));
    //         cur_pipeline->addSimpleTransform(
    //             [&](const Block & cur_header) { return std::make_shared<ExpressionTransform>(cur_header, converting_actions); });
    //     }

    //     /// For the case of union.
    //     cur_pipeline->addTransform(std::make_shared<ResizeProcessor>(getOutputStream().header, cur_pipeline->getNumStreams(), 1));
    // }

    // *pipeline = QueryPipeline::unitePipelines(std::move(pipelines), context.context->getSettingsRef().max_threads);
    // pipeline->addTransform(std::make_shared<IntersectOrExceptTransform>(
    //     getOutputStream().header,
    //     distinct ? ASTSelectIntersectExceptQuery::Operator::INTERSECT_DISTINCT : ASTSelectIntersectExceptQuery::Operator::INTERSECT_ALL));

    // processors = collector.detachProcessors();
    // return pipeline;
}

void IntersectStep::serialize(WriteBuffer & buffer) const
{
    writeBinary(input_streams.size(), buffer);
    for (const auto & input_stream : input_streams)
        serializeDataStream(input_stream, buffer);

    serializeDataStream(output_stream.value(), buffer);

    writeBinary(distinct, buffer);

    writeVarUInt(output_to_inputs.size(), buffer);
    for (const auto & item : output_to_inputs)
    {
        writeStringBinary(item.first, buffer);
        writeVarUInt(item.second.size(), buffer);
        for (const auto & str : item.second)
        {
            writeStringBinary(str, buffer);
        }
    }
}

QueryPlanStepPtr IntersectStep::deserialize(ReadBuffer & buffer, ContextPtr)
{
    size_t size;
    readBinary(size, buffer);

    DataStreams input_streams(size);
    for (size_t i = 0; i < size; ++i)
        input_streams[i] = deserializeDataStream(buffer);

    auto output_stream = deserializeDataStream(buffer);

    bool distinct;
    readBinary(distinct, buffer);

    std::unordered_map<String, std::vector<String>> output_to_inputs;
    readVarUInt(size, buffer);
    for (size_t index = 0; index < size; index++)
    {
        String output;
        readStringBinary(output, buffer);
        size_t count;
        readVarUInt(count, buffer);
        for (size_t i = 0; i < count; i++)
        {
            String str;
            readStringBinary(str, buffer);
            output_to_inputs[output].emplace_back(str);
        }
    }
    return std::make_unique<IntersectStep>(input_streams, output_stream, output_to_inputs, distinct);
}

bool IntersectStep::isDistinct() const
{
    return distinct;
}

std::shared_ptr<IQueryPlanStep> IntersectStep::copy(ContextPtr) const
{
    return std::make_unique<IntersectStep>(input_streams, output_stream.value(), distinct);
}

}
