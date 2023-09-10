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

#include <QueryPlan/SortingStep.h>
#include <QueryPipeline/QueryPipeline.h>
#include <Processors/Transforms/MergeSortingTransform.h>
#include <Processors/Transforms/PartialSortingTransform.h>
#include <Processors/Transforms/FinishSortingTransform.h>
#include <Processors/Transforms/LimitsCheckingTransform.h>
#include <Processors/Merges/MergingSortedTransform.h>
#include <IO/Operators.h>
#include <Common/JSONBuilder.h>
#include <Interpreters/Context.h>

namespace DB
{

static ITransformingStep::Traits getTraits(size_t limit, bool is_final_sorting = false)
{
    return ITransformingStep::Traits
        {
            {
                .preserves_distinct_columns = true,
                .returns_single_stream = is_final_sorting,
                .preserves_number_of_streams = !is_final_sorting,
                .preserves_sorting = false,
            },
            {
                .preserves_number_of_rows = limit == 0,
            }
        };
}

SortingStep::SortingStep(
    const DataStream & input_stream_,
    SortDescription result_description_,
    UInt64 limit_,
    bool partial_,
    SortDescription prefix_description_)
    : ITransformingStep(input_stream_, input_stream_.header, getTraits(limit_, !partial_))
    , prefix_description(prefix_description_)
    , result_description(result_description_)
    , limit(limit_)
    , partial(partial_)
{
    /// TODO: check input_stream is partially sorted by the same description.
    output_stream->sort_description = result_description;
    output_stream->sort_mode = input_stream_.has_single_port ? DataStream::SortScope::Stream
                                                             : DataStream::SortScope::Port;
}

void SortingStep::setInputStreams(const DataStreams & input_streams_)
{
    input_streams = input_streams_;
    output_stream->header = input_streams_[0].header;
}

void SortingStep::updateLimit(size_t limit_)
{
    if (limit_ && (limit == 0 || limit_ < limit))
    {
        limit = limit_;
        transform_traits.preserves_number_of_rows = false;
    }
}

void SortingStep::transformPipeline(QueryPipeline & pipeline, const BuildQueryPipelineSettings & settings)
{
    auto local_settings = settings.context->getSettingsRef();
    SizeLimits size_limits(local_settings.max_rows_to_sort, local_settings.max_bytes_to_sort, local_settings.sort_overflow_mode);

    auto desc_copy = result_description;

    // finish sorting
    if (!prefix_description.empty())
    {
        bool need_finish_sorting = (prefix_description.size() < result_description.size());
        if (pipeline.getNumStreams() > 1)
        {
            UInt64 limit_for_merging = (need_finish_sorting ? 0 : limit);
            auto transform = std::make_shared<MergingSortedTransform>(
                    pipeline.getHeader(),
                    pipeline.getNumStreams(),
                    prefix_description,
                    local_settings.max_block_size, limit_for_merging);

            pipeline.addTransform(std::move(transform));
        }

        if (need_finish_sorting)
        {
            pipeline.addSimpleTransform([&](const Block & header, QueryPipeline::StreamType stream_type) -> ProcessorPtr
            {
                if (stream_type != QueryPipeline::StreamType::Main)
                    return nullptr;

                return std::make_shared<PartialSortingTransform>(header, result_description, limit);
            });

            /// NOTE limits are not applied to the size of temporary sets in FinishSortingTransform
            pipeline.addSimpleTransform([&](const Block & header) -> ProcessorPtr
            {
                return std::make_shared<FinishSortingTransform>(
                    header, prefix_description, result_description, local_settings.max_block_size, limit);
            });
        }
        return;
    }

    pipeline.addSimpleTransform([&](const Block & header, QueryPipeline::StreamType stream_type) -> ProcessorPtr
                                {
                                    if (stream_type != QueryPipeline::StreamType::Main)
                                        return nullptr;

                                    return std::make_shared<PartialSortingTransform>(header, desc_copy, limit);
                                });

    StreamLocalLimits limits;
    limits.mode = LimitsMode::LIMITS_CURRENT; //-V1048
    limits.size_limits = size_limits;

    pipeline.addSimpleTransform([&](const Block & header, QueryPipeline::StreamType stream_type) -> ProcessorPtr
                                {
                                    if (stream_type != QueryPipeline::StreamType::Main)
                                        return nullptr;

                                    auto transform = std::make_shared<LimitsCheckingTransform>(header, limits);
                                    return transform;
                                });

    pipeline.addSimpleTransform([&](const Block & header, QueryPipeline::StreamType stream_type) -> ProcessorPtr
                                {
                                    if (stream_type == QueryPipeline::StreamType::Totals)
                                        return nullptr;

                                    return std::make_shared<MergeSortingTransform>(
                                        header, result_description, local_settings.max_block_size, limit,
                                        local_settings.max_bytes_before_remerge_sort / pipeline.getNumStreams(),
                                        local_settings.remerge_sort_lowered_memory_bytes_ratio,
                                        local_settings.max_bytes_before_external_sort,
                                        settings.context->getTemporaryVolume(),
                                        local_settings.min_free_disk_space_for_temporary_data);
                                });
    if (!partial)
    {
        /// If there are several streams, then we merge them into one
        if (pipeline.getNumStreams() > 1)
        {

            auto transform = std::make_shared<MergingSortedTransform>(
                pipeline.getHeader(),
                pipeline.getNumStreams(),
                desc_copy,
                local_settings.max_block_size, limit);

            pipeline.addTransform(std::move(transform));
        }
    }
}

void SortingStep::describeActions(FormatSettings & settings) const
{
    String prefix(settings.offset, ' ');
    settings.out << prefix << "Sort description: ";
    dumpSortDescription(result_description, input_streams.front().header, settings.out);
    settings.out << '\n';

    if (limit)
        settings.out << prefix << "Limit " << limit << '\n';
}

void SortingStep::describeActions(JSONBuilder::JSONMap & map) const
{
    map.add("Sort Description", explainSortDescription(result_description, input_streams.front().header));

    if (limit)
        map.add("Limit", limit);
}

void SortingStep::serialize(WriteBuffer & buffer) const
{
    IQueryPlanStep::serializeImpl(buffer);
    serializeItemVector<SortColumnDescription>(result_description, buffer);
    writeBinary(limit, buffer);
    writeBinary(partial, buffer);
    serializeItemVector<SortColumnDescription>(prefix_description, buffer);

}

QueryPlanStepPtr SortingStep::deserialize(ReadBuffer & buffer, ContextPtr)
{
    String step_description;
    readBinary(step_description, buffer);

    DataStream input_stream;
    input_stream = deserializeDataStream(buffer);

    SortDescription sort_description;
    sort_description = deserializeItemVector<SortColumnDescription>(buffer);

    UInt64 limit;
    readBinary(limit, buffer);

    bool partial;
    readVarUInt(partial, buffer);

    SortDescription prefix_description;
    prefix_description = deserializeItemVector<SortColumnDescription>(buffer);

    auto step = std::make_unique<SortingStep>(
        input_stream,
        sort_description,
        limit,
        partial,
        prefix_description);

    step->setStepDescription(step_description);
    return step;
}

std::shared_ptr<IQueryPlanStep> SortingStep::copy(ContextPtr) const
{
    return std::make_shared<SortingStep>(
        input_streams[0],
        result_description,
        limit,
        partial,
        prefix_description);
}

}
