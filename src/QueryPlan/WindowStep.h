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

#pragma once
#include <Processors/QueryPlan/ITransformingStep.h>

#include <Interpreters/WindowDescription.h>

namespace DB
{

class ActionsDAG;
using ActionsDAGPtr = std::shared_ptr<ActionsDAG>;

class WindowTransform;

class WindowStep : public ITransformingStep
{
public:
    explicit WindowStep(const DataStream & input_stream_,
            const WindowDescription & window_description_,
            const std::vector<WindowFunctionDescription> & window_functions_,
            bool need_sort_,
            SortDescription prefix_description_ = {});

    WindowStep(const DataStream & input_stream_, const WindowDescription & window_description_, bool need_sort_, SortDescription prefix_description_);

    String getName() const override { return "Window"; }

    Type getType() const override { return Type::Window; }
    const WindowDescription & getWindow() const { return window_description; }
    const std::vector<WindowFunctionDescription> & getFunctions() const { return window_functions; }
    bool needSort() const { return need_sort; }

    void transformPipeline(QueryPipeline & pipeline, const BuildQueryPipelineSettings &) override;

    void describeActions(JSONBuilder::JSONMap & map) const override;
    void describeActions(FormatSettings & settings) const override;
    std::shared_ptr<IQueryPlanStep> copy(ContextPtr ptr) const override;
    void serialize(WriteBuffer & buffer) const override;
    static QueryPlanStepPtr deserialize(ReadBuffer & buf, ContextPtr);
    void setInputStreams(const DataStreams & input_streams_) override;

    const WindowDescription & getWindowDescription() const { return window_description; }
    const SortDescription & getPrefixDescription() const { return prefix_description; }
    void setPrefixDescription(const SortDescription & prefix_description_) { prefix_description = prefix_description_; }

private:
    WindowDescription window_description;
    std::vector<WindowFunctionDescription> window_functions;
    Block input_header;
    bool need_sort;
    SortDescription prefix_description;
};

}
