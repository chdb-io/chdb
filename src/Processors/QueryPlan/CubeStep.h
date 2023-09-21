#pragma once
#include <Processors/QueryPlan/ITransformingStep.h>
#include <QueryPipeline/SizeLimits.h>
#include <Interpreters/Aggregator.h>

namespace DB
{

struct AggregatingTransformParams;
using AggregatingTransformParamsPtr = std::shared_ptr<AggregatingTransformParams>;

/// WITH CUBE. See CubeTransform.
class CubeStep : public ITransformingStep
{
public:
    CubeStep(const DataStream & input_stream_, Aggregator::Params params_, bool final_, bool use_nulls_);

    String getName() const override { return "Cube"; }

    Type getType() const override { return Type::Cube; }

    void transformPipeline(QueryPipelineBuilder & pipeline, const BuildQueryPipelineSettings &) override;

    const Aggregator::Params & getParams() const;

    std::shared_ptr<IQueryPlanStep> copy(ContextPtr ptr) const override;
    void setInputStreams(const DataStreams & input_streams_) override;

private:
    void updateOutputStream() override;

    size_t keys_size;
    Aggregator::Params params;
    bool final;
    bool use_nulls;
};

}
