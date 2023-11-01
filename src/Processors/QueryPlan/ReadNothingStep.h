#pragma once
#include <Processors/QueryPlan/ISourceStep.h>

namespace DB
{

/// Create NullSource with specified structure.
class ReadNothingStep : public ISourceStep
{
public:
    explicit ReadNothingStep(Block output_header);

    String getName() const override { return "ReadNothing"; }

    Type getType() const override { return Type::ReadNothing; }

    void initializePipeline(QueryPipelineBuilder & pipeline, const BuildQueryPipelineSettings &) override;

    std::shared_ptr<IQueryPlanStep> copy(ContextPtr ptr) const override;
};

}
