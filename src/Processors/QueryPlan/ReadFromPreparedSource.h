#pragma once
#include <Processors/QueryPlan/ISourceStep.h>
#include <QueryPipeline/Pipe.h>

namespace DB
{

/// Create source from prepared pipe.
class ReadFromPreparedSource : public ISourceStep
{
public:
    explicit ReadFromPreparedSource(Pipe pipe_);

    String getName() const override { return "ReadFromPreparedSource"; }

    Type getType() const override { return Type::ReadFromPreparedSource; }

    void initializePipeline(QueryPipelineBuilder & pipeline, const BuildQueryPipelineSettings &) override;
    std::shared_ptr<IQueryPlanStep> copy(ContextPtr ptr) const override;

protected:
    Pipe pipe;
    ContextPtr context;
};

class ReadFromStorageStep : public ReadFromPreparedSource
{
public:
    ReadFromStorageStep(Pipe pipe_, String storage_name, std::shared_ptr<const StorageLimitsList> storage_limits_)
        : ReadFromPreparedSource(std::move(pipe_)), storage_limits(std::move(storage_limits_))
    {
        setStepDescription(storage_name);

        for (const auto & processor : pipe.getProcessors())
            processor->setStorageLimits(storage_limits);
    }

    String getName() const override { return "ReadFromStorage"; }

    Type getType() const override { return Type::ReadFromStorage; }

    std::shared_ptr<IQueryPlanStep> copy(ContextPtr) const override;

private:
    std::shared_ptr<const StorageLimitsList> storage_limits;
};

}
