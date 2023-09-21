#include <Processors/QueryPlan/ReadFromPreparedSource.h>
#include <QueryPipeline/QueryPipelineBuilder.h>

namespace DB
{

ReadFromPreparedSource::ReadFromPreparedSource(Pipe pipe_)
    : ISourceStep(DataStream{.header = pipe_.getHeader()})
    , pipe(std::move(pipe_))
{
}

void ReadFromPreparedSource::initializePipeline(QueryPipelineBuilder & pipeline, const BuildQueryPipelineSettings &)
{
    for (const auto & processor : pipe.getProcessors())
        processors.emplace_back(processor);

    pipeline.init(std::move(pipe));
}

std::shared_ptr<IQueryPlanStep> ReadFromPreparedSource::copy(ContextPtr) const
{
    throw Exception("ReadFromPreparedSource can not copy", ErrorCodes::NOT_IMPLEMENTED);
}

std::shared_ptr<IQueryPlanStep> ReadFromStorageStep::copy(ContextPtr) const
{
    throw Exception("ReadFromStorageStep can not copy", ErrorCodes::NOT_IMPLEMENTED);
}

}
