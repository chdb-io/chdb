#pragma once
#include <Processors/ISimpleTransform.h>
#include <Processors/QueryPlan/ExplainAnalyzeStep.h>
#include <Interpreters/ProcessorProfile.h>
#include <Interpreters/Context.h>
//#include <Interpreters/profile/ProfileLogHub.h>
#include <Interpreters/ProcessorsProfileLog.h>


namespace DB
{
using ProcessorsSet = std::unordered_set<const IProcessor *>;

class ExplainAnalyzeTransform : public ISimpleTransform
{
public:
    ExplainAnalyzeTransform(const Block & header_,
                            ASTExplainQuery::ExplainKind kind_,
                            std::shared_ptr<QueryPlan> query_plan_ptr_,
                            ContextMutablePtr context_);

    String getName() const override { return "ExplainAnalyzeTransform"; }
protected:
    void transform(Chunk & chunk) override;
    ISimpleTransform::Status prepare() override;

    static void getProcessorProfiles(ProcessorsSet & processors_set, ProcessorProfiles & profiles, const IProcessor * processor);
    void getRemoteProcessorProfiles(std::unordered_map<size_t, std::unordered_map<String, ProcessorProfiles>> & segment_profiles);
private:
    ASTExplainQuery::ExplainKind kind;
    ContextMutablePtr context;
    std::shared_ptr<QueryPlan> query_plan_ptr;
    bool has_final_transform = false;
};
}
