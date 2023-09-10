#pragma once

#include <Interpreters/SystemLog.h>
#include <Core/NamesAndTypes.h>
#include <Core/NamesAndAliases.h>
#include <Processors/IProcessor.h>

#include <chrono>

namespace DB
{

struct ProcessorProfile;
using ProcessorProfilePtr = std::shared_ptr<ProcessorProfile>;
using ProcessorProfiles = std::vector<ProcessorProfilePtr>;
using QueryProcessorProfiles = std::map<size_t, std::vector<ProcessorProfilePtr>>; // plan_segment_id -> stream_level_operator_profile
using ProcessorId = UInt64;

struct ProcessorProfile
{
    ProcessorId id{};
    String processor_name;
    std::vector<ProcessorId> parent_ids;

    int64_t step_id = -1;
    UInt64 segment_id = 0;
    String worker_address;

    /// Milliseconds spend in IProcessor::work()
    UInt64 elapsed_us{};
    /// IProcessor::NeedData
    UInt64 input_wait_elapsed_us{};
    /// IProcessor::PortFull
    UInt64 output_wait_elapsed_us{};

    size_t input_rows{};
    size_t input_bytes{};
    size_t output_rows{};
    size_t output_bytes{};

public:
    ProcessorProfile() = default;
    explicit ProcessorProfile(String name): processor_name(name) {}
};

struct GroupedProcessorProfile;
using GroupedProcessorProfilePtr = std::shared_ptr<GroupedProcessorProfile>;
using GroupedProcessorProfiles = std::vector<GroupedProcessorProfilePtr>;

struct GroupedProcessorProfile
{
    ProcessorId id{};
    String processor_name;

    int64_t step_id = -1;
    UInt64 grouped_elapsed_us{};
    UInt64 grouped_input_wait_elapsed_us{};
    UInt64 grouped_output_wait_elapsed_us{};

    UInt64 grouped_input_rows{};
    UInt64 grouped_input_bytes{};
    UInt64 grouped_output_rows{};
    UInt64 grouped_output_bytes{};

    std::unordered_set<ProcessorId> processor_ids;
    bool visited = false;
    std::unordered_map<String, GroupedProcessorProfilePtr> parents;
public:
    GroupedProcessorProfile() = default;
    explicit GroupedProcessorProfile(ProcessorId id_, String name_) : id(id_), processor_name(name_) { }
    static GroupedProcessorProfilePtr getGroupedProfiles(ProcessorProfiles & profiles);

    void add(ProcessorId processor_id, const ProcessorProfilePtr & profile);
};


struct StepOperatorProfile;
using StepOperatorProfilePtr = std::shared_ptr<StepOperatorProfile>;
using StepOperatorProfiles = std::vector<StepOperatorProfilePtr>;
using StepsOperatorProfiles = std::unordered_map<size_t, StepOperatorProfiles>; // step_id -> step_operator_profile

struct InputProfile
{
    UInt64 input_wait_elapsed_us; // max_input_wait_elapsed_us in AggregatedStepOperatorProfile
    UInt64 input_rows;
    UInt64 input_bytes;
};

struct StepOperatorProfile
{
    int64_t step_id = -1;
    UInt64 sum_elapsed_us{};

    /// input step_id -> (input_wait_elapsed_us, input_rows, input_bytes)
    std::unordered_map<int64_t, InputProfile> inputs_profile;

    UInt64 output_wait_elapsed_us{};
    UInt64 output_rows{};
    UInt64 output_bytes{};

    static StepsOperatorProfiles aggregateOperatorProfileToStepLevel(std::unordered_map<size_t, std::vector<GroupedProcessorProfilePtr>> & segment_profile_tree);
};

struct AggregatedStepOperatorProfile;
using AggregatedStepOperatorProfilePtr = std::shared_ptr<AggregatedStepOperatorProfile>;
using StepAggregatedOperatorProfiles = std::unordered_map<size_t, AggregatedStepOperatorProfilePtr>; // step_id -> aggregated profile
struct AggregatedStepOperatorProfile
{
    size_t step_id{};
    UInt64 max_elapsed_us{};

    std::unordered_map<int64_t, InputProfile> inputs_profile;

    UInt64 max_output_wait_elapsed_us{};
    UInt64 output_rows{};
    UInt64 output_bytes{};

    static StepAggregatedOperatorProfiles aggregateStepOperatorProfileBetweenWorkers(StepsOperatorProfiles & steps_operator_profiles);
    String toJSONString(size_t indent = 0) const;
};

}
