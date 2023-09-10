#include <Interpreters/ProcessorProfile.h>

namespace DB
{

GroupedProcessorProfilePtr GroupedProcessorProfile::getGroupedProfiles(ProcessorProfiles & profiles)
{
    /// input processor id -> target processor id
    std::unordered_map<ProcessorId, std::vector<ProcessorId>> dag;
    std::unordered_map<ProcessorId, ProcessorProfilePtr> profile_map;
    std::unordered_set<ProcessorId> non_root_set;

    for (auto & profile : profiles)
    {
        for (auto parents_id : profile->parent_ids)
        {
            dag[profile->id].push_back(parents_id);
            non_root_set.insert(parents_id);
        }
        profile_map[profile->id] = profile;
    }

    std::vector<ProcessorId> roots;
    /// get root input processors
    for (auto & profile : profiles)
        if (!non_root_set.contains(profile->id))
            roots.emplace_back(profile->id);

    /// Build profile group tree
    std::vector<GroupedProcessorProfilePtr> groups;
    std::unordered_map<ProcessorId, GroupedProcessorProfilePtr> profile_to_group;
    auto root = std::make_shared<GroupedProcessorProfile>(0, "input_root");
    root->visited = true;
    groups.emplace_back(root);

    for (const auto & id : roots)
    {
        auto & group_parents = root->parents;
        auto & root_profile = profile_map[id];
        if (!group_parents.contains(root_profile->processor_name))
        {
            auto parent = std::make_shared<GroupedProcessorProfile>(groups.size(), root_profile->processor_name);
            group_parents.emplace(root_profile->processor_name, parent);
            groups.emplace_back(parent);
        }
        group_parents[root_profile->processor_name]->add(id, root_profile);
        profile_to_group.emplace(id, group_parents[root_profile->processor_name]);
    }

    for (size_t i = 1; i < groups.size(); i++)
    {
        auto group = groups[i];
        if (group->visited)
            continue;
        group->visited = true;

        /// Build parent group, then add it to children.
        auto & group_parents = group->parents;
        for (const auto & processor_id : group->processor_ids)
        {
            for (auto & parent_processor_id : dag[processor_id])
            {
                const auto & parent_processor = profile_map[parent_processor_id];
                if (!parent_processor)
                    continue;

                if (!group_parents.contains(parent_processor->processor_name))
                {
                    if (profile_to_group.count(parent_processor_id))
                        group_parents.emplace(parent_processor->processor_name, profile_to_group[parent_processor_id]);
                    else
                    {
                        auto child = std::make_shared<GroupedProcessorProfile>(groups.size(), parent_processor->processor_name);
                        group_parents.emplace(parent_processor->processor_name, child);
                        groups.emplace_back(child);
                    }
                }
                group_parents[parent_processor->processor_name]->add(parent_processor_id, parent_processor);
                profile_to_group.emplace(parent_processor_id, group_parents[parent_processor->processor_name]);
            }
        }
    }

    return root;
}

void GroupedProcessorProfile::add(ProcessorId processor_id, const ProcessorProfilePtr & profile)
{
    if (processor_ids.count(processor_id))
        return;

    if (profile->step_id != -1)
        step_id = profile->step_id;
    processor_ids.emplace(processor_id);
    grouped_elapsed_us = std::max(grouped_elapsed_us, profile->elapsed_us);
    grouped_input_wait_elapsed_us = std::max(grouped_input_wait_elapsed_us, profile->input_wait_elapsed_us);
    grouped_output_wait_elapsed_us = std::max(grouped_output_wait_elapsed_us, profile->output_wait_elapsed_us);
    grouped_input_rows +=  profile->input_rows;
    grouped_input_bytes +=  profile->input_bytes;
    grouped_output_rows +=  profile->output_rows;
    grouped_output_bytes +=  profile->output_bytes;
}

StepsOperatorProfiles StepOperatorProfile::aggregateOperatorProfileToStepLevel(std::unordered_map<size_t, std::vector<GroupedProcessorProfilePtr>> & segment_profile_tree)
{
    StepsOperatorProfiles res;

    struct ProfilesList
    {
        /// input step_id -> processor profile
        std::unordered_map<int64_t, GroupedProcessorProfilePtr> input_profiles;
        std::vector<GroupedProcessorProfilePtr> output_profiles;
        std::unordered_map<size_t, GroupedProcessorProfiles> profiles_at_each_level;
    };

    for (auto & [segment_id, processor_profile_roots] : segment_profile_tree)
    {
        for (auto & processor_profile_root : processor_profile_roots)
        {
            /// step_id -> map<level, ProfilesList>
            std::unordered_map<size_t, ProfilesList> step_processor_profiles_at_each_level;

            size_t level = 0;
            std::queue<GroupedProcessorProfilePtr> q;
            std::unordered_set<ProcessorId> id_set;
            q.push(processor_profile_root);
            id_set.emplace(processor_profile_root->id);
            while (!q.empty())
            {
                size_t size = q.size();
                for (size_t i = 0; i < size; i++)
                {
                    auto processor_profile = q.front();
                    q.pop();
                    auto & current_step_id = processor_profile->step_id;
                    auto & outputs = processor_profile->parents;

                    if (current_step_id == -1 && !outputs.empty() && processor_profile->processor_name != "input_root")
                        current_step_id = outputs.begin()->second->step_id;

                    step_processor_profiles_at_each_level[current_step_id].profiles_at_each_level[level].push_back(processor_profile);
                    if (outputs.empty())
                        step_processor_profiles_at_each_level[current_step_id].output_profiles.push_back(processor_profile);

                    for (auto & output_profile : outputs)
                    {
                        if (current_step_id != output_profile.second->step_id)
                        {
                            step_processor_profiles_at_each_level[current_step_id].output_profiles.push_back(processor_profile);
                            step_processor_profiles_at_each_level[output_profile.second->step_id].input_profiles[current_step_id]
                                = output_profile.second;
                        }
                        if (!id_set.contains(output_profile.second->id))
                        {
                            q.push(output_profile.second);
                            id_set.emplace(output_profile.second->id);
                        }
                    }
                }
                level++;
            }

            for (auto & [step_id, profiles_list] : step_processor_profiles_at_each_level)
            {
                auto step_profile = std::make_shared<StepOperatorProfile>();

                for (auto & output_profile : profiles_list.output_profiles)
                {
                    step_profile->output_bytes += output_profile->grouped_output_bytes;
                    step_profile->output_rows += output_profile->grouped_output_rows;
                    step_profile->output_wait_elapsed_us = std::max(step_profile->output_wait_elapsed_us, output_profile->grouped_output_wait_elapsed_us);
                }

                for (auto & [input_step_id, input_profile] : profiles_list.input_profiles)
                {
                    step_profile->inputs_profile[input_step_id].input_rows = input_profile->grouped_input_rows;
                    step_profile->inputs_profile[input_step_id].input_bytes = input_profile->grouped_input_bytes;
                    step_profile->inputs_profile[input_step_id].input_wait_elapsed_us = input_profile->grouped_input_wait_elapsed_us;
                }

                for (auto & [_, level_profiles] : profiles_list.profiles_at_each_level)
                {
                    UInt64 sum_elapsed_us = 0;
                    for (auto & profile : level_profiles)
                        sum_elapsed_us = std::max(sum_elapsed_us, profile->grouped_elapsed_us);
                    step_profile->sum_elapsed_us += sum_elapsed_us;
                }
                res[step_id].push_back(step_profile);
            }
        }
    }
    return res;
}


StepAggregatedOperatorProfiles
AggregatedStepOperatorProfile::aggregateStepOperatorProfileBetweenWorkers(StepsOperatorProfiles & steps_operator_profiles)
{
    StepAggregatedOperatorProfiles res;
    for (auto & [step_id, step_profiles] : steps_operator_profiles)
    {
        if (step_profiles.empty())
            continue;
        auto agg_profile_ptr = std::make_shared<AggregatedStepOperatorProfile>();
        agg_profile_ptr->step_id = step_id;
        std::unordered_map<int64_t, InputProfile> inputs_profile;
        for (auto & input : step_profiles[0]->inputs_profile)
            inputs_profile[input.first] = {};

        for (auto & step_profile : step_profiles)
        {
            agg_profile_ptr->max_elapsed_us = std::max(agg_profile_ptr->max_elapsed_us, step_profile->sum_elapsed_us);
            agg_profile_ptr->max_output_wait_elapsed_us = std::max(agg_profile_ptr->max_output_wait_elapsed_us, step_profile->output_wait_elapsed_us);
            agg_profile_ptr->output_rows += step_profile->output_rows;
            agg_profile_ptr->output_bytes += step_profile->output_bytes;

            for (auto & [id, input_profile] : step_profile->inputs_profile)
            {
                inputs_profile[id].input_wait_elapsed_us = std::max(inputs_profile[id].input_wait_elapsed_us, input_profile.input_wait_elapsed_us);
                inputs_profile[id].input_rows += input_profile.input_rows;
                inputs_profile[id].input_bytes += input_profile.input_bytes;
            }
        }

        agg_profile_ptr->inputs_profile = std::move(inputs_profile);
        res[step_id] = agg_profile_ptr;
    }
    return res;
}

}
