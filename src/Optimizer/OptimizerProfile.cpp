#include <Optimizer/OptimizerProfile.h>
#include <common/types.h>
#include <unordered_map>
#include <IO/WriteBufferFromString.h>
#include <IO/WriteHelpers.h>

namespace DB
{

void OptimizerProfile::setTime(String name, String time_str, bool is_rule)
{
    if (is_rule)
        rule_profile_map.emplace_back(name, time_str);
    else
        optimizer_profile_map[name] = time_str;
}

String OptimizerProfile::getOptimizerProfile(bool print_detail)
{
    std::ostringstream os;

    os << getFormatTime("Optimizer Total")
       << String(2, ' ') << getFormatTime("Rewrite")
       << String(2, ' ') << getFormatTime("Analyzer")
       << String(2, ' ') << getFormatTime("Planning");

    if (!print_detail)
    {
        os << String(2, ' ') << getFormatTime("Optimizer");
    }
    else
    {
        String suffix = " [" + std::to_string(rule_profile_map.size()) + "]" + "\n";
        os << String(2, ' ') << getFormatTime("Optimizer", "-- ", suffix);
        for (auto & item : rule_profile_map)
            os << std::string(4, ' ') << "-- " + item.first + " " + item.second + "\n";
    }

    os << String(2, ' ') << getFormatTime("Plan Normalize")
       << String(2, ' ') << getFormatTime("PlanSegment build");
    return os.str();
}

String OptimizerProfile::getFormatTime(String name, String prefix , String suffix)
{
    if (!optimizer_profile_map.contains(name))
        return prefix + name + suffix;
    return prefix + name + " " + optimizer_profile_map.at(name) + suffix;
}

}
