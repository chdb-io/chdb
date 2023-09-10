#pragma once

#include <common/types.h>
#include <unordered_map>
#include <vector>

namespace DB
{

class OptimizerProfile
{
private:
    std::unordered_map<String, String> optimizer_profile_map;
    std::vector<std::pair<String, String>> rule_profile_map;
public:
    void setTime(String name, String time_str, bool is_rule = false);

    String getOptimizerProfile(bool print_detail = false);

    String getFormatTime(String name, String prefix = "-- " , String suffix = "\n");

    void clear()
    {
        optimizer_profile_map.clear();
        rule_profile_map.clear();
    }

};

}
