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
#include <Core/Settings.h>
namespace DB::Statistics
{
struct CollectorSettings
{
    bool collect_histogram = true;
    bool collect_floating_histogram = true;
    bool collect_floating_histogram_ndv = true;
    bool enable_sample = true;
    UInt64 sample_row_count = 40000000;
    double sample_ratio = 0.01;
    StatisticsAccurateSampleNdvMode accurate_sample_ndv = StatisticsAccurateSampleNdvMode::AUTO;
    StatisticsCachePolicy cache_policy = StatisticsCachePolicy::Default;

    CollectorSettings() { }

    explicit CollectorSettings(const Settings & settings)
    {
        // read from context Settings
        fromContextSettings(settings);
    }

    void fromContextSettings(const Settings & settings)
    {
        collect_histogram = settings.statistics_collect_histogram;
        collect_floating_histogram = settings.statistics_collect_floating_histogram;
        collect_floating_histogram_ndv = settings.statistics_collect_floating_histogram_ndv;
        enable_sample = settings.statistics_enable_sample;
        sample_row_count = settings.statistics_sample_row_count;
        sample_ratio = settings.statistics_sample_ratio;
        accurate_sample_ndv = settings.statistics_accurate_sample_ndv;
        cache_policy = settings.statistics_cache_policy;
        // other settings should be manually set
        // like if not exists
    }
};

}
