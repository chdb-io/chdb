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

#include <QueryPlan/BuildQueryPipelineSettings.h>
#include <Core/Settings.h>
#include <Interpreters/ExpressionActions.h>
#include <Interpreters/Context.h>

namespace DB
{

BuildQueryPipelineSettings BuildQueryPipelineSettings::fromSettings(const Settings & from)
{
    BuildQueryPipelineSettings settings;
    settings.actions_settings = ExpressionActionsSettings::fromSettings(from, CompileExpressions::yes);
    //In all of its usage, there is no use of an uninitialized variable
    //settings.distributed_settings.coordinator_address.port will be initialized when it's used
    //coverity[uninit_use]
    return settings;
}

BuildQueryPipelineSettings BuildQueryPipelineSettings::fromContext(ContextPtr from)
{
    auto settings = fromSettings(from->getSettingsRef());
    settings.context = from;
    return settings;
}

BuildQueryPipelineSettings BuildQueryPipelineSettings::fromPlanSegment(PlanSegment * plan_segment, ContextPtr context, bool is_explain)
{
    auto settings = fromContext(context);
    // settings.distributed_settings = DistributedPipelineSettings::fromPlanSegment(plan_segment);
    // settings.distributed_settings.is_explian = is_explain;
    settings.context = context;
    return settings;
}

}
