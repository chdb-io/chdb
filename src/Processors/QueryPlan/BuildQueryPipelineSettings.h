#pragma once

#include <IO/Progress.h>
#include <Interpreters/Context.h>
#include <Interpreters/ExpressionActionsSettings.h>

#include <cstddef>


namespace DB
{

struct Settings;
class QueryStatus;
using QueryStatusPtr = std::shared_ptr<QueryStatus>;
using ContextPtr = std::shared_ptr<const Context>;

struct BuildQueryPipelineSettings
{
    ExpressionActionsSettings actions_settings;
    QueryStatusPtr process_list_element;
    ProgressCallback progress_callback = nullptr;
    ContextPtr context;

    const ExpressionActionsSettings & getActionsSettings() const { return actions_settings; }
    static BuildQueryPipelineSettings fromContext(ContextPtr from);
};

}
