#pragma once

#include <Interpreters/Context_fwd.h>
#include <Parsers/IAST_fwd.h>
#include <Storages/SelectQueryInfo.h>

namespace DB
{
// expect a simple SELECT query looks like: SELECT ... FROM merge_tree_table [PREWHERE ...] [WHERE ...]
SelectQueryInfo buildSelectQueryInfoForQuery(const ASTPtr & query, ContextPtr context);
}
