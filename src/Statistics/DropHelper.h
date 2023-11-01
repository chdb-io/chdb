#pragma once
#include <Statistics/CatalogAdaptor.h>
#include <Statistics/StatsTableBasic.h>

namespace DB::ErrorCodes
{
extern const int NOT_IMPLEMENTED;
extern const int LOGICAL_ERROR;
extern const int UNKNOWN_TABLE;
extern const int UNKNOWN_DATABASE;
}

namespace DB::Statistics
{
void dropStatsDatabase(ContextPtr context, const String & db, StatisticsCachePolicy cache_policy, bool throw_exception);

void dropStatsTable(ContextPtr context, const StatsTableIdentifier & table, StatisticsCachePolicy cache_policy, bool throw_exception);

void dropStatsColumns(
    ContextPtr context,
    const StatsTableIdentifier & table,
    const std::vector<String> & columns,
    StatisticsCachePolicy cache_policy,
    bool throw_exception);

bool shouldDrop(const StoragePtr& ptr);

}

