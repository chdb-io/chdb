#include <Statistics/CachedStatsProxy.h>
#include <Statistics/CatalogAdaptor.h>
#include <Statistics/StatsTableBasic.h>
#include <Storages/StorageDistributed.h>

namespace DB::ErrorCodes
{
extern const int NOT_IMPLEMENTED;
extern const int LOGICAL_ERROR;
extern const int UNKNOWN_TABLE;
extern const int UNKNOWN_DATABASE;
}


namespace DB::Statistics
{
void dropStatsTable(ContextPtr context, const StatsTableIdentifier & table, StatisticsCachePolicy cache_policy, bool throw_exception)
{
    try
    {
        auto catalog = createCatalogAdaptor(context);
        catalog->checkHealth(/*is_write=*/true);
        auto proxy = createCachedStatsProxy(catalog, cache_policy);

        proxy->drop(table);
        catalog->invalidateClusterStatsCache(table);
    }
    catch (...)
    {
        if (throw_exception)
            throw;
    }
}
void dropStatsDatabase(ContextPtr context, const String & db, StatisticsCachePolicy cache_policy, bool throw_exception)
{
    try
    {
        auto catalog = createCatalogAdaptor(context);
        catalog->checkHealth(/*is_write=*/true);
        std::vector<StatsTableIdentifier> tables;
        auto proxy = createCachedStatsProxy(catalog, cache_policy);

        tables = catalog->getAllTablesID(db);
        // TODO: invalidate them in batch
        for (auto & table : tables)
        {
            proxy->drop(table);
            catalog->invalidateClusterStatsCache(table);
        }
    }
    catch (...)
    {
        if (throw_exception)
            throw;
    }
}

void dropStatsColumns(
    ContextPtr context,
    const StatsTableIdentifier & table,
    const std::vector<String> & columns,
    StatisticsCachePolicy cache_policy,
    bool throw_exception)
{
    try
    {
        auto catalog = createCatalogAdaptor(context);
        catalog->checkHealth(/*is_write=*/true);
        auto proxy = createCachedStatsProxy(catalog, cache_policy);
        auto cols_desc = filterCollectableColumns(catalog->getCollectableColumns(table), columns, true);
        proxy->dropColumns(table, cols_desc);
        catalog->invalidateClusterStatsCache(table);
    }
    catch (...)
    {
        if (throw_exception)
            throw;
    }
}

bool shouldDrop(const StoragePtr & ptr)
{
    if (ptr && ptr->as<StorageDistributed>())
    {
        return true;
    }
    return false;
}

}
