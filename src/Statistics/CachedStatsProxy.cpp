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

#include <Statistics/CacheManager.h>
#include <Statistics/CachedStatsProxy.h>
#include <Statistics/TypeUtils.h>
#include <Poco/SharedPtr.h>

namespace DB::Statistics
{

// No cache policy
class CachedStatsProxyDirectImpl : public CachedStatsProxy
{
public:
    CachedStatsProxyDirectImpl(const CatalogAdaptorPtr & catalog_) : catalog(catalog_) { }
    StatsData get(const StatsTableIdentifier & table_id) override
    {
        auto columns = catalog->getCollectableColumns(table_id);
        return get(table_id, true, columns);
    }

    StatsData get(const StatsTableIdentifier & table_id, bool table_info, const ColumnDescVector & columns) override
    {
        StatsData result;
        auto all_stats = catalog->readStatsData(table_id);
        if (table_info)
        {
            result.table_stats = all_stats.table_stats;
        }

        for (const auto & pr : columns)
        {
            const auto & col_name = pr.name;
            if (all_stats.column_stats.count(col_name))
            {
                result.column_stats.emplace(pr.name, all_stats.column_stats.at(col_name));
            }
            else
            {
                result.column_stats.emplace(pr.name, StatsCollection{});
            }
        }
        return result;
    }
    void put(const StatsTableIdentifier & table_id, StatsData && data) override { catalog->writeStatsData(table_id, data); }
    void drop(const StatsTableIdentifier & table_id) override { catalog->dropStatsData(table_id); }
    void dropColumns(const StatsTableIdentifier & table_id, const ColumnDescVector & cols_desc) override
    {
        catalog->dropStatsColumnData(table_id, cols_desc);
    }

private:
    CatalogAdaptorPtr catalog;
};


// cached policy
template <bool cache_only = false>
class CachedStatsProxyImpl : public CachedStatsProxy
{
public:
    CachedStatsProxyImpl(const CatalogAdaptorPtr & catalog_) : catalog(catalog_) { }
    StatsData get(const StatsTableIdentifier & table_id) override;
    StatsData get(const StatsTableIdentifier & table_id, bool table_info, const ColumnDescVector & columns) override;
    void put(const StatsTableIdentifier & table_id, StatsData && data) override;
    void drop(const StatsTableIdentifier & table_id) override;
    void dropColumns(const StatsTableIdentifier & table_id, const ColumnDescVector & cols_desc) override;

private:
    CatalogAdaptorPtr catalog;
};


template <bool cache_only>
StatsData CachedStatsProxyImpl<cache_only>::get(const StatsTableIdentifier & table_id)
{
    auto columns = catalog->getCollectableColumns(table_id);
    return get(table_id, true, columns);
}

template <typename T, typename... Args>
static Poco::SharedPtr<T> makePocoShared(Args &&... args)
{
    return Poco::SharedPtr<T>(new T(std::forward<Args>(args)...));
}

template <bool cache_only>
StatsData CachedStatsProxyImpl<cache_only>::get(const StatsTableIdentifier & table_id, bool table_info, const ColumnDescVector & columns)
{
    StatsData result;
    auto & cache = Statistics::CacheManager::instance();
    std::optional<StatsData> all_stats_opt;

    auto fetch_data = [&]() {
        if (!all_stats_opt.has_value())
        {
            // read all data once instead of repeatedly read
            all_stats_opt = catalog->readStatsData(table_id);
        }
    };

    if (table_info)
    {
        // collect table stats
        auto col_name = ""; // empty string represent table info
        auto key = std::make_pair(table_id.getUniqueKey(), col_name);
        auto item_ptr = cache.get(key);
        if (!item_ptr)
        {
            if constexpr (cache_only)
            {
                item_ptr = makePocoShared<StatsCollection>();
            }
            else
            {
                fetch_data();
                item_ptr = makePocoShared<StatsCollection>(all_stats_opt.value().table_stats);
                // SharedPtr is Poco's implementation of a shared pointer and
                // it will only free the memory when the counter reaches 0
                // coverity[double_free]
                cache.update(key, item_ptr);
            }
        }
        result.table_stats = *item_ptr;
    }

    for (auto & pr : columns)
    {
        auto & col_name = pr.name;
        auto key = std::make_pair(table_id.getUniqueKey(), col_name);
        auto item_ptr = cache.get(key);

        if (!item_ptr)
        {
            if constexpr (cache_only)
            {
                item_ptr = makePocoShared<StatsCollection>();
            }
            else
            {
                fetch_data();
                if (all_stats_opt.value().column_stats.count(col_name))
                {
                    item_ptr = makePocoShared<StatsCollection>(all_stats_opt.value().column_stats.at(col_name));
                }
                else
                {
                    item_ptr = makePocoShared<StatsCollection>();
                }
                cache.update(key, item_ptr);
            }
        }
        // SharedPtr is Poco's implementation of a shared pointer and
        // it will only free the memory when the counter reaches 0
        // coverity[double_free]
        result.column_stats.emplace(pr.name, *item_ptr);
    }
    return result;
}

template <bool cache_only>
void CachedStatsProxyImpl<cache_only>::put(const StatsTableIdentifier & table_id, StatsData && data)
{
    // each server has its own cache
    // to clear the cache, we need to send a request to each server
    // so we don't clear the cache here
    if (cache_only)
    {
        return;
    }

    catalog->writeStatsData(table_id, data);
}

template <bool cache_only>
void CachedStatsProxyImpl<cache_only>::drop(const StatsTableIdentifier & table_id)
{
    // each server has its own cache
    // to clear the cache, we need to send a request to each server
    // so we don't clear the cache here
    if (cache_only)
    {
        return;
    }

    catalog->dropStatsData(table_id);
}

template <bool cache_only>
void CachedStatsProxyImpl<cache_only>::dropColumns(
    const DB::Statistics::StatsTableIdentifier & table_id, const DB::Statistics::ColumnDescVector & cols_desc)
{
    // each server has its own cache
    // to clear the cache, we need to send a request to each server
    // so we don't clear the cache here
    if (cache_only)
    {
        return;
    }

    catalog->dropStatsColumnData(table_id, cols_desc);
}


// dispatcher
CachedStatsProxyPtr createCachedStatsProxy(const CatalogAdaptorPtr & catalog, StatisticsCachePolicy policy)
{
    // when enable_memory_catalog is true, we won't use cache
    if (catalog->getSettingsRef().enable_memory_catalog)
    {
        if (policy != StatisticsCachePolicy::Default)
            throw Exception("memory catalog don't support cache policy", ErrorCodes::BAD_ARGUMENTS);

        return std::make_unique<CachedStatsProxyDirectImpl>(catalog);
    }

    if (policy == StatisticsCachePolicy::Default)
    {
        return std::make_unique<CachedStatsProxyImpl<false>>(catalog);
    }
    else if (policy == StatisticsCachePolicy::Catalog)
    {
        return std::make_unique<CachedStatsProxyDirectImpl>(catalog);
    }
    else if (policy == StatisticsCachePolicy::Cache)
    {
        return std::make_unique<CachedStatsProxyImpl<true>>(catalog);
    }
    else
    {
        throw Exception("Unknown statistics cache policy", ErrorCodes::LOGICAL_ERROR);
    }
}


} // namespace DB
