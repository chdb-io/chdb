#pragma once

#include <Interpreters/Context.h>
#include <Poco/ExpireLRUCache.h>
#include <Common/HashTable/Hash.h>

namespace DB
{

namespace PlanCacheConfig
{
    constexpr UInt64 max_cache_size = 1024UL ;
    constexpr UInt64 cache_expire_time = 600; // in seconds
}


class PlanCacheManager
{
public:
    using CacheType = Poco::ExpireLRUCache<UInt128, String>;

    static void initialize(ContextPtr context);
    // for testing
    static void initialize(UInt64 max_size, std::chrono::seconds expire_time);

    static CacheType & instance()
    {
        if (!cache)
        {
            throw Exception("plan cache has to be initialized", ErrorCodes::LOGICAL_ERROR);
        }
        return *cache;
    }

    // invalidate cache on current server
    static void invalidate(const ContextPtr context);

    static UInt128 hash(const ASTPtr & query_ast, const Settings & settings);
private:
    static std::unique_ptr<CacheType> cache;
};


}
