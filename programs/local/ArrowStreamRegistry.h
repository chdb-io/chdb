#pragma once

#include "chdb-internal.h"

#include <unordered_map>
#include <shared_mutex>
#include <optional>
#include <vector>

#include <base/types.h>

struct ArrowArrayStream;

namespace CHDB
{

class ArrowStreamRegistry
{
public:
    struct ArrowStreamInfo
    {
        ArrowArrayStream * stream = nullptr;
        bool is_owner = false;
    };

private:
    std::unordered_map<String, ArrowStreamInfo> registered_streams;
    mutable std::shared_mutex registry_mutex;

public:
    static ArrowStreamRegistry & instance()
    {
        static ArrowStreamRegistry instance;
        return instance;
    }

    bool registerArrowStream(const String & name, ArrowArrayStream * arrow_stream, bool is_owner)
    {
        std::unique_lock<std::shared_mutex> lock(registry_mutex);

        ArrowStreamInfo info;
        info.stream = arrow_stream;
        info.is_owner = is_owner;

        auto [iter, inserted] = registered_streams.emplace(name, std::move(info));
        return inserted;
    }

    std::optional<ArrowStreamInfo> getArrowStream(const String & name) const
    {
        std::shared_lock<std::shared_mutex> lock(registry_mutex);
        auto it = registered_streams.find(name);
        if (it != registered_streams.end())
            return it->second;
        return {};
    }

    bool unregisterArrowStream(const String & name)
    {
        std::unique_lock<std::shared_mutex> lock(registry_mutex);
        auto it = registered_streams.find(name);
        if (it != registered_streams.end())
        {
            if (it->second.is_owner && it->second.stream)
            {
                /// Clean up owned Arrow stream
                chdb_destroy_arrow_stream(it->second.stream);
            }
            registered_streams.erase(it);
            return true;
        }
        return false;
    }

    std::vector<String> listRegisteredNames() const
    {
        std::shared_lock<std::shared_mutex> lock(registry_mutex);
        std::vector<String> names;
        names.reserve(registered_streams.size());

        for (const auto& [name, info] : registered_streams)
            names.push_back(name);

        return names;
    }

    size_t size() const
    {
        std::shared_lock<std::shared_mutex> lock(registry_mutex);
        return registered_streams.size();
    }

    void clear()
    {
        std::unique_lock<std::shared_mutex> lock(registry_mutex);
        registered_streams.clear();
    }
};

}
