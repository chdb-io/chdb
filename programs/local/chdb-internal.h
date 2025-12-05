#pragma once

#include "chdb.h"
#include "QueryResult.h"

#include <memory>
#include <string>
#include <arrow/c/abi.h>

namespace DB
{
    class ChdbClient;
}

/// Connection validity check function
inline bool checkConnectionValidity(chdb_conn * connection)
{
    return connection && connection->connected;
}

extern thread_local bool chdb_memory_tracking;

namespace CHDB
{

class ChdbMemoryTrackingGuard
{
public:
    ChdbMemoryTrackingGuard()
        : previous_value(chdb_memory_tracking)
    {
        chdb_memory_tracking = true;
    }

    ~ChdbMemoryTrackingGuard()
    {
        chdb_memory_tracking = previous_value;
    }

private:
    bool previous_value;
};

std::unique_ptr<MaterializedQueryResult> pyEntryClickHouseLocal(int argc, char ** argv);

const std::string & chdb_result_error_string(chdb_result * result);

const std::string & chdb_streaming_result_error_string(chdb_streaming_result * result);

void chdb_destroy_arrow_stream(ArrowArrayStream * arrow_stream);
}
