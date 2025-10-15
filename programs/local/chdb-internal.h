#pragma once

#include "chdb.h"
#include "QueryResult.h"

#include <memory>
#include <string>

namespace DB
{
    class ChdbClient;
}

namespace CHDB
{

struct QueryRequestBase
{
    virtual ~QueryRequestBase() = default;
    virtual bool isStreaming() const = 0;
    virtual bool isIteration() const { return false; }
};

struct MaterializedQueryRequest : QueryRequestBase
{
    std::string query;
    std::string format;

    MaterializedQueryRequest() = default;
    MaterializedQueryRequest(const char * q, size_t q_len, const char * f, size_t f_len)
        : query(q, q_len)
        , format(f, f_len)
    {
    }

    bool isStreaming() const override { return false; }
};

struct StreamingInitRequest : QueryRequestBase
{
    std::string query;
    std::string format;

    StreamingInitRequest() = default;
    StreamingInitRequest(const char * q, size_t q_len, const char * f, size_t f_len)
        : query(q, q_len)
        , format(f, f_len)
    {
    }

    bool isStreaming() const override { return true; }
};

struct StreamingIterateRequest : QueryRequestBase
{
    void * streaming_result = nullptr;
    bool is_canceled = false;

    bool isStreaming() const override { return true; }
    bool isIteration() const override { return true; }
};

std::unique_ptr<MaterializedQueryResult> pyEntryClickHouseLocal(int argc, char ** argv);

void chdbCleanupConnection();

void cancelStreamQuery(DB::ChdbClient * client, void * stream_result);

const std::string & chdb_result_error_string(chdb_result * result);

const std::string & chdb_streaming_result_error_string(chdb_streaming_result * result);
}
