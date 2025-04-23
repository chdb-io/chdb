#pragma once

#include "chdb.h"
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <boost/iostreams/detail/select.hpp>

namespace CHDB
{

struct QueryRequestBase {
    virtual ~QueryRequestBase() = default;
    virtual bool isStreaming() const = 0;
    virtual bool isIteration() const { return false; }
};

struct MaterializedQueryRequest : QueryRequestBase {
    std::string query;
    std::string format;

    bool isStreaming() const override { return false; }
};

struct StreamingInitRequest : QueryRequestBase {
    std::string query;
    std::string format;

    bool isStreaming() const override { return true; }
};

struct StreamingIterateRequest : QueryRequestBase {
    chdb_streaming_result * streaming_result = nullptr;
    bool is_canceled = false;

    bool isStreaming() const override { return true; }
    bool isIteration() const override { return true; }
};

enum class QueryType : uint8_t
{
    TYPE_MATERIALIZED = 0,
    TYPE_STREAMING_INIT = 1,
    TYPE_STREAMING_ITER = 2
};

enum class QueryResultType : uint8_t
{
    RESULT_TYPE_MATERIALIZED = 0,
    RESULT_TYPE_STREAMING = 1,
    RESULT_TYPE_NONE = 2
};

struct StreamingResultData
{
    std::string error_message;
};

struct ResultData
{
    bool is_end = false;
	QueryResultType result_type = QueryResultType::RESULT_TYPE_NONE;
    union
    {
        local_result_v2 * materialized_result = nullptr;
        chdb_streaming_result * streaming_result;
    };

    void clear()
    {
        result_type = QueryResultType::RESULT_TYPE_MATERIALIZED;
        materialized_result = nullptr;
        is_end = false;
    }

    void reset()
    {
        if (result_type == QueryResultType::RESULT_TYPE_MATERIALIZED && materialized_result)
            free_result_v2(materialized_result);
        else if (result_type == QueryResultType::RESULT_TYPE_STREAMING && streaming_result)
            chdb_destroy_result(streaming_result);

        clear();
    }
};

struct QueryQueue
{
    std::mutex mutex;
    std::condition_variable query_cv; // For query submission
    std::condition_variable result_cv; // For query result retrieval
    std::unique_ptr<QueryRequestBase> current_query;
    ResultData current_result;
    bool has_result = false;
    bool has_query = false;
    bool has_streaming_query = false;
    bool shutdown = false;
    bool cleanup_done = false;
};

}
