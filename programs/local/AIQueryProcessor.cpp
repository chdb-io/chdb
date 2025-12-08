#include "AIQueryProcessor.h"

#include "chdb-internal.h"
#include "PybindWrapper.h"

#include <pybind11/pybind11.h>
#include <pybind11/detail/non_limited_api.h>

#if USE_CLIENT_AI
#include <Client/AI/AIClientFactory.h>
#include <Client/AI/AISQLGenerator.h>
#endif

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace py = pybind11;

#if USE_CLIENT_AI

AIQueryProcessor::AIQueryProcessor(chdb_connection * connection_, const DB::AIConfiguration & config_)
    : connection(connection_), ai_config(config_)
{
}

AIQueryProcessor::~AIQueryProcessor() = default;

namespace
{
void applyEnvFallback(DB::AIConfiguration & config)
{
    if (config.api_key.empty())
    {
        if (const char * api_key = std::getenv("AI_API_KEY"))
            config.api_key = api_key;
        else if (const char * openai_key = std::getenv("OPENAI_API_KEY"))
            config.api_key = openai_key;
        else if (const char * anthropic_key = std::getenv("ANTHROPIC_API_KEY"))
            config.api_key = anthropic_key;
    }
}
}

std::string AIQueryProcessor::executeQueryForAI(const std::string & query)
{
    auto run_query = [this, &query]()
    {
        return chdb_query_n(*connection, query.data(), query.size(), "TSV", 3);
    };

    chdb_result * result = nullptr;
    {
        chassert(py::gil_check());
        py::gil_scoped_release release;
        result = run_query();
    }

    const auto & error_msg = CHDB::chdb_result_error_string(result);
    if (!error_msg.empty())
    {
        std::string msg_copy(error_msg);
        chdb_destroy_query_result(result);
        throw std::runtime_error(msg_copy);
    }

    std::string data(chdb_result_buffer(result), chdb_result_length(result));
    chdb_destroy_query_result(result);
    return data;
}

void AIQueryProcessor::initializeGenerator()
{
    if (generator)
        return;

    applyEnvFallback(ai_config);

    if (ai_config.api_key.empty())
        throw std::runtime_error("AI SQL generator is not configured. Provide ai_api_key (or set OPENAI_API_KEY/ANTHROPIC_API_KEY) when creating the connection or session.");

    auto ai_result = DB::AIClientFactory::createClient(ai_config);

    if (ai_result.no_configuration_found || !ai_result.client.has_value())
        throw std::runtime_error("AI SQL generator is not configured. Provide ai_api_key (or set OPENAI_API_KEY/ANTHROPIC_API_KEY) when creating the connection or session.");

    auto query_executor = [this](const std::string & query_text) { return executeQueryForAI(query_text); };
    generator = std::make_unique<DB::AISQLGenerator>(ai_config, std::move(ai_result.client.value()), query_executor, std::cerr);
}

std::string AIQueryProcessor::generateSQLFromPrompt(const std::string & prompt)
{
    initializeGenerator();

    if (!generator)
        throw std::runtime_error("AI SQL generator is not configured. Provide ai_api_key (or set OPENAI_API_KEY/ANTHROPIC_API_KEY) when creating the connection or session.");

    std::string sql;
    {
        chassert(py::gil_check());
        py::gil_scoped_release release;
        sql = generator->generateSQL(prompt);
    }

    if (sql.empty())
        throw std::runtime_error("AI did not return a SQL query.");

    return sql;
}

std::string AIQueryProcessor::generateSQL(const std::string & prompt)
{
    return generateSQLFromPrompt(prompt);
}

#else

AIQueryProcessor::AIQueryProcessor(chdb_connection *, const DB::AIConfiguration &) : connection(nullptr) { }
AIQueryProcessor::~AIQueryProcessor() = default;
std::string AIQueryProcessor::executeQueryForAI(const std::string &) { return {}; }
void AIQueryProcessor::initializeGenerator() { }
std::string AIQueryProcessor::generateSQLFromPrompt(const std::string &) { return {}; }
std::string AIQueryProcessor::generateSQL(const std::string &) { return {}; }

#endif
