#pragma once

#include <Client/ClientBase.h>
#include <Client/LocalConnection.h>
#include <Interpreters/Session.h>
#include <Common/Config/ConfigProcessor.h>
#include "QueryResult.h"

#include <memory>
#include <mutex>

namespace DB
{
class EmbeddedServer;
using EmbeddedServerPtr = std::shared_ptr<EmbeddedServer>;

/**
 * ChdbClient - Client for executing queries in chDB
 *
 * Designed for chDB's embedded use case and inherits from ClientBase
 * to reuse all query execution logic.
 * Each client has its own LocalConnection.
 * Holds a shared_ptr to EmbeddedServer to ensure it stays alive while client exists.
 */
class ChdbClient : public ClientBase
{
public:
    static std::unique_ptr<ChdbClient> create(EmbeddedServerPtr server_ptr = nullptr);

    explicit ChdbClient(EmbeddedServerPtr server_ptr);
    ~ChdbClient() override;

    CHDB::QueryResultPtr executeMaterializedQuery(const char * query, size_t query_len, const char * format, size_t format_len);

    CHDB::QueryResultPtr executeStreamingInit(const char * query, size_t query_len, const char * format, size_t format_len);

    CHDB::QueryResultPtr executeStreamingIterate(void * streaming_result, bool is_canceled = false);

    void cancelStreamingQuery(void * streaming_result);

    bool hasStreamingQuery() const;

    size_t getStorageRowsRead() const;
    size_t getStorageBytesRead() const;

#if USE_PYTHON
    void findQueryableObjFromPyCache(const String & query_str) const;
#endif

protected:
    void connect() override;
    Poco::Util::LayeredConfiguration & getClientConfiguration() override;
    void processError(std::string_view query) const override;
    String getName() const override { return "chdb"; }
    bool isEmbeeddedClient() const override { return false; }

    void printHelpMessage(const OptionsDescription &) override {}
    void addExtraOptions(OptionsDescription &) override {}
    void processOptions(const OptionsDescription &, const CommandLineOptions &,
                       const std::vector<Arguments> &, const std::vector<Arguments> &) override {}
    void processConfig() override {}
    void setupSignalHandler() override {}

private:
    void cleanup();
    bool parseQueryTextWithOutputFormat(const String & query, const String & format);

    EmbeddedServerPtr server;
    std::unique_ptr<Session> session;
    ConfigurationPtr configuration;
    Poco::AutoPtr<Poco::Util::LayeredConfiguration> layered_configuration;
    std::unique_ptr<ReadBufferFromFile> input;
#if USE_PYTHON
    std::shared_ptr<CHDB::PythonTableCache> python_table_cache;
#endif
    mutable std::mutex client_mutex;
};

} // namespace DB
