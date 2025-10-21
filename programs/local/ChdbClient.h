#pragma once

#include <Client/ClientBase.h>
#include <Client/LocalConnection.h>
#include <Interpreters/Session.h>
#include <Common/Config/ConfigProcessor.h>
#include "QueryResult.h"

#include <memory>

namespace DB
{
class EmbeddedServer;

/**
 * ChdbClient - Client for executing queries in chDB
 *
 * Designed for chDB's embedded use case and inherits from ClientBase
 * to reuse all query execution logic.
 * Each client has its own LocalConnection.
 */
class ChdbClient : public ClientBase
{
public:
    explicit ChdbClient(EmbeddedServer & server);
    ~ChdbClient() override;

    CHDB::QueryResultPtr executeMaterializedQuery(const char * query, size_t query_len, const char * format, size_t format_len);

    CHDB::QueryResultPtr executeStreamingInit(const char * query, size_t query_len, const char * format, size_t format_len);

    CHDB::QueryResultPtr executeStreamingIterate(void * streaming_result, bool is_canceled = false);

    void cancelStreamingQuery(void * streaming_result);

    bool hasStreamingQuery() const { return streaming_query_context != nullptr; }

    size_t getStorageRowsRead() const;
    size_t getStorageBytesRead() const;

protected:
    void connect() override;
    Poco::Util::LayeredConfiguration & getClientConfiguration() override;
    void processError(std::string_view query) const override;
    String getName() const override { return "chdb"; }
    bool isEmbeeddedClient() const override { return true; }

    void printHelpMessage(const OptionsDescription &) override {}
    void addExtraOptions(OptionsDescription &) override {}
    void processOptions(const OptionsDescription &, const CommandLineOptions &,
                       const std::vector<Arguments> &, const std::vector<Arguments> &) override {}
    void processConfig() override {}
    void setupSignalHandler() override {}

private:
    void cleanup();
    bool parseQueryTextWithOutputFormat(const String & query, const String & format);

    EmbeddedServer & server;
    std::unique_ptr<Session> session;
    ConfigurationPtr configuration;
    Poco::AutoPtr<Poco::Util::LayeredConfiguration> layered_configuration;
    std::unique_ptr<ReadBufferFromFile> input;
};

} // namespace DB
