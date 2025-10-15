#include <ChdbClient.h>
#include <chdb-internal.h>
#include <Interpreters/Context.h>
#include <Interpreters/Session.h>
#include <base/getFQDNOrHostName.h>
#include <Common/Config/ConfigHelper.h>
#include <Common/Exception.h>
#include <EmbeddedServer.h>

#if USE_PYTHON
#include <PythonTableCache.h>
#endif

namespace DB
{

namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
    extern const int LOGICAL_ERROR;
}

ChdbClient::ChdbClient(EmbeddedServer & server)
    : ClientBase()
    , server(server)
{
    session = std::make_unique<Session>(server.getGlobalContext(), ClientInfo::Interface::LOCAL);
    global_context = session->makeSessionContext();
    global_context->setCurrentDatabase("default");
    global_context->setApplicationType(Context::ApplicationType::LOCAL);
    configuration = ConfigHelper::createEmpty();
    layered_configuration = new Poco::Util::LayeredConfiguration();
    layered_configuration->addWriteable(configuration, 0);
    server_display_name = "chDB-embedded";
    query_processing_stage = QueryProcessingStage::Enum::Complete;
    is_interactive = false;
    ignore_error = false;
    echo_queries = false;
    print_stack_trace = false;
}

ChdbClient::~ChdbClient()
{
    cleanup();
    resetQueryOutputVector();
}

void ChdbClient::cleanup()
{
    try
    {
        if (streaming_query_context && streaming_query_context->streaming_result)
            CHDB::cancelStreamQuery(this, streaming_query_context->streaming_result);
        streaming_query_context.reset();
        connection.reset();
        client_context.reset();
    }
    catch (...)
    {
        tryLogCurrentException(__PRETTY_FUNCTION__);
    }
}

void ChdbClient::connect()
{
    connection_parameters = ConnectionParameters::createForEmbedded(
        session->sessionContext()->getUserName(),
        "default");
    connection = LocalConnection::createConnection(
        connection_parameters,
        std::move(session),
        std_in.get(),
        false,
        false,
        server_display_name);
        connection->setDefaultDatabase("default");
}

Poco::Util::LayeredConfiguration & ChdbClient::getClientConfiguration()
{
    chassert(layered_configuration);
    return *layered_configuration;
}

void ChdbClient::processError(std::string_view) const
{
    if (server_exception)
        server_exception->rethrow();
    if (client_exception)
        client_exception->rethrow();
}

size_t ChdbClient::getStorageRowsRead() const
{
    if (connection)
    {
        auto * local_connection = static_cast<LocalConnection *>(connection.get());
        return local_connection->getCHDBProgress().read_rows;
    }
    return 0;
}

size_t ChdbClient::getStorageBytesRead() const
{
    if (connection)
    {
        auto * local_connection = static_cast<LocalConnection *>(connection.get());
        return local_connection->getCHDBProgress().read_bytes;
    }
    return 0;
}

CHDB::QueryResultPtr ChdbClient::executeMaterializedQuery(
    const char * query, size_t query_len,
    const char * format, size_t format_len)
{
    String query_str(query, query_len);
    String format_str(format, format_len);

    try
    {
        if (!parseQueryTextWithOutputFormat(query_str, format_str))
        {
            return std::make_unique<CHDB::MaterializedQueryResult>(getErrorMsg());
        }

        size_t storage_rows_read = getStorageRowsRead();
        size_t storage_bytes_read = getStorageBytesRead();

        return std::make_unique<CHDB::MaterializedQueryResult>(
            CHDB::ResultBuffer(stealQueryOutputVector()),
            getElapsedTime(),
            getProcessedRows(),
            getProcessedBytes(),
            storage_rows_read,
            storage_bytes_read);
    }
    catch (const Exception & e)
    {
        return std::make_unique<CHDB::MaterializedQueryResult>(getExceptionMessage(e, false));
    }
    catch (...)
    {
        return std::make_unique<CHDB::MaterializedQueryResult>(getCurrentExceptionMessage(true));
    }
}

CHDB::QueryResultPtr ChdbClient::executeStreamingInit(
    const char * query, size_t query_len,
    const char * format, size_t format_len)
{
    String query_str(query, query_len);
    String format_str(format, format_len);

    try
    {
        streaming_query_context = std::make_shared<StreamingQueryContext>();
        if (!parseQueryTextWithOutputFormat(query_str, format_str))
        {
            streaming_query_context.reset();
            return std::make_unique<CHDB::StreamQueryResult>(getErrorMsg());
        }
        streaming_query_context->streaming_result = nullptr;
        return std::make_unique<CHDB::StreamQueryResult>();
    }
    catch (const Exception & e)
    {
        streaming_query_context.reset();
        return std::make_unique<CHDB::StreamQueryResult>(getExceptionMessage(e, false));
    }
    catch (...)
    {
        streaming_query_context.reset();
        return std::make_unique<CHDB::StreamQueryResult>(getCurrentExceptionMessage(true));
    }
}

/** 
CHDB::QueryResultPtr ChdbClient::executeStreamingIterate(void * streaming_result, bool is_canceled)
{
    if (!streaming_query_context)
        return std::make_unique<CHDB::MaterializedQueryResult>("No active streaming query");

    // Handle cancellation
    if (is_canceled)
    {
        streaming_query_context.reset();
#if USE_PYTHON
        if (connection)
        {
            auto * local_connection = static_cast<LocalConnection *>(connection.get());
            local_connection->resetQueryContext();
        }
        CHDB::PythonTableCache::clear();
#endif
        return std::make_unique<CHDB::MaterializedQueryResult>(nullptr, 0.0, 0, 0, 0, 0);
    }

    try
    {
        // TODO: Implement proper streaming using ClientBase's streaming methods
        // For now, this is a placeholder that returns end-of-stream
        // In the future, we should:
        // 1. Use processStreamingQuery() or similar ClientBase method
        // 2. Fetch data in batches
        // 3. Return each batch until end-of-stream

        bool is_end = true;

        if (is_end)
        {
            // End of stream reached, cleanup
            streaming_query_context.reset();
#if USE_PYTHON
            if (connection)
            {
                auto * local_connection = static_cast<LocalConnection *>(connection.get());
                local_connection->resetQueryContext();
            }
            CHDB::PythonTableCache::clear();
#endif
        }

        // Get statistics
        size_t storage_rows_read = getStorageRowsRead();
        size_t storage_bytes_read = getStorageBytesRead();

        // Return batch result
        return std::make_unique<CHDB::MaterializedQueryResult>(
            CHDB::ResultBuffer(stealQueryOutputVector()),
            getElapsedTime(),
            getProcessedRows(),
            getProcessedBytes(),
            storage_rows_read,
            storage_bytes_read);
    }
    catch (const Exception & e)
    {
        streaming_query_context.reset();
        return std::make_unique<CHDB::MaterializedQueryResult>(getExceptionMessage(e, false));
    }
    catch (...)
    {
        streaming_query_context.reset();
        return std::make_unique<CHDB::MaterializedQueryResult>(getCurrentExceptionMessage(true));
    }
}

void ChdbClient::cancelStreamingQuery(void * streaming_result)
{
    if (streaming_query_context)
    {
        streaming_query_context.reset();
#if USE_PYTHON
        if (connection)
        {
            auto * local_connection = static_cast<LocalConnection *>(connection.get());
            local_connection->resetQueryContext();
        }
        CHDB::PythonTableCache::clear();
#endif
    }
}
*/

} // namespace DB
