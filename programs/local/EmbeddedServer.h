#pragma once

#include <Client/ClientApplicationBase.h>
#include <Client/LocalConnection.h>

#include <Core/ServerSettings.h>
#include <Interpreters/Context_fwd.h>
#include <Loggers/Loggers.h>
#include <Common/MemoryWorker.h>
#include <Common/StatusFile.h>

#include <filesystem>
#include <memory>
#include <optional>


namespace DB
{

/// Lightweight Application for embeeded server
/// No networking, no extra configs and working directories, no pid and status files, no dictionaries, no logging.
/// Quiet mode by default
///
/// EmbeddedServer is managed via shared_ptr by ChdbClient instances.
/// When the last ChdbClient is destroyed, the EmbeddedServer is automatically destroyed.
/// Only one EmbeddedServer instance can exist globally at a time.
class EmbeddedServer : public Poco::Util::Application, public IHints<2>, public Loggers
{
public:
    EmbeddedServer() = default;

    ~EmbeddedServer() override;

    void initialize(Poco::Util::Application & self) override;

    int main(const std::vector<String> & /*args*/) override;

    std::vector<String> getAllRegisteredNames() const override { return {}; }

    ContextMutablePtr getGlobalContext() { return global_context; }

    std::string getErrorMsg() const { return error_message_oss.str(); }

    static std::shared_ptr<EmbeddedServer> getInstance(int argc = 0, char ** argv = nullptr);

    std::string getPath() const { return db_path; }

private:
    void tryInitPath();
    void setupUsers();
    void cleanup();
    void processConfig();
    void applyCmdOptions(ContextMutablePtr context);
    void initializeWithArgs(int argc, char ** argv);
    static std::weak_ptr<EmbeddedServer> global_instance;
    static std::mutex instance_mutex;
    std::string db_path;
    ServerSettings server_settings;
    std::optional<StatusFile> status;
    std::optional<std::filesystem::path> temporary_directory_to_delete;
    std::unique_ptr<MemoryWorker> memory_worker;
    ContextMutablePtr global_context;
    String home_path;
    std::stringstream error_message_oss;
    SharedPtrContextHolder shared_context;
};
} // namespace DB
