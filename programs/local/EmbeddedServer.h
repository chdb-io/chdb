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
class EmbeddedServer : public Poco::Util::Application, public IHints<2>, public Loggers
{
public:
    EmbeddedServer() = default;

    ~EmbeddedServer() override;

    void initialize(Poco::Util::Application & self) override;

    int main(const std::vector<String> & /*args*/) override;

private:
    void tryInitPath();
    void setupUsers();
    void cleanup();
    void processConfig();
    void applyCmdOptions(ContextMutablePtr context);
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
