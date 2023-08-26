#pragma once

#include <Client/ClientBase.h>
#include <Client/LocalConnection.h>

#include <Common/StatusFile.h>
#include <Common/InterruptListener.h>
#include <Loggers/Loggers.h>
#include <Core/Settings.h>
#include <Interpreters/Context.h>

#include <filesystem>
#include <memory>
#include <optional>
#include "chdb.h"


namespace DB
{

/// Lightweight Application for clickhouse-local
/// No networking, no extra configs and working directories, no pid and status files, no dictionaries, no logging.
/// Quiet mode by default
class LocalServer : public ClientBase, public Loggers
{
public:
    LocalServer() = default;

    void initialize(Poco::Util::Application & self) override;

    int main(const std::vector<String> & /*args*/) override;

    void cleanup();

    std::vector<char> * run_chdb_query(char * query, char * format);

    void free_result(chdb_local_result * result);

    void disconnect() {
        ClientBase::uninitialize();
    }

protected:
    void connect() override;

    void processError(const String & query) const override;

    String getName() const override { return "local"; }

    void printHelpMessage(const OptionsDescription & options_description) override;

    void addOptions(OptionsDescription & options_description) override;

    void processOptions(const OptionsDescription & options_description, const CommandLineOptions & options,
                        const std::vector<Arguments> &, const std::vector<Arguments> &) override;

    void processConfig() override;
    void readArguments(int argc, char ** argv, Arguments & common_arguments, std::vector<Arguments> &, std::vector<Arguments> &) override;


    void updateLoggerLevel(const String & logs_level) override;

    void uninitialize() override {
        // Do nothing in standard uninitialize. It's called automatically at the end of main(),
        // but we don't want to uninitialize at that point. We call the original unitialize
        // in disconnect().k
    }

private:
    /** Composes CREATE subquery based on passed arguments (--structure --file --table and --input-format)
      * This query will be executed first, before queries passed through --query argument
      * Returns empty string if it cannot compose that query.
      */
    std::string getInitialCreateTableQuery();

    void tryInitPath();
    void setupUsers();

    void applyCmdOptions(ContextMutablePtr context);
    void applyCmdSettings(ContextMutablePtr context);

    std::optional<StatusFile> status;
    std::optional<std::filesystem::path> temporary_directory_to_delete;
};
}
