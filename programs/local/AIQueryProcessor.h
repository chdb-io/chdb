#pragma once

#include "chdb.h"
#include <Client/AI/AIConfiguration.h>
#include <Client/AI/AISQLGenerator.h>

#include <memory>
#include <string>

/// AI query processor that delegates to AISQLGenerator.
class AIQueryProcessor
{
public:
    AIQueryProcessor(chdb_connection * connection_, const DB::AIConfiguration & config_);
    ~AIQueryProcessor();

    /// Generate SQL using the configured AI provider.
    std::string generateSQL(const std::string & prompt);

private:
    chdb_connection * connection;
    std::unique_ptr<DB::AISQLGenerator> generator;
    DB::AIConfiguration ai_config;

    std::string executeQueryForAI(const std::string & query);
    std::string generateSQLFromPrompt(const std::string & prompt);
    void initializeGenerator();
};
