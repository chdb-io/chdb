#include "EmbeddedServer.h"

#if USE_PYTHON
#include "TableFunctionPython.h"
#include "ChunkCollectorOutputFormat.h"
#else
#include "StorageArrowStream.h"
#include "TableFunctionArrowStream.h"
#endif
#include <Formats/FormatFactory.h>
#include <TableFunctions/TableFunctionFactory.h>

#include <filesystem>
#include <Access/AccessControl.h>
#include <AggregateFunctions/registerAggregateFunctions.h>
#include <Core/UUID.h>
#include <Databases/DatabaseAtomic.h>
#include <Databases/DatabaseFilesystem.h>
#include <Databases/DatabaseMemory.h>
#include <Databases/DatabaseOverlay.h>
#include <Databases/registerDatabases.h>
#include <Dictionaries/registerDictionaries.h>
#include <Disks/registerDisks.h>
#include <Formats/registerFormats.h>
#include <Functions/UserDefined/IUserDefinedSQLObjectsStorage.h>
#include <Functions/registerFunctions.h>
#include <IO/ReadBufferFromFile.h>
#include <IO/ReadBufferFromString.h>
#include <IO/SharedThreadPools.h>
#include <Interpreters/Cache/FileCacheFactory.h>
#include <Interpreters/DatabaseCatalog.h>
#include <Interpreters/JIT/CompiledExpressionCache.h>
#include <Interpreters/ProcessList.h>
#include <Interpreters/loadMetadata.h>
#include <Interpreters/registerInterpreters.h>
#include <Loggers/OwnFormattingChannel.h>
#include <Loggers/OwnPatternFormatter.h>
#include <Loggers/OwnSplitChannel.h>
#include <Parsers/ASTAlterQuery.h>
#include <Parsers/ASTInsertQuery.h>
#include <Storages/System/attachInformationSchemaTables.h>
#include <Storages/System/attachSystemTables.h>
#include <Storages/registerStorages.h>
#include <TableFunctions/registerTableFunctions.h>
#include <base/argsToConfig.h>
#include <base/getMemoryAmount.h>
#include <boost/program_options/options_description.hpp>
#include <sys/resource.h>
#include <Poco/Logger.h>
#include <Poco/NullChannel.h>
#include <Poco/SimpleFileChannel.h>
#include <Poco/String.h>
#include <Poco/Util/XMLConfiguration.h>
#include <Common/Config/ConfigProcessor.h>
#include <Common/Config/getLocalConfigPath.h>
#include <Common/CurrentMetrics.h>
#include <Common/ErrorHandlers.h>
#include <Common/EventNotifier.h>
#include <Common/Exception.h>
#include <Common/Macros.h>
#include <Common/NamedCollections/NamedCollectionsFactory.h>
#include <Common/PoolId.h>
#include <Common/TLDListsHolder.h>
#include <Common/ThreadPool.h>
#include <Common/ThreadStatus.h>
#include <Common/filesystemHelpers.h>
#include <Common/formatReadable.h>
#include <Common/logger_useful.h>
#include <Common/quoteString.h>

#include "config.h"

#if USE_AZURE_BLOB_STORAGE
#    include <azure/storage/common/internal/xml_wrapper.hpp>
#endif

bool chdb_embedded_server_initialized = false;
extern std::atomic<bool> g_memory_tracking_disabled;

namespace fs = std::filesystem;

namespace CurrentMetrics
{
extern const Metric MemoryTracking;
}

namespace DB
{

namespace Setting
{
extern const SettingsBool allow_introspection_functions;
extern const SettingsBool implicit_select;
extern const SettingsLocalFSReadMethod storage_file_read_method;
}

namespace ServerSetting
{
extern const ServerSettingsUInt32 allow_feature_tier;
extern const ServerSettingsDouble cache_size_to_ram_max_ratio;
extern const ServerSettingsUInt64 compiled_expression_cache_elements_size;
extern const ServerSettingsUInt64 compiled_expression_cache_size;
extern const ServerSettingsUInt64 database_catalog_drop_table_concurrency;
extern const ServerSettingsString default_database;
extern const ServerSettingsString index_mark_cache_policy;
extern const ServerSettingsUInt64 index_mark_cache_size;
extern const ServerSettingsDouble index_mark_cache_size_ratio;
extern const ServerSettingsString index_uncompressed_cache_policy;
extern const ServerSettingsUInt64 index_uncompressed_cache_size;
extern const ServerSettingsDouble index_uncompressed_cache_size_ratio;
extern const ServerSettingsString vector_similarity_index_cache_policy;
extern const ServerSettingsUInt64 vector_similarity_index_cache_size;
extern const ServerSettingsUInt64 vector_similarity_index_cache_max_entries;
extern const ServerSettingsDouble vector_similarity_index_cache_size_ratio;
extern const ServerSettingsUInt64 io_thread_pool_queue_size;
extern const ServerSettingsString mark_cache_policy;
extern const ServerSettingsUInt64 mark_cache_size;
extern const ServerSettingsDouble mark_cache_size_ratio;
extern const ServerSettingsString iceberg_metadata_files_cache_policy;
extern const ServerSettingsUInt64 iceberg_metadata_files_cache_size;
extern const ServerSettingsUInt64 iceberg_metadata_files_cache_max_entries;
extern const ServerSettingsDouble iceberg_metadata_files_cache_size_ratio;
extern const ServerSettingsUInt64 max_active_parts_loading_thread_pool_size;
extern const ServerSettingsUInt64 max_io_thread_pool_free_size;
extern const ServerSettingsUInt64 max_io_thread_pool_size;
extern const ServerSettingsUInt64 max_outdated_parts_loading_thread_pool_size;
extern const ServerSettingsUInt64 max_parts_cleaning_thread_pool_size;
extern const ServerSettingsUInt64 max_server_memory_usage;
extern const ServerSettingsDouble max_server_memory_usage_to_ram_ratio;
extern const ServerSettingsUInt64 max_thread_pool_free_size;
extern const ServerSettingsUInt64 max_thread_pool_size;
extern const ServerSettingsUInt64 max_unexpected_parts_loading_thread_pool_size;
extern const ServerSettingsUInt64 mmap_cache_size;
extern const ServerSettingsBool show_addresses_in_stack_traces;
extern const ServerSettingsUInt64 thread_pool_queue_size;
extern const ServerSettingsString uncompressed_cache_policy;
extern const ServerSettingsUInt64 uncompressed_cache_size;
extern const ServerSettingsDouble uncompressed_cache_size_ratio;
extern const ServerSettingsString primary_index_cache_policy;
extern const ServerSettingsUInt64 primary_index_cache_size;
extern const ServerSettingsDouble primary_index_cache_size_ratio;
extern const ServerSettingsUInt64 max_prefixes_deserialization_thread_pool_size;
extern const ServerSettingsUInt64 max_prefixes_deserialization_thread_pool_free_size;
extern const ServerSettingsUInt64 prefixes_deserialization_thread_pool_thread_pool_queue_size;
extern const ServerSettingsUInt64 max_format_parsing_thread_pool_size;
extern const ServerSettingsUInt64 max_format_parsing_thread_pool_free_size;
extern const ServerSettingsUInt64 format_parsing_thread_pool_queue_size;
extern const ServerSettingsUInt64 memory_worker_period_ms;
extern const ServerSettingsBool memory_worker_correct_memory_tracker;
}

namespace ErrorCodes
{
extern const int BAD_ARGUMENTS;
extern const int CANNOT_LOAD_CONFIG;
extern const int FILE_ALREADY_EXISTS;
extern const int UNKNOWN_FORMAT;
}

static void applySettingsOverridesForLocal(ContextMutablePtr context)
{
    Settings settings = context->getSettingsCopy();

    settings[Setting::allow_introspection_functions] = true;
    settings[Setting::storage_file_read_method] = LocalFSReadMethod::mmap;
    settings[Setting::implicit_select] = true;

    context->setSettings(settings);
}

EmbeddedServer::~EmbeddedServer()
{
    cleanup();
}

void EmbeddedServer::initialize(Poco::Util::Application & self)
{
    setShuttingDown(false);

    Poco::Util::Application::initialize(self);

    const char * home_path_cstr = getenv("HOME"); // NOLINT(concurrency-mt-unsafe)
    if (home_path_cstr)
        home_path = home_path_cstr;

    /// Load config files if exists
    std::string config_path;
    if (config().has("config-file"))
        config_path = config().getString("config-file");
    else if (config_path.empty() && fs::exists("config.xml"))
        config_path = "config.xml";
    else if (config_path.empty())
        config_path = getLocalConfigPath(home_path).value_or("");

    if (fs::exists(config_path))
    {
        ConfigProcessor config_processor(config_path);
        ConfigProcessor::setConfigPath(fs::path(config_path).parent_path());
        auto loaded_config = config_processor.loadConfig();
        config().add(loaded_config.configuration.duplicate(), PRIO_DEFAULT, false);
    }

    server_settings.loadSettingsFromConfig(config());

    GlobalThreadPool::initialize(
        server_settings[ServerSetting::max_thread_pool_size],
        server_settings[ServerSetting::max_thread_pool_free_size],
        server_settings[ServerSetting::thread_pool_queue_size]);

    static std::once_flag atexit_registered;
    std::call_once(atexit_registered, []
    {
        (void)std::atexit([]
        {
            g_memory_tracking_disabled.store(true, std::memory_order_relaxed);
            GlobalThreadPool::shutdown();
        });

#if USE_AZURE_BLOB_STORAGE
        /// See the explanation near the same line in Server.cpp
        GlobalThreadPool::instance().addOnDestroyCallback([] { Azure::Storage::_internal::XmlGlobalDeinitialize(); });
#endif
    });

#if defined(OS_LINUX)
    memory_worker = std::make_unique<MemoryWorker>(
        server_settings[ServerSetting::memory_worker_period_ms],
        server_settings[ServerSetting::memory_worker_correct_memory_tracker],
        /* use_cgroup */ true,
        nullptr);
    memory_worker->start();
#endif

    getIOThreadPool().initialize(
        server_settings[ServerSetting::max_io_thread_pool_size],
        server_settings[ServerSetting::max_io_thread_pool_free_size],
        server_settings[ServerSetting::io_thread_pool_queue_size]);

    const size_t active_parts_loading_threads = server_settings[ServerSetting::max_active_parts_loading_thread_pool_size];
    getActivePartsLoadingThreadPool().initialize(
        active_parts_loading_threads,
        0, // We don't need any threads one all the parts will be loaded
        active_parts_loading_threads);

    const size_t outdated_parts_loading_threads = server_settings[ServerSetting::max_outdated_parts_loading_thread_pool_size];
    getOutdatedPartsLoadingThreadPool().initialize(
        outdated_parts_loading_threads,
        0, // We don't need any threads one all the parts will be loaded
        outdated_parts_loading_threads);

    getOutdatedPartsLoadingThreadPool().setMaxTurboThreads(active_parts_loading_threads);

    const size_t unexpected_parts_loading_threads = server_settings[ServerSetting::max_unexpected_parts_loading_thread_pool_size];
    getUnexpectedPartsLoadingThreadPool().initialize(
        unexpected_parts_loading_threads,
        0, // We don't need any threads one all the parts will be loaded
        unexpected_parts_loading_threads);

    getUnexpectedPartsLoadingThreadPool().setMaxTurboThreads(active_parts_loading_threads);

    const size_t cleanup_threads = server_settings[ServerSetting::max_parts_cleaning_thread_pool_size];
    getPartsCleaningThreadPool().initialize(
        cleanup_threads,
        0, // We don't need any threads one all the parts will be deleted
        cleanup_threads);

    getDatabaseCatalogDropTablesThreadPool().initialize(
        server_settings[ServerSetting::database_catalog_drop_table_concurrency],
        0, // We don't need any threads if there are no DROP queries.
        server_settings[ServerSetting::database_catalog_drop_table_concurrency]);

    getMergeTreePrefixesDeserializationThreadPool().initialize(
        server_settings[ServerSetting::max_prefixes_deserialization_thread_pool_size],
        server_settings[ServerSetting::max_prefixes_deserialization_thread_pool_free_size],
        server_settings[ServerSetting::prefixes_deserialization_thread_pool_thread_pool_queue_size]);

    getFormatParsingThreadPool().initialize(
        server_settings[ServerSetting::max_format_parsing_thread_pool_size],
        server_settings[ServerSetting::max_format_parsing_thread_pool_free_size],
        server_settings[ServerSetting::format_parsing_thread_pool_queue_size]);
}

static DatabasePtr createMemoryDatabaseIfNotExists(ContextPtr context, const String & database_name)
{
    DatabasePtr system_database = DatabaseCatalog::instance().tryGetDatabase(database_name);
    if (!system_database)
    {
        /// TODO: add attachTableDelayed into DatabaseMemory to speedup loading
        system_database = std::make_shared<DatabaseMemory>(database_name, context);
        /// Lock the UUID before attaching the database to avoid assertion failure in addUUIDMapping
        if (UUID uuid = system_database->getUUID(); uuid != UUIDHelpers::Nil)
            DatabaseCatalog::instance().addUUIDMapping(uuid);
        DatabaseCatalog::instance().attachDatabase(database_name, system_database);
    }
    return system_database;
}

static DatabasePtr createClickHouseLocalDatabaseOverlay(const String & name_, ContextPtr context)
{
    auto overlay = std::make_shared<DatabaseOverlay>(name_, context);

    UUID default_database_uuid;

    fs::path existing_path_symlink = fs::weakly_canonical(context->getPath()) / "metadata" / "default";
    if (FS::isSymlinkNoThrow(existing_path_symlink))
        default_database_uuid = parse<UUID>(FS::readSymlink(existing_path_symlink).filename());
    else
        default_database_uuid = UUIDHelpers::generateV4();

    fs::path default_database_metadata_path
        = fs::weakly_canonical(context->getPath()) / "store" / DatabaseCatalog::getPathForUUID(default_database_uuid);

    overlay->registerNextDatabase(std::make_shared<DatabaseAtomic>(name_, default_database_metadata_path, default_database_uuid, context));
    overlay->registerNextDatabase(std::make_shared<DatabaseFilesystem>(name_, "", context));
    return overlay;
}

/// If path is specified and not empty, will try to setup server environment and load existing metadata
void EmbeddedServer::tryInitPath()
{
    std::string path;

    if (config().has("path"))
    {
        /// User-supplied path.
        path = config().getString("path");
        Poco::trimInPlace(path);

        if (path.empty())
        {
            throw Exception(
                ErrorCodes::BAD_ARGUMENTS,
                "Cannot work with empty storage path that is explicitly specified"
                " by the --path option. Please check the program options and"
                " correct the --path.");
        }
    }
    else
    {
        /// The user requested to use a temporary path - use a unique path in the system temporary directory
        /// (or in the current dir if a temporary doesn't exist)
        LoggerRawPtr log = &logger();
        std::filesystem::path parent_folder;
        std::filesystem::path default_path;

        try
        {
            /// Try to guess a tmp folder name, and check if it's a directory (throw an exception otherwise).
            parent_folder = std::filesystem::temp_directory_path();
        }
        catch (const fs::filesystem_error & e)
        {
            // The tmp folder doesn't exist? Is it a misconfiguration? Or chroot?
            LOG_DEBUG(log, "Can not get temporary folder: {}", e.what());
            parent_folder = std::filesystem::current_path();

            std::filesystem::is_directory(parent_folder); // that will throw an exception if it's not a directory
            LOG_DEBUG(log, "Will create working directory inside current directory: {}", parent_folder.string());
        }

        /// we can have another clickhouse-embedded running simultaneously, even with the same PID (for ex. - several dockers mounting the same folder)
        /// or it can be some leftovers from other clickhouse-embedded runs
        /// as we can't accurately distinguish those situations we don't touch any existent folders
        /// we just try to pick some free name for our working folder

        default_path = parent_folder / fmt::format("clickhouse-embedded-{}", UUIDHelpers::generateV4());

        if (fs::exists(default_path))
            throw Exception(
                ErrorCodes::FILE_ALREADY_EXISTS,
                "Unsuccessful attempt to set up the working directory: {} already exists.",
                default_path.string());

        /// The directory can be created lazily during the runtime.
        temporary_directory_to_delete = default_path;

        path = default_path.string();

        LOG_DEBUG(log, "Working directory will be created as needed: {}", path);

        config().setString("path", path);
    }
    fs::create_directories(path);

    global_context->setPath(fs::path(path) / "");
    DatabaseCatalog::instance().fixPath(global_context->getPath());

    global_context->setTemporaryStoragePath(fs::path(path) / "tmp" / "", 0);
    global_context->setFlagsPath(fs::path(path) / "flags" / "");

    global_context->setUserFilesPath(""); /// user's files are everywhere

    std::string user_scripts_path = config().getString("user_scripts_path", fs::path(path) / "user_scripts" / "");
    global_context->setUserScriptsPath(user_scripts_path);

    /// Set path for filesystem caches
    String filesystem_caches_path(config().getString("filesystem_caches_path", fs::path(path) / "cache" / ""));
    if (!filesystem_caches_path.empty())
        global_context->setFilesystemCachesPath(filesystem_caches_path);

    /// top_level_domains_lists
    const std::string & top_level_domains_path = config().getString("top_level_domains_path", fs::path(path) / "top_level_domains/");
    if (!top_level_domains_path.empty())
        TLDListsHolder::getInstance().parseConfig(fs::path(top_level_domains_path) / "", config());
}


void EmbeddedServer::cleanup()
{
    /// Mark that we're shutting down to prevent logging operations from
    /// crashing when Poco::Logger's internal data structures are destroyed
    /// (which can happen during Python interpreter exit).
    setShuttingDown();

    /// Clear JIT cache BEFORE shutting down context to avoid use-after-free.
#if USE_EMBEDDED_COMPILER
    try
    {
        if (auto * cache = CompiledExpressionCacheFactory::instance().tryGetCache())
            cache->clear();
    }
    catch (const std::exception & e)
    {
        std::cerr << "Exception clearing JIT cache: " << e.what() << std::endl;
    }
    catch (...) {}
#endif

    try
    {
        EventNotifier::shutdown();
    }
    catch (...) {}

    try
    {
        if (global_context)
        {
            global_context->shutdown();
            global_context.reset();
        }
    }
    catch (const std::exception & e)
    {
        /// During Python interpreter exit, mutexes may be in invalid state
        /// due to static object destruction order. This is expected and harmless.
        /// Only print error for unexpected exceptions.
        std::string_view msg = e.what();
        if (msg.find("mutex") == std::string_view::npos &&
            msg.find("Invalid argument") == std::string_view::npos)
        {
            std::cerr << "Exception in global_context->shutdown(): " << e.what() << std::endl;
        }
    }
    catch (...) {}

    try
    {
        status.reset();
    }
    catch (...) {}

    // Delete the temporary directory if needed.
    try
    {
        if (temporary_directory_to_delete)
        {
            fs::remove_all(*temporary_directory_to_delete);
            temporary_directory_to_delete.reset();
        }
    }
    catch (...) {}
}

static ConfigurationPtr getConfigurationFromXMLString(const char * xml_data)
{
    std::stringstream ss{std::string{xml_data}}; // STYLE_CHECK_ALLOW_STD_STRING_STREAM
    Poco::XML::InputSource input_source{ss};
    return {new Poco::Util::XMLConfiguration{&input_source}};
}


void EmbeddedServer::setupUsers()
{
    static const char * minimal_default_user_xml = "<clickhouse>"
                                                   "    <profiles>"
                                                   "        <default></default>"
                                                   "    </profiles>"
                                                   "    <users>"
                                                   "        <default>"
                                                   "            <password></password>"
                                                   "            <networks>"
                                                   "                <ip>::/0</ip>"
                                                   "            </networks>"
                                                   "            <profile>default</profile>"
                                                   "            <quota>default</quota>"
                                                   "            <named_collection_control>1</named_collection_control>"
                                                   "        </default>"
                                                   "    </users>"
                                                   "    <quotas>"
                                                   "        <default></default>"
                                                   "    </quotas>"
                                                   "</clickhouse>";

    ConfigurationPtr users_config;
    auto & access_control = global_context->getAccessControl();
    access_control.setNoPasswordAllowed(config().getBool("allow_no_password", true));
    access_control.setPlaintextPasswordAllowed(config().getBool("allow_plaintext_password", true));
    if (config().has("config-file") || fs::exists("config.xml"))
    {
        String config_path = config().getString("config-file", "");
        bool has_user_directories = config().has("user_directories");
        const auto config_dir = fs::path{config_path}.remove_filename().string();
        String users_config_path = config().getString("users_config", "");

        if (users_config_path.empty() && has_user_directories)
        {
            users_config_path = config().getString("user_directories.users_xml.path");
            if (fs::path(users_config_path).is_relative() && fs::exists(fs::path(config_dir) / users_config_path))
                users_config_path = fs::path(config_dir) / users_config_path;
        }

        if (users_config_path.empty())
            users_config = getConfigurationFromXMLString(minimal_default_user_xml);
        else
        {
            ConfigProcessor config_processor(users_config_path);
            const auto loaded_config = config_processor.loadConfig();
            users_config = loaded_config.configuration;
        }
    }
    else
        users_config = getConfigurationFromXMLString(minimal_default_user_xml);
    if (users_config)
    {
        global_context->setUsersConfig(users_config);
        // NamedCollectionUtils::loadIfNot();
    }
    else
        throw Exception(ErrorCodes::CANNOT_LOAD_CONFIG, "Can't load config for users");
}

int EmbeddedServer::main(const std::vector<std::string> & /*args*/)
try
{
    StackTrace::setShowAddresses(server_settings[ServerSetting::show_addresses_in_stack_traces]);
    std::cout << std::fixed << std::setprecision(3);
    std::cerr << std::fixed << std::setprecision(3);

    /// Try to increase limit on number of open files.
    {
        rlimit rlim;
        if (getrlimit(RLIMIT_NOFILE, &rlim))
            throw Poco::Exception("Cannot getrlimit");

        if (rlim.rlim_cur < rlim.rlim_max)
        {
            rlim.rlim_cur = config().getUInt("max_open_files", static_cast<unsigned>(rlim.rlim_max));
            int rc = setrlimit(RLIMIT_NOFILE, &rlim);
            if (rc != 0)
                std::cerr << fmt::format(
                    "Cannot set max number of file descriptors to {}. Try to specify max_open_files according to your system limits. "
                    "error: {}",
                    rlim.rlim_cur,
                    errnoToString())
                          << '\n';
        }
    }

    std::call_once(
        global_register_once_flag,
        []()
        {
            chdb_embedded_server_initialized = true;

            registerInterpreters();
            /// Don't initialize DateLUT
            registerFunctions();
            registerAggregateFunctions();

            registerTableFunctions();
            auto & table_function_factory = TableFunctionFactory::instance();
#if USE_PYTHON
            registerTableFunctionPython(table_function_factory);
#else
            registerTableFunctionArrowStream(table_function_factory);
#endif

            registerDatabases();
            registerStorages();
#if USE_PYTHON
            CHDB::registerDataFrameOutputFormat();
#else
            auto & storage_factory = StorageFactory::instance();
            registerStorageArrowStream(storage_factory);
#endif

            registerDictionaries();
            registerDisks(/* global_skip_access_check= */ true);
            registerFormats();
        });

    processConfig();
    /// try to load user defined executable functions, throw on error and die
    try
    {
        global_context->loadOrReloadUserDefinedExecutableFunctions(config());
    }
    catch (...)
    {
        tryLogCurrentException(&logger(), "Caught exception while loading user defined executable functions.");
        throw;
    }

#if USE_FUZZING_MODE
    runLibFuzzer();
#endif

    return Application::EXIT_OK;
}
catch (DB::Exception & e)
{
    bool need_print_stack_trace = config().getBool("stacktrace", false);
    std::cerr << getExceptionMessageForLogging(e, need_print_stack_trace, true) << std::endl;
    auto code = DB::getCurrentExceptionCode();
    return static_cast<UInt8>(code) ? code : 1;
}
catch (...)
{
    error_message_oss << DB::getCurrentExceptionMessage(true) << '\n';
    auto code = DB::getCurrentExceptionCode();
    return static_cast<UInt8>(code) ? code : 1;
}

void EmbeddedServer::processConfig()
{

    auto logging
        = (config().has("logger.console") || config().has("logger.level")
           || config().has("log-level") || config().has("logger.log"));

    auto level = config().getString("log-level", config().getString("send_logs_level", "trace"));
    config().setString("logger", "logger");
    config().setString("logger.level", logging ? level : "fatal");
    buildLoggers(config(), logger(), "clickhouse-embedded");
    shared_context = Context::createSharedHolder();
    global_context = Context::createGlobal(shared_context.get());
    global_context->makeGlobalContext();
    global_context->setApplicationType(Context::ApplicationType::LOCAL);

    tryInitPath();

    LoggerRawPtr log = &logger();

    /// Maybe useless
    if (config().has("macros"))
        global_context->setMacros(std::make_unique<Macros>(config(), "macros", log));

    /// Sets external authenticators config (LDAP, Kerberos).
    global_context->setExternalAuthenticatorsConfig(config());

    setupUsers();

    /// Limit on total number of concurrently executing queries.
    /// There is no need for concurrent queries, override max_concurrent_queries.
    global_context->getProcessList().setMaxSize(0);

    size_t max_server_memory_usage = server_settings[ServerSetting::max_server_memory_usage];
    const double max_server_memory_usage_to_ram_ratio = server_settings[ServerSetting::max_server_memory_usage_to_ram_ratio];
    const size_t physical_server_memory = getMemoryAmount();
    const size_t default_max_server_memory_usage = static_cast<size_t>(physical_server_memory * max_server_memory_usage_to_ram_ratio);

    if (max_server_memory_usage == 0)
    {
        max_server_memory_usage = default_max_server_memory_usage;
        LOG_INFO(
            log,
            "Changed setting 'max_server_memory_usage' to {}"
            " ({} available memory * {:.2f} max_server_memory_usage_to_ram_ratio)",
            formatReadableSizeWithBinarySuffix(max_server_memory_usage),
            formatReadableSizeWithBinarySuffix(physical_server_memory),
            max_server_memory_usage_to_ram_ratio);
    }
    else if (max_server_memory_usage > default_max_server_memory_usage)
    {
        max_server_memory_usage = default_max_server_memory_usage;
        LOG_INFO(
            log,
            "Lowered setting 'max_server_memory_usage' to {}"
            " because the system has little few memory. The new value was"
            " calculated as {} available memory * {:.2f} max_server_memory_usage_to_ram_ratio",
            formatReadableSizeWithBinarySuffix(max_server_memory_usage),
            formatReadableSizeWithBinarySuffix(physical_server_memory),
            max_server_memory_usage_to_ram_ratio);
    }

    total_memory_tracker.setHardLimit(max_server_memory_usage);
    total_memory_tracker.setDescription("(total)");
    total_memory_tracker.setMetric(CurrentMetrics::MemoryTracking);

    const double cache_size_to_ram_max_ratio = server_settings[ServerSetting::cache_size_to_ram_max_ratio];
    const size_t max_cache_size = static_cast<size_t>(physical_server_memory * cache_size_to_ram_max_ratio);

    String uncompressed_cache_policy = server_settings[ServerSetting::uncompressed_cache_policy];
    size_t uncompressed_cache_size = server_settings[ServerSetting::uncompressed_cache_size];
    double uncompressed_cache_size_ratio = server_settings[ServerSetting::uncompressed_cache_size_ratio];
    if (uncompressed_cache_size > max_cache_size)
    {
        uncompressed_cache_size = max_cache_size;
        LOG_DEBUG(
            log,
            "Lowered uncompressed cache size to {} because the system has limited RAM",
            formatReadableSizeWithBinarySuffix(uncompressed_cache_size));
    }
    global_context->setUncompressedCache(uncompressed_cache_policy, uncompressed_cache_size, uncompressed_cache_size_ratio);

    String mark_cache_policy = server_settings[ServerSetting::mark_cache_policy];
    size_t mark_cache_size = server_settings[ServerSetting::mark_cache_size];
    double mark_cache_size_ratio = server_settings[ServerSetting::mark_cache_size_ratio];
    if (!mark_cache_size)
        LOG_ERROR(log, "Too low mark cache size will lead to severe performance degradation.");
    if (mark_cache_size > max_cache_size)
    {
        mark_cache_size = max_cache_size;
        LOG_DEBUG(
            log, "Lowered mark cache size to {} because the system has limited RAM", formatReadableSizeWithBinarySuffix(mark_cache_size));
    }
    global_context->setMarkCache(mark_cache_policy, mark_cache_size, mark_cache_size_ratio);

    String index_uncompressed_cache_policy = server_settings[ServerSetting::index_uncompressed_cache_policy];
    size_t index_uncompressed_cache_size = server_settings[ServerSetting::index_uncompressed_cache_size];
    double index_uncompressed_cache_size_ratio = server_settings[ServerSetting::index_uncompressed_cache_size_ratio];
    if (index_uncompressed_cache_size > max_cache_size)
    {
        index_uncompressed_cache_size = max_cache_size;
        LOG_INFO(
            log,
            "Lowered index uncompressed cache size to {} because the system has limited RAM",
            formatReadableSizeWithBinarySuffix(index_uncompressed_cache_size));
    }
    global_context->setIndexUncompressedCache(
        index_uncompressed_cache_policy, index_uncompressed_cache_size, index_uncompressed_cache_size_ratio);

    String index_mark_cache_policy = server_settings[ServerSetting::index_mark_cache_policy];
    size_t index_mark_cache_size = server_settings[ServerSetting::index_mark_cache_size];
    double index_mark_cache_size_ratio = server_settings[ServerSetting::index_mark_cache_size_ratio];
    if (index_mark_cache_size > max_cache_size)
    {
        index_mark_cache_size = max_cache_size;
        LOG_INFO(
            log,
            "Lowered index mark cache size to {} because the system has limited RAM",
            formatReadableSizeWithBinarySuffix(index_mark_cache_size));
    }
    global_context->setIndexMarkCache(index_mark_cache_policy, index_mark_cache_size, index_mark_cache_size_ratio);

    String primary_index_cache_policy = server_settings[ServerSetting::primary_index_cache_policy];
    size_t primary_index_cache_size = server_settings[ServerSetting::primary_index_cache_size];
    double primary_index_cache_size_ratio = server_settings[ServerSetting::primary_index_cache_size_ratio];
    if (primary_index_cache_size > max_cache_size)
    {
        primary_index_cache_size = max_cache_size;
        LOG_INFO(
            log,
            "Lowered primary index cache size to {} because the system has limited RAM",
            formatReadableSizeWithBinarySuffix(primary_index_cache_size));
    }
    global_context->setPrimaryIndexCache(primary_index_cache_policy, primary_index_cache_size, primary_index_cache_size_ratio);

    String vector_similarity_index_cache_policy = server_settings[ServerSetting::vector_similarity_index_cache_policy];
    size_t vector_similarity_index_cache_size = server_settings[ServerSetting::vector_similarity_index_cache_size];
    size_t vector_similarity_index_cache_max_count = server_settings[ServerSetting::vector_similarity_index_cache_max_entries];
    double vector_similarity_index_cache_size_ratio = server_settings[ServerSetting::vector_similarity_index_cache_size_ratio];
    if (vector_similarity_index_cache_size > max_cache_size)
    {
        vector_similarity_index_cache_size = max_cache_size;
        LOG_INFO(
            log,
            "Lowered vector similarity index cache size to {} because the system has limited RAM",
            formatReadableSizeWithBinarySuffix(vector_similarity_index_cache_size));
    }
    global_context->setVectorSimilarityIndexCache(
        vector_similarity_index_cache_policy,
        vector_similarity_index_cache_size,
        vector_similarity_index_cache_max_count,
        vector_similarity_index_cache_size_ratio);

    size_t mmap_cache_size = server_settings[ServerSetting::mmap_cache_size];
    if (mmap_cache_size > max_cache_size)
    {
        mmap_cache_size = max_cache_size;
        LOG_INFO(
            log,
            "Lowered mmap file cache size to {} because the system has limited RAM",
            formatReadableSizeWithBinarySuffix(mmap_cache_size));
    }
    global_context->setMMappedFileCache(mmap_cache_size);

#if USE_AVRO
    String iceberg_metadata_files_cache_policy = server_settings[ServerSetting::iceberg_metadata_files_cache_policy];
    size_t iceberg_metadata_files_cache_size = server_settings[ServerSetting::iceberg_metadata_files_cache_size];
    size_t iceberg_metadata_files_cache_max_entries = server_settings[ServerSetting::iceberg_metadata_files_cache_max_entries];
    double iceberg_metadata_files_cache_size_ratio = server_settings[ServerSetting::iceberg_metadata_files_cache_size_ratio];
    if (iceberg_metadata_files_cache_size > max_cache_size)
    {
        iceberg_metadata_files_cache_size = max_cache_size;
        LOG_INFO(
            log,
            "Lowered Iceberg metadata cache size to {} because the system has limited RAM",
            formatReadableSizeWithBinarySuffix(iceberg_metadata_files_cache_size));
    }
    global_context->setIcebergMetadataFilesCache(
        iceberg_metadata_files_cache_policy,
        iceberg_metadata_files_cache_size,
        iceberg_metadata_files_cache_max_entries,
        iceberg_metadata_files_cache_size_ratio);
#endif

    /// Initialize a dummy query condition cache.
    global_context->setQueryConditionCache(DEFAULT_QUERY_CONDITION_CACHE_POLICY, 0, 0);

    /// Initialize a dummy query result cache.
    global_context->setQueryResultCache(0, 0, 0, 0);

    /// Initialize allowed tiers
    global_context->getAccessControl().setAllowTierSettings(server_settings[ServerSetting::allow_feature_tier]);

#if USE_EMBEDDED_COMPILER
    size_t compiled_expression_cache_max_size_in_bytes = server_settings[ServerSetting::compiled_expression_cache_size];
    size_t compiled_expression_cache_max_elements = server_settings[ServerSetting::compiled_expression_cache_elements_size];
    CompiledExpressionCacheFactory::instance().init(compiled_expression_cache_max_size_in_bytes, compiled_expression_cache_max_elements);
#endif

    NamedCollectionFactory::instance().loadIfNot();
    FileCacheFactory::instance().loadDefaultCaches(config(), global_context);
    applySettingsOverridesForLocal(global_context);
    applyCmdOptions(global_context);

    /// Load global settings from default_profile and system_profile.
    global_context->setDefaultProfiles(config());
    /// We load temporary database first, because projections need it.
    DatabaseCatalog::instance().initializeAndLoadTemporaryDatabase();

    std::string server_default_database = server_settings[ServerSetting::default_database];
    if (!server_default_database.empty())
    {
        DatabasePtr database = createClickHouseLocalDatabaseOverlay(server_default_database, global_context);
        if (UUID uuid = database->getUUID(); uuid != UUIDHelpers::Nil)
            DatabaseCatalog::instance().addUUIDMapping(uuid);
        DatabaseCatalog::instance().attachDatabase(server_default_database, database);
        global_context->setCurrentDatabase(server_default_database);
    }

    if (config().has("path"))
    {
        attachInformationSchema(global_context, *createMemoryDatabaseIfNotExists(global_context, DatabaseCatalog::INFORMATION_SCHEMA));
        attachInformationSchema(
            global_context, *createMemoryDatabaseIfNotExists(global_context, DatabaseCatalog::INFORMATION_SCHEMA_UPPERCASE));

        /// Attaching "automatic" tables in the system database is done after attaching the system database.
        /// Consequently, it depends on whether we load it from the path.
        /// If it is loaded from a user-specified path, we load it as usual. If not, we create it as a memory (ephemeral) database.
        bool attached_system_database = false;

        String path = global_context->getPath();

        /// Lock path directory before read
        fs::create_directories(fs::path(path));
        status.emplace(fs::path(path) / "status", StatusFile::write_full_info);

        if (fs::exists(fs::path(path) / "metadata"))
        {
            LOG_DEBUG(log, "Loading metadata from {}", path);

            if (fs::exists(std::filesystem::path(path) / "metadata" / "system.sql"))
            {
                LoadTaskPtrs load_system_metadata_tasks = loadMetadataSystem(global_context);
                waitLoad(TablesLoaderForegroundPoolId, load_system_metadata_tasks);

                attachSystemTablesServer(
                    global_context, *DatabaseCatalog::instance().tryGetDatabase(DatabaseCatalog::SYSTEM_DATABASE), false);
                attached_system_database = true;
            }

            if (!config().has("only-system-tables"))
            {
                DatabaseCatalog::instance().loadMarkedAsDroppedTables();
                DatabaseCatalog::instance().createBackgroundTasks();
                waitLoad(loadMetadata(global_context));
                DatabaseCatalog::instance().startupBackgroundTasks();
            }

            LOG_DEBUG(log, "Loaded metadata.");
        }

        if (!attached_system_database)
            attachSystemTablesServer(
                global_context, *createMemoryDatabaseIfNotExists(global_context, DatabaseCatalog::SYSTEM_DATABASE), false);

        if (fs::exists(fs::path(path) / "user_defined"))
            global_context->getUserDefinedSQLObjectsStorage().loadObjects();
    }
    else if (!config().has("no-system-tables"))
    {
        attachSystemTablesServer(global_context, *createMemoryDatabaseIfNotExists(global_context, DatabaseCatalog::SYSTEM_DATABASE), false);
        attachInformationSchema(global_context, *createMemoryDatabaseIfNotExists(global_context, DatabaseCatalog::INFORMATION_SCHEMA));
        attachInformationSchema(
            global_context, *createMemoryDatabaseIfNotExists(global_context, DatabaseCatalog::INFORMATION_SCHEMA_UPPERCASE));

        /// Create background tasks necessary for DDL operations like DROP VIEW SYNC,
        /// even in temporary mode (--path not set) without persistent storage
        DatabaseCatalog::instance().createBackgroundTasks();
        DatabaseCatalog::instance().startupBackgroundTasks();
    }
    else
    {
        /// Similarly, for other cases, create background tasks for DDL operations like
        /// DROP VIEW SYNC in temporaty mode (--path not set) without persistent storage
        DatabaseCatalog::instance().createBackgroundTasks();
        DatabaseCatalog::instance().startupBackgroundTasks();
    }

    std::string default_database = config().getString("database", server_default_database);
    if (default_database.empty())
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "default_database cannot be empty");
    global_context->setCurrentDatabase(default_database);
}

void EmbeddedServer::applyCmdOptions(ContextMutablePtr context)
{
    context->setDefaultFormat(
        config().getString(
            "output-format", config().getString("format",  "TSV")));
}

std::unique_ptr<EmbeddedServer> EmbeddedServer::global_instance;
std::mutex EmbeddedServer::instance_mutex;
size_t EmbeddedServer::client_ref_count = 0;

EmbeddedServer & EmbeddedServer::getInstance(int argc, char ** argv)
{
    std::lock_guard<std::mutex> lock(instance_mutex);

    if (global_instance)
    {
        if (argc > 0 && argv)
        {
            std::string path = ":memory:"; // Default path
            for (int i = 1; i < argc; i++)
            {
                if (strncmp(argv[i], "--path=", 7) == 0)
                {
                    path = argv[i] + 7;
                    break;
                }
            }
            if (!global_instance->db_path.empty() && global_instance->db_path != path)
            {
                throw DB::Exception(
                    ErrorCodes::BAD_ARGUMENTS,
                    "EmbeddedServer already initialized with path '{}', cannot connect with different path '{}'",
                    global_instance->db_path,
                    path);
            }
        }
        ++client_ref_count;
        return *global_instance;
    }

    global_instance = std::make_unique<EmbeddedServer>();
    try
    {
        if (argc == 0 || !argv)
        {
            const char * default_argv[] = {"chdb"};
            global_instance->initializeWithArgs(1, const_cast<char **>(default_argv));
        }
        else
        {
            global_instance->initializeWithArgs(argc, argv);
        }
    }
    catch (...)
    {
        global_instance.reset();
        throw;
    }

    client_ref_count = 1;
    return *global_instance;
}

void EmbeddedServer::releaseInstance()
{
    std::lock_guard<std::mutex> lock(instance_mutex);

    if (client_ref_count == 0)
        return;

    --client_ref_count;
    if (client_ref_count == 0)
    {
        global_instance.reset();
    }
}

void EmbeddedServer::initializeWithArgs(int argc, char ** argv)
{
    db_path = ":memory:"; // Default path
    for (int i = 1; i < argc; i++)
    {
        if (strncmp(argv[i], "--path=", 7) == 0)
        {
            db_path = argv[i] + 7;
            break;
        }
    }

    try
    {
        std::vector<std::string> args;
        for (int i = 0; i < argc; ++i)
        {
            args.push_back(argv[i]);
        }

        Poco::Util::Application::ArgVec arg_vec;
        for (const auto & arg : args)
        {
            arg_vec.push_back(arg);
        }
        argsToConfig(arg_vec, config(), 100);

        initialize(*this);
        int ret = main(args);
        if (ret != 0)
        {
            auto err_msg = getErrorMsg();
            LOG_ERROR(&logger(), "Error initializing EmbeddedServer: {}", err_msg);
            throw DB::Exception(ErrorCodes::BAD_ARGUMENTS, "Error initializing EmbeddedServer: {}", err_msg);
        }
    }
    catch (const std::exception & e)
    {
        LOG_ERROR(&Poco::Logger::get("EmbeddedServer"), "Failed to initialize EmbeddedServer: {}", e.what());
        throw;
    }
}
} // namespace DB
