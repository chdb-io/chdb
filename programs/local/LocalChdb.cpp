#include "LocalChdb.h"
#include "chdb-internal.h"
#include "PandasDataFrameBuilder.h"
#include "ChunkCollectorOutputFormat.h"
#include "PythonImporter.h"
#include "StoragePython.h"
#include <ChdbClient.h>

#include <pybind11/detail/non_limited_api.h>
#include <pybind11/pybind11.h>
#include <Poco/String.h>
#include <Common/logger_useful.h>
#include <vector>
#if USE_JEMALLOC
#    include <Common/memory.h>
#endif

#if USE_EMBEDDED_COMPILER
#    include <Interpreters/JIT/CompiledExpressionCache.h>
#endif

#if USE_CLIENT_AI
#    include "AIQueryProcessor.h"
#endif

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;

extern bool inside_main = true;

namespace
{

std::string paramValueToString(const py::handle & value)
{
    if (value.is_none())
        return "NULL";
    if (py::isinstance<py::bool_>(value))
        return py::cast<bool>(value) ? "1" : "0";

    if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value))
    {
        std::vector<std::string> parts;
        parts.reserve(static_cast<size_t>(py::len(value)));
        for (const auto & item : value)
            parts.emplace_back(paramValueToString(item));

        std::ostringstream out;
        out << '[';
        for (size_t i = 0; i < parts.size(); ++i)
        {
            if (i)
                out << ',';
            out << parts[i];
        }
        out << ']';
        return out.str();
    }

    return py::cast<std::string>(py::str(value));
}

DB::NameToNameMap parseParametersDict(const py::dict & params)
{
    DB::NameToNameMap parsed;
    for (const auto & item : params)
    {
        const std::string key = py::cast<std::string>(item.first);
        const auto & value = item.second;
        parsed.emplace(key, paramValueToString(value));
    }
    return parsed;
}

class QueryParameterGuard
{
public:
    QueryParameterGuard(DB::ChdbClient * client_, const DB::NameToNameMap & params) : client(client_)
    {
        if (client && !params.empty())
        {
            client->setQueryParameters(params);
            applied = true;
        }
    }

    ~QueryParameterGuard()
    {
        if (client && applied)
            client->clearQueryParameters();
    }

private:
    DB::ChdbClient * client;
    bool applied = false;
};

DB::ChdbClient * getChdbClient(chdb_connection handle)
{
    auto * connection = reinterpret_cast<chdb_conn *>(handle);
    if (!checkConnectionValidity(connection))
        return nullptr;

    return static_cast<DB::ChdbClient *>(connection->server);
}

}

namespace CHDB
{
extern chdb_connection * connect_chdb_with_exception(int argc, char ** argv);
extern void cachePythonTablesFromQuery(chdb_conn * conn, const std::string & query_str);
}

const static char * CURSOR_DEFAULT_FORMAT = "JSONCompactEachRowWithNamesAndTypes";
const static size_t CURSOR_DEFAULT_FORMAT_LEN = strlen(CURSOR_DEFAULT_FORMAT);

chdb_result * queryToBuffer(
    const std::string & queryStr,
    const std::string & output_format = "CSV",
    const std::string & path = {},
    const std::string & udfPath = {},
    const DB::NameToNameMap & params = {})
{
    std::vector<std::string> argv = {"clickhouse", "--multiquery"};

    // If format is "Debug" or "debug", then we will add `--verbose` and `--log-level=trace` to argv
    if (output_format == "Debug" || output_format == "debug")
    {
        argv.push_back("--verbose");
        argv.push_back("--log-level=test");
        // Add format string
        argv.push_back("--output-format=CSV");
    }
    else
    {
        // Add format string
        argv.push_back("--output-format=" + output_format);
    }

    // If path is not empty, then we will add `--path` to argv. This is used for chdb.Session to support stateful query
    if (!path.empty())
    {
        // Add path string
        argv.push_back("--path=" + path);
    }
    // argv.push_back("--no-system-tables");
    // Add query string
    argv.push_back("--query=" + queryStr);

    // If udfPath is not empty, then we will add `--user_scripts_path` and `--user_defined_executable_functions_config` to argv
    // the path should be a one time thing, so the caller should take care of the temporary files deletion
    if (!udfPath.empty())
    {
        argv.push_back("--");
        argv.push_back("--user_scripts_path=" + udfPath);
        argv.push_back("--user_defined_executable_functions_config=" + udfPath + "/*.xml");
    }

    for (const auto & [key, value] : params)
    {
        argv.push_back("--param_" + key + "=" + value);
    }

    // Convert std::string to char*
    std::vector<char *> argv_char;
    argv_char.reserve(argv.size());
    for (auto & arg : argv)
        argv_char.push_back(const_cast<char *>(arg.c_str()));

    py::gil_scoped_release release;
    return chdb_query_cmdline(argv_char.size(), argv_char.data());
}

// Pybind11 will take over the ownership of the `query_result` object
// using smart ptr will cause early free of the object
query_result * query(
    const std::string & queryStr,
    const std::string & output_format = "CSV",
    const std::string & path = {},
    const std::string & udfPath = {},
    const py::dict & params = py::dict())
{
    return new query_result(queryToBuffer(queryStr, output_format, path, udfPath, parseParametersDict(params)));
}

// The `query_result` and `memoryview_wrapper` will hold `local_result_wrapper` with shared_ptr
memoryview_wrapper * query_result::get_memview()
{
    return new memoryview_wrapper(this->result_wrapper);
}


// Parse SQLite-style connection string
std::pair<std::string, std::map<std::string, std::string>> connection_wrapper::parse_connection_string(const std::string & conn_str)
{
    std::string path;
    std::map<std::string, std::string> params;

    if (conn_str.empty() || conn_str == ":memory:")
    {
        return {":memory:", params};
    }

    std::string working_str = conn_str;

    // Handle file: prefix
    if (working_str.starts_with("file:"))
    {
        working_str = working_str.substr(5);

        // Handle triple slash for absolute paths
        if (working_str.starts_with("///"))
        {
            working_str = working_str.substr(2); // Remove two slashes, keep one
        }
    }

    // Split path and parameters
    auto query_pos = working_str.find('?');
    if (query_pos != std::string::npos)
    {
        path = working_str.substr(0, query_pos);
        std::string params_str = working_str.substr(query_pos + 1);

        // Parse parameters
        std::istringstream params_stream(params_str);
        std::string param;
        while (std::getline(params_stream, param, '&'))
        {
            auto eq_pos = param.find('=');
            if (eq_pos != std::string::npos)
            {
                std::string key = param.substr(0, eq_pos);
                std::string value = param.substr(eq_pos + 1);
                params[key] = value;
            }
            else if (!param.empty())
            {
                // Handle parameters without values
                params[param] = "";
            }
        }
        // Handle udf_path
        // add user_scripts_path and user_defined_executable_functions_config to params
        // these two parameters need "--" as prefix
        if (params.contains("udf_path"))
        {
            std::string udf_path = params["udf_path"];
            if (!udf_path.empty())
            {
                params["--"] = "";
                params["user_scripts_path"] = udf_path;
                params["user_defined_executable_functions_config"] = udf_path + "/*.xml";
            }
            // remove udf_path from params
            params.erase("udf_path");
        }
    }
    else
    {
        path = working_str;
    }

    // Convert relative paths to absolute
    if (!path.empty() && path[0] != '/' && path != ":memory:")
    {
        std::error_code ec;
        path = std::filesystem::absolute(path, ec).string();
        if (ec)
        {
            throw std::runtime_error("Failed to resolve path: " + path);
        }
    }

    return {path, params};
}

std::vector<std::string>
connection_wrapper::build_clickhouse_args(const std::string & path, const std::map<std::string, std::string> & params)
{
    std::vector<std::string> argv = {"clickhouse"};

    if (path != ":memory:")
    {
        argv.push_back("--path=" + path);
    }

    // Map SQLite parameters to ClickHouse arguments
    for (const auto & [key, value] : params)
    {
        if (key == "mode")
        {
            if (value == "ro")
            {
                is_readonly = true;
                argv.push_back("--readonly=1");
            }
        }
        else if (key == "--")
        {
            // Handle special parameters "--"
            argv.push_back("--");
        }
        else if (value.empty())
        {
            // Handle parameters without values (like ?withoutarg)
            argv.push_back("--" + key);
        }
        else
        {
            argv.push_back("--" + key + "=" + value);
        }
    }

    return argv;
}

connection_wrapper::connection_wrapper(const std::string & conn_str)
{
    is_readonly = false;
    auto [path, params] = parse_connection_string(conn_str);

#if USE_CLIENT_AI
    applyAIParams(params);
#endif

    auto argv = build_clickhouse_args(path, params);
    std::vector<char *> argv_char;
    argv_char.reserve(argv.size());
    for (auto & arg : argv)
    {
        argv_char.push_back(const_cast<char *>(arg.c_str()));
    }

    conn = CHDB::connect_chdb_with_exception(argv_char.size(), argv_char.data());
    db_path = path;
    is_memory_db = (path == ":memory:");
}

connection_wrapper::~connection_wrapper()
{
    py::gil_scoped_release release;
    chdb_close_conn(conn);
}

void connection_wrapper::close()
{
    {
        py::gil_scoped_release release;
        chdb_close_conn(conn);
    }
    // Ensure that if a new connection is created before this object is destroyed that we don't try to close it.
    conn = nullptr;
}

cursor_wrapper * connection_wrapper::cursor()
{
    return new cursor_wrapper(this);
}

void connection_wrapper::commit()
{
    // do nothing
}

query_result * connection_wrapper::query(const std::string & query_str, const std::string & format, const py::dict & params)
{
    if (Poco::toLower(format) == "dataframe")
        throw std::runtime_error("Unsupported output format dataframe, please use 'query_df' function");

    const auto parsed_params = parseParametersDict(params);
    auto * client = getChdbClient(*conn);
    QueryParameterGuard guard(client, parsed_params);

    CHDB::cachePythonTablesFromQuery(reinterpret_cast<chdb_conn *>(*conn), query_str);
    py::gil_scoped_release release;

    auto * result = chdb_query_n(*conn, query_str.data(), query_str.size(), format.data(), format.size());

    const auto & error_msg = CHDB::chdb_result_error_string(result);
    if (!error_msg.empty())
    {
        std::string msg_copy(error_msg);
        chdb_destroy_query_result(result);
        throw std::runtime_error(msg_copy);
    }

    return new query_result(result, false);
}

py::object connection_wrapper::query_df(const std::string & query_str, const py::dict & params)
{
    static const std::string format = "dataframe";

    chdb_result * result = nullptr;
    CHDB::ChunkQueryResult * chunk_result = nullptr;

    const auto parsed_params = parseParametersDict(params);
    auto * client = getChdbClient(*conn);
    QueryParameterGuard guard(client, parsed_params);

    CHDB::cachePythonTablesFromQuery(reinterpret_cast<chdb_conn *>(*conn), query_str);

    {
        py::gil_scoped_release release;

        result = chdb_query_n(*conn, query_str.data(), query_str.size(), format.data(), format.size());

        const auto & error_msg = CHDB::chdb_result_error_string(result);
        if (!error_msg.empty())
        {
            std::string msg_copy(error_msg);
            chdb_destroy_query_result(result);
            throw std::runtime_error(msg_copy);
        }

        if (!(chunk_result = dynamic_cast<CHDB::ChunkQueryResult *>(reinterpret_cast<CHDB::QueryResult *>(result))))
            throw std::runtime_error("Expected ChunkQueryResult for dataframe format");
    }

    CHDB::PandasDataFrameBuilder builder(*chunk_result);
    auto df = builder.getDataFrame();
    chdb_destroy_query_result(result);

    return df;
}

std::string connection_wrapper::generate_sql(const std::string & prompt)
{
#if USE_CLIENT_AI
    if (!ai_config.has_value())
        ai_config = DB::AIConfiguration{};

    try
    {
        if (!ai_processor)
            ai_processor = std::make_unique<AIQueryProcessor>(conn, *ai_config);
        return ai_processor->generateSQL(prompt);
    }
    catch (const std::exception & e)
    {
        throw std::runtime_error(std::string("AI SQL generation failed: ") + e.what());
    }
#else
    (void)prompt;
    throw std::runtime_error("AI SQL generation is not available in this build. Rebuild with USE_CLIENT_AI enabled.");
#endif
}

streaming_query_result * connection_wrapper::send_query(const std::string & query_str, const std::string & format, const py::dict & params)
{
    const auto parsed_params = parseParametersDict(params);
    auto * client = getChdbClient(*conn);
    QueryParameterGuard guard(client, parsed_params);

    CHDB::cachePythonTablesFromQuery(reinterpret_cast<chdb_conn *>(*conn), query_str);
    py::gil_scoped_release release;
    auto * result = chdb_stream_query_n(*conn, query_str.data(), query_str.size(), format.data(), format.size());
    const auto & error_msg = CHDB::chdb_result_error_string(result);
    if (!error_msg.empty())
    {
        std::string msg_copy(error_msg);
        chdb_destroy_query_result(result);
        throw std::runtime_error(msg_copy);
    }

    return new streaming_query_result(result);
}

query_result * connection_wrapper::streaming_fetch_result(streaming_query_result * streaming_result)
{
    py::gil_scoped_release release;

    if (!streaming_result || !streaming_result->get_result())
        return nullptr;

    auto * result  = chdb_stream_fetch_result(*conn, streaming_result->get_result());

    const auto & error_msg = CHDB::chdb_result_error_string(result);
    if (!error_msg.empty())
    {
        std::string msg_copy(error_msg);
        chdb_destroy_query_result(result);
        throw std::runtime_error(msg_copy);
    }

    return new query_result(result, false);
}

py::object connection_wrapper::streaming_fetch_df(streaming_query_result * streaming_result)
{
    if (!streaming_result || !streaming_result->get_result())
        return py::none();

    chdb_result * result = nullptr;
    CHDB::DataFrameQueryResult * chunk_result = nullptr;

    {
        py::gil_scoped_release release;

        result = chdb_stream_fetch_result(*conn, streaming_result->get_result());

        const auto & error_msg = CHDB::chdb_result_error_string(result);
        if (!error_msg.empty())
        {
            std::string msg_copy(error_msg);
            chdb_destroy_query_result(result);
            throw std::runtime_error(msg_copy);
        }

        if (!(chunk_result = dynamic_cast<CHDB::DataFrameQueryResult *>(reinterpret_cast<CHDB::QueryResult*>(result))))
            throw std::runtime_error("Expected DataFrameQueryResult for dataframe format");
    }

    py::handle df_handle = chunk_result->dataframe;
    chdb_destroy_query_result(result);

    return py::reinterpret_steal<py::object>(df_handle);
}

void connection_wrapper::streaming_cancel_query(streaming_query_result * streaming_result)
{
    py::gil_scoped_release release;

    if (!streaming_result || !streaming_result->get_result())
        return;

    chdb_stream_cancel_query(*conn, streaming_result->get_result());
}

#if USE_CLIENT_AI
void connection_wrapper::applyAIParams(std::map<std::string, std::string> & params)
{
    DB::AIConfiguration config;

    auto consume_string = [&](const std::string & target, std::string DB::AIConfiguration::* field)
    {
        for (auto it = params.begin(); it != params.end();)
        {
            if (Poco::toLower(it->first) == target)
            {
                if (!it->second.empty())
                {
                    if ((config.*field).empty())
                        config.*field = it->second;
                }
                it = params.erase(it);
            }
            else
            {
                ++it;
            }
        }
    };

    auto consume_double = [&](const std::string & target, double DB::AIConfiguration::* field)
    {
        for (auto it = params.begin(); it != params.end();)
        {
            if (Poco::toLower(it->first) == target)
            {
                if (!it->second.empty())
                {
                    try
                    {
                        config.*field = std::stod(it->second);
                    }
                    catch (...)
                    {
                    }
                }
                it = params.erase(it);
            }
            else
            {
                ++it;
            }
        }
    };

    auto consume_size_t = [&](const std::string & target, size_t DB::AIConfiguration::* field)
    {
        for (auto it = params.begin(); it != params.end();)
        {
            if (Poco::toLower(it->first) == target)
            {
                if (!it->second.empty())
                {
                    try
                    {
                        config.*field = static_cast<size_t>(std::stoul(it->second));
                    }
                    catch (...)
                    {
                    }
                }
                it = params.erase(it);
            }
            else
            {
                ++it;
            }
        }
    };

    auto consume_bool = [&](const std::string & target, bool DB::AIConfiguration::* field)
    {
        for (auto it = params.begin(); it != params.end();)
        {
            if (Poco::toLower(it->first) == target)
            {
                if (!it->second.empty())
                {
                    std::string val = Poco::toLower(it->second);
                    if (val == "1" || val == "true" || val == "yes" || val == "on")
                        config.*field = true;
                    else if (val == "0" || val == "false" || val == "no" || val == "off")
                        config.*field = false;
                }
                it = params.erase(it);
            }
            else
            {
                ++it;
            }
        }
    };

    consume_string("ai_api_key", &DB::AIConfiguration::api_key);
    consume_string("ai_base_url", &DB::AIConfiguration::base_url);
    consume_string("ai_model", &DB::AIConfiguration::model);
    consume_string("ai_provider", &DB::AIConfiguration::provider);
    consume_double("ai_temperature", &DB::AIConfiguration::temperature);
    consume_size_t("ai_max_tokens", &DB::AIConfiguration::max_tokens);
    consume_size_t("ai_timeout_seconds", &DB::AIConfiguration::timeout_seconds);
    consume_string("ai_system_prompt", &DB::AIConfiguration::system_prompt);
    consume_size_t("ai_max_steps", &DB::AIConfiguration::max_steps);
    consume_bool("ai_enable_schema_access", &DB::AIConfiguration::enable_schema_access);

    ai_config = config;
}
#endif

void cursor_wrapper::execute(const std::string & query_str)
{
    release_result();
    CHDB::cachePythonTablesFromQuery(reinterpret_cast<chdb_conn *>(conn->get_conn()), query_str);
    // Use JSONCompactEachRowWithNamesAndTypes format for better type support
    py::gil_scoped_release release;
    current_result = chdb_query_n(conn->get_conn(), query_str.data(), query_str.size(), CURSOR_DEFAULT_FORMAT, CURSOR_DEFAULT_FORMAT_LEN);
}


/// Cleanup function to be called before Python interpreter exits.
/// This ensures JIT cache is cleared before static CHJIT instances are destroyed.
static void chdb_cleanup_at_exit()
{
#if USE_EMBEDDED_COMPILER
    try
    {
        if (auto * cache = DB::CompiledExpressionCacheFactory::instance().tryGetCache())
            cache->clear();
    }
    catch (...)
    {
        // Ignore errors during cleanup at exit
    }
#endif
}

PYBIND11_MODULE(_chdb, m)
{
    m.doc() = "chDB module for query function";

    /// Register atexit handler to clean up JIT cache before interpreter exits.
    /// This prevents use-after-free when CompiledFunctionHolder destructors
    /// try to access already-destroyed static CHJIT instances.
    py::module_::import("atexit").attr("register")(py::cpp_function(&chdb_cleanup_at_exit));

    py::class_<memoryview_wrapper>(m, "memoryview_wrapper")
        .def(py::init<std::shared_ptr<local_result_wrapper>>(), py::return_value_policy::take_ownership)
        .def("tobytes", &memoryview_wrapper::bytes)
        .def("__len__", &memoryview_wrapper::size)
        .def("size", &memoryview_wrapper::size)
        .def("release", &memoryview_wrapper::release)
        .def("view", &memoryview_wrapper::view);

    py::class_<query_result>(m, "query_result")
        .def(py::init<chdb_result *>(), py::return_value_policy::take_ownership)
        .def("data", &query_result::data)
        .def("bytes", &query_result::bytes)
        .def("__str__", &query_result::str)
        .def("__len__", &query_result::size)
        .def("__repr__", &query_result::str)
        .def("show", [](query_result & self) { py::print(self); })
        .def("size", &query_result::size)
        .def("rows_read", &query_result::rows_read)
        .def("bytes_read", &query_result::bytes_read)
        .def("storage_rows_read", &query_result::storage_rows_read)
        .def("storage_bytes_read", &query_result::storage_bytes_read)
        .def("elapsed", &query_result::elapsed)
        .def("get_memview", &query_result::get_memview)
        .def("has_error", &query_result::has_error)
        .def("error_message", &query_result::error_message);

    py::class_<streaming_query_result>(m, "streaming_query_result")
        .def(py::init<chdb_result *>(), py::return_value_policy::take_ownership)
        .def("has_error", &streaming_query_result::has_error)
        .def("error_message", &streaming_query_result::error_message);

    py::class_<DB::PyReader, std::shared_ptr<DB::PyReader>>(m, "PyReader")
        .def(
            py::init<const py::object &>(),
            "Initialize the reader with data. The exact type and structure of `data` can vary."
            "you must hold the data with `self.data` in your inherit class\n\n"
            "Args:\n"
            "    data (Any): The data with which to initialize the reader, format and type are not strictly defined.")
        .def(
            "read",
            [](DB::PyReader & self, const std::vector<std::string> & col_names, int count)
            {
                // GIL is held when called from Python code. Release it to avoid deadlock
                py::gil_scoped_release release;
                return std::move(self.read(col_names, count));
            },
            "Read a specified number of rows from the given columns and return a list of objects, "
            "where each object is a sequence of values for a column.\n\n"
            "Args:\n"
            "    col_names (List[str]): List of column names to read.\n"
            "    count (int): Maximum number of rows to read.\n\n"
            "Returns:\n"
            "    List[Any]: List of sequences, one for each column.")
        .def(
            "get_schema",
            &DB::PyReader::getSchema,
            "Return a list of column names and their types.\n\n"
            "Returns:\n"
            "    List[str, str]: List of column name and type pairs.");

    py::class_<cursor_wrapper>(m, "cursor")
        .def(py::init<connection_wrapper *>())
        .def("execute", &cursor_wrapper::execute)
        .def("commit", &cursor_wrapper::commit)
        .def("close", &cursor_wrapper::close)
        .def("get_memview", &cursor_wrapper::get_memview)
        .def("data_size", &cursor_wrapper::data_size)
        .def("rows_read", &cursor_wrapper::rows_read)
        .def("bytes_read", &cursor_wrapper::bytes_read)
        .def("storage_rows_read", &cursor_wrapper::storage_rows_read)
        .def("storage_bytes_read", &cursor_wrapper::storage_bytes_read)
        .def("elapsed", &cursor_wrapper::elapsed)
        .def("has_error", &cursor_wrapper::has_error)
        .def("error_message", &cursor_wrapper::error_message);

    py::class_<connection_wrapper>(m, "connect")
        .def(py::init([](const std::string & path) { return new connection_wrapper(path); }), py::arg("path") = ":memory:")
        .def("cursor", &connection_wrapper::cursor)
        .def("execute", &connection_wrapper::query)
        .def("commit", &connection_wrapper::commit)
        .def("close", &connection_wrapper::close)
        .def(
            "query",
            &connection_wrapper::query,
            py::arg("query_str"),
            py::arg("format") = "CSV",
            py::kw_only(),
            py::arg("params") = py::dict(),
            "Execute a query and return a query_result object")
        .def(
            "query_df",
            &connection_wrapper::query_df,
            py::arg("query_str"),
            py::kw_only(),
            py::arg("params") = py::dict(),
            "Execute a query and return a DataFrame")
#if USE_CLIENT_AI
        .def(
            "generate_sql",
            &connection_wrapper::generate_sql,
            py::arg("prompt"),
            "Generate SQL text from a natural language prompt using the configured AI provider")
#endif
        .def(
            "send_query",
            &connection_wrapper::send_query,
            py::arg("query_str"),
            py::arg("format") = "CSV",
            py::kw_only(),
            py::arg("params") = py::dict(),
            "Send a streaming query and return a streaming query result object")
        .def(
            "streaming_fetch_result",
            &connection_wrapper::streaming_fetch_result,
                py::arg("streaming_result"),
                "Fetches a data chunk from the streaming result. This function should be called repeatedly until the result is exhausted")
        .def(
            "streaming_fetch_df",
            &connection_wrapper::streaming_fetch_df,
                py::arg("streaming_result"),
                "Fetches a DataFrame from the streaming result. This function should be called repeatedly until the result is exhausted")
        .def(
            "streaming_cancel_query",
            &connection_wrapper::streaming_cancel_query,
            py::arg("streaming_result"),
            "Cancel a streaming query");

    m.def(
        "query",
        &query,
        py::arg("queryStr"),
        py::arg("output_format") = "CSV",
        py::kw_only(),
        py::arg("path") = "",
        py::arg("udf_path") = "",
        py::arg("params") = py::dict(),
        "Query chDB and return a query_result object or DataFrame");

    auto destroy_import_cache = []()
    {
        CHDB::PythonImporter::destroy();
    };
    m.add_object("_destroy_import_cache", py::capsule(destroy_import_cache));
}
