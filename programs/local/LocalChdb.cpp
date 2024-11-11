#include "LocalChdb.h"
#include "chdb.h"

#if USE_PYTHON

#    include <Storages/StoragePython.h>


namespace py = pybind11;

extern bool inside_main = true;


local_result_v2 * queryToBuffer(
    const std::string & queryStr,
    const std::string & output_format = "CSV",
    const std::string & path = {},
    const std::string & udfPath = {})
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

    // Convert std::string to char*
    std::vector<char *> argv_char;
    argv_char.reserve(argv.size());
    for (auto & arg : argv)
        argv_char.push_back(const_cast<char *>(arg.c_str()));

    py::gil_scoped_release release;
    return query_stable_v2(argv_char.size(), argv_char.data());
}

// Pybind11 will take over the ownership of the `query_result` object
// using smart ptr will cause early free of the object
query_result * query(
    const std::string & queryStr,
    const std::string & output_format = "CSV",
    const std::string & path = {},
    const std::string & udfPath = {})
{
    return new query_result(queryToBuffer(queryStr, output_format, path, udfPath));
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

    if (conn_str == ":memory:")
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
        std::string query = working_str.substr(query_pos + 1);

        // Parse parameters
        std::istringstream params_stream(query);
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
        }
    }
    else
    {
        path = working_str;
    }

    // Convert relative paths to absolute
    if (!path.empty() && path[0] != '/')
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
                argv.push_back("--readonly=1");
            }
        }
    }

    return argv;
}

void connection_wrapper::initialize_database()
{
    if (is_memory_db)
    {
        // Setup memory engine
        query_result * ret = query("CREATE DATABASE IF NOT EXISTS default ENGINE = Memory; USE default");
        if (ret->has_error())
        {
            auto err_msg = fmt::format("Failed to create memory database: {}", std::string(ret->error_message()));
            delete ret;
            throw std::runtime_error(err_msg);
        }
    }
    else
    {
        // Create directory if it doesn't exist
        std::filesystem::create_directories(db_path);
        // Setup Atomic database
        query_result * ret = query("CREATE DATABASE IF NOT EXISTS default ENGINE = Atomic; USE default");
        if (ret->has_error())
        {
            auto err_msg = fmt::format("Failed to create database: {}", std::string(ret->error_message()));
            delete ret;
            throw std::runtime_error(err_msg);
        }
    }
}

connection_wrapper::connection_wrapper(int argc, char ** argv)
{
    conn = connect_chdb(argc, argv);
    if (!conn)
    {
        throw std::runtime_error("Failed to connect to chdb");
    }
}

connection_wrapper::connection_wrapper(const std::string & conn_str)
{
    auto [path, params] = parse_connection_string(conn_str);

    db_path = path;
    is_memory_db = (path == ":memory:");

    auto argv = build_clickhouse_args(path, params);

    // Convert to char* array
    std::vector<char *> argv_char;
    argv_char.reserve(argv.size());
    for (auto & arg : argv)
    {
        argv_char.push_back(const_cast<char *>(arg.c_str()));
    }

    conn = connect_chdb(argv_char.size(), argv_char.data());
    if (!conn)
    {
        throw std::runtime_error("Failed to connect to chdb");
    }

    initialize_database();
}

connection_wrapper::~connection_wrapper()
{
    if (conn)
    {
        close_conn(conn);
        conn = nullptr;
    }
}

cursor_wrapper * connection_wrapper::cursor()
{
    return new cursor_wrapper(this);
}

void connection_wrapper::commit()
{
    // do nothing
}

void connection_wrapper::close()
{
    if (conn)
    {
        close_conn(conn);
        conn = nullptr;
    }
}

query_result * connection_wrapper::query(const std::string & query_str, const std::string & format)
{
    return new query_result(query_conn(conn, query_str.c_str(), format.c_str()));
}

void cursor_wrapper::execute(const std::string & query_str)
{
    if (current_result)
    {
        // The free_result_v2 vector is managed by the ClickHouse Engine
        // As we don't want to copy the data, so just release the memory here.
        // The memory will be released when the ClientBase.query_result_buf is reassigned.
        if (current_result->_vec)
        {
            current_result->_vec = nullptr;
        }
        free_result_v2(current_result);

        current_result = nullptr;
    }

    // Always use Arrow format internally
    current_result = query_conn(conn->get_conn(), query_str.c_str(), "ArrowStream");
}


#    ifdef PY_TEST_MAIN
#        include <string_view>
#        include <arrow/api.h>
#        include <arrow/buffer.h>
#        include <arrow/io/memory.h>
#        include <arrow/ipc/api.h>
#        include <arrow/python/pyarrow.h>


std::shared_ptr<arrow::Table> queryToArrow(const std::string & queryStr)
{
    auto result = queryToBuffer(queryStr, "Arrow");
    if (result)
    {
        // Create an Arrow input stream from the Arrow buffer
        auto input_stream = std::make_shared<arrow::io::BufferReader>(reinterpret_cast<uint8_t *>(result->buf), result->len);
        auto arrow_reader = arrow::ipc::RecordBatchFileReader::Open(input_stream, result->len).ValueOrDie();

        // Read all the record batches from the Arrow reader
        auto batch = arrow_reader->ReadRecordBatch(0).ValueOrDie();
        std::shared_ptr<arrow::Table> arrow_table = arrow::Table::FromRecordBatches({batch}).ValueOrDie();

        // Free the memory used by the result
        free_result(result);

        return arrow_table;
    }
    else
    {
        return nullptr;
    }
}

int main()
{
    // auto out = queryToVector("SELECT * FROM file('/home/Clickhouse/bench/result.parquet', Parquet) LIMIT 10");
    // out with string_view
    // std::cerr << std::string_view(out->data(), out->size()) << std::endl;
    // std::cerr << "out.size() = " << out->size() << std::endl;
    auto out = queryToArrow("SELECT * FROM file('/home/Clickhouse/bench/result.parquet', Parquet) LIMIT 10");
    std::cerr << "out->num_columns() = " << out->num_columns() << std::endl;
    std::cerr << "out->num_rows() = " << out->num_rows() << std::endl;
    std::cerr << "out.ToString() = " << out->ToString() << std::endl;
    std::cerr << "out->schema()->ToString() = " << out->schema()->ToString() << std::endl;

    return 0;
}
#    else
PYBIND11_MODULE(_chdb, m)
{
    m.doc() = "chDB module for query function";

    py::class_<memoryview_wrapper>(m, "memoryview_wrapper")
        .def(py::init<std::shared_ptr<local_result_wrapper>>(), py::return_value_policy::take_ownership)
        .def("tobytes", &memoryview_wrapper::bytes)
        .def("__len__", &memoryview_wrapper::size)
        .def("size", &memoryview_wrapper::size)
        .def("release", &memoryview_wrapper::release)
        .def("view", &memoryview_wrapper::view);

    py::class_<query_result>(m, "query_result")
        .def(py::init<local_result_v2 *>(), py::return_value_policy::take_ownership)
        .def("data", &query_result::data)
        .def("bytes", &query_result::bytes)
        .def("__str__", &query_result::str)
        .def("__len__", &query_result::size)
        .def("__repr__", &query_result::str)
        .def("show", [](query_result & self) { py::print(self); })
        .def("size", &query_result::size)
        .def("rows_read", &query_result::rows_read)
        .def("bytes_read", &query_result::bytes_read)
        .def("elapsed", &query_result::elapsed)
        .def("get_memview", &query_result::get_memview)
        .def("has_error", &query_result::has_error)
        .def("error_message", &query_result::error_message);

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
        .def("get_memview", &cursor_wrapper::get_memview)
        .def("data_size", &cursor_wrapper::data_size)
        .def("rows_read", &cursor_wrapper::rows_read)
        .def("bytes_read", &cursor_wrapper::bytes_read)
        .def("elapsed", &cursor_wrapper::elapsed)
        .def("has_error", &cursor_wrapper::has_error)
        .def("error_message", &cursor_wrapper::error_message);

    py::class_<connection_wrapper>(m, "connect")
        .def(
            py::init(
                [](const std::string & path)
                {
                    try
                    {
                        return new connection_wrapper(path);
                    }
                    catch (const std::exception & e)
                    {
                        throw py::error_already_set();
                    }
                }),
            py::arg("path") = ":memory:")
        .def("cursor", &connection_wrapper::cursor)
        .def("execute", &connection_wrapper::query)
        .def("commit", &connection_wrapper::commit)
        .def("close", &connection_wrapper::close)
        .def(
            "query",
            &connection_wrapper::query,
            py::arg("query_str"),
            py::arg("format") = "CSV",
            "Execute a query and return a query_result object");

    m.def(
        "query",
        &query,
        py::arg("queryStr"),
        py::arg("output_format") = "CSV",
        py::kw_only(),
        py::arg("path") = "",
        py::arg("udf_path") = "",
        "Query chDB and return a query_result object");
}

#    endif // PY_TEST_MAIN
#endif // USE_PYTHON
