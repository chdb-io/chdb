#include "LocalChdb.h"

#include <iostream>


extern bool inside_main = true;


local_result * queryToBuffer(
    const std::string & queryStr,
    const std::string & output_format = "CSV",
    const std::string & path = {},
    const std::string & udfPath = {})
{
    std::vector<std::string> argv = {"clickhouse", "--", "--multiquery"};

    // If format is "Debug" or "debug", then we will add `--verbose` and `--log-level=trace` to argv
    if (output_format == "Debug" || output_format == "debug")
    {
        argv.push_back("--verbose");
        argv.push_back("--log-level=trace");
        // Add format string
        argv.push_back("--output-format=CSV");
    }
    else
    {
        // Add format string
        argv.push_back("--output-format=" + output_format);
    }

    // If udfPath is not empty, then we will add `--user_scripts_path` and `--user_defined_executable_functions_config` to argv
    // the path should be a one time thing, so the caller should take care of the temporary files deletion
    if (!udfPath.empty())
    {
        argv.push_back("--user_scripts_path=" + udfPath);
        argv.push_back("--user_defined_executable_functions_config=" + udfPath + "/*.xml");
    }

    // If path is not empty, then we will add `--path` to argv. This is used for chdb.Session to support stateful query
    if (!path.empty())
    {
        // Add path string
        argv.push_back("--path=" + path);
    }
    // Add query string
    argv.push_back("--query=" + queryStr);

    // Convert std::string to char*
    std::vector<char *> argv_char;
    for (auto & arg : argv)
        argv_char.push_back(const_cast<char *>(arg.c_str()));

    return query_stable(argv_char.size(), argv_char.data());
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

#ifdef PY_TEST_MAIN
#    include <string_view>
#    include <arrow/api.h>
#    include <arrow/buffer.h>
#    include <arrow/io/memory.h>
#    include <arrow/ipc/api.h>
#    include <arrow/python/pyarrow.h>


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
#else
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
        .def(py::init<local_result *>(), py::return_value_policy::take_ownership)
        .def("data", &query_result::data)
        .def("bytes", &query_result::bytes)
        .def("__str__", &query_result::str)
        .def("__len__", &query_result::size)
        .def("__repr__", &query_result::str)
        .def("size", &query_result::size)
        .def("rows_read", &query_result::rows_read)
        .def("bytes_read", &query_result::bytes_read)
        .def("elapsed", &query_result::elapsed)
        .def("get_memview", &query_result::get_memview);


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

#endif // PY_TEST_MAIN
