#include <iostream>
// #include <string_view>
#include <arrow/api.h>
#include <arrow/buffer.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/api.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

bool inside_main = true;

extern "C" {
struct local_result
{
    char * buf;
    size_t len;
};

local_result * query_stable(int argc, char ** argv);
void free_result(local_result * result);
}

local_result * queryToBuffer(const std::string & queryStr, const std::string & format = "CSV")
{
    char * argv_[] = {"clickhouse", "--multiquery"};

    std::vector<char *> argv(argv_, argv_ + sizeof(argv_) / sizeof(argv_[0]));
    std::unique_ptr<char[]> formatStr(new char[format.size() + 17]);
    snprintf(formatStr.get(), format.size() + 17, "--output-format=%s", format.c_str());
    argv.push_back(formatStr.get());
    std::unique_ptr<char[]> qStr(new char[queryStr.size() + 11]);
    snprintf(qStr.get(), queryStr.size() + 11, "--query=%s", queryStr.c_str());
    argv.push_back(qStr.get());
    return query_stable(argv.size(), argv.data());
}

std::shared_ptr<std::vector<char>> queryToVector(const std::string & queryStr, const std::string & format = "CSV")
{
    auto result = queryToBuffer(queryStr, format);
    if (result)
    {
        auto vec = std::make_shared<std::vector<char>>(result->buf, result->buf + result->len);
        free_result(result);
        return vec;
    }
    else
    {
        return nullptr;
    }
}

//same as queryToVector but return a CSV string
std::string queryToCSV(const std::string & queryStr)
{
    auto vec = queryToVector(queryStr);
    if (vec)
    {
        // std::cerr << "vec->size() = " << vec->size() << std::endl;
        return std::string(vec->begin(), vec->end());
    }
    else
    {
        return {};
    }
}

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

#ifdef PY_TEST_MAIN
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

    return 0;
}
#else
PYBIND11_MODULE(example, m)
{
    m.doc() = "My module for query function";

    // Bind the query function to the module
    // m.def(
    //     "query",
    //     &queryToVector,
    //     pybind11::arg("queryStr"),
    //     pybind11::return_value_policy::reference_internal,
    //     "Execute a query and return a vector of chars");

    // m.def("queryToVector", &queryToVector, "Execute SQL query");
    m.def("queryToCSV", &queryToCSV, "Execute SQL query and return CSV");
    m.def(
        "queryToArrowObject",
        [](const char * q_str) -> py::object
        {
            std::shared_ptr<arrow::Table> table = queryToArrow(q_str);
            // 创建一个 Python Capsule 对象
            py::capsule arrow_table_capsule(
                table.get(),
                [](void * ptr)
                {
                    // 释放指向 arrow::Table 的 shared_ptr
                    arrow::Table * table = reinterpret_cast<arrow::Table *>(ptr);
                    delete table;
                });
            // 使用 arrow::Table Capsule 创建一个 pyarrow.lib.Table 对象
            py::module pyarrow = py::module::import("pyarrow");
            return pyarrow.attr("lib").attr("Table").attr("_import_from_c")("example", "arrow_table", arrow_table_capsule);
        },
        py::return_value_policy::take_ownership);
}

#endif // PY_TEST_MAIN
