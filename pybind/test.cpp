#include <iostream>
// #include <string_view>
#include <arrow/api.h>
#include <arrow/buffer.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/api.h>
#include <arrow/python/pyarrow.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
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

// std::shared_ptr<arrow::Table> queryToArrow(const std::string & queryStr)
// {
//     auto result = queryToBuffer(queryStr, "Arrow");
//     if (result)
//     {
//         // Create an Arrow input stream from the Arrow buffer
//         auto input_stream = std::make_shared<arrow::io::BufferReader>(reinterpret_cast<uint8_t *>(result->buf), result->len);
//         auto arrow_reader = arrow::ipc::RecordBatchFileReader::Open(input_stream, result->len).ValueOrDie();

//         // Read all the record batches from the Arrow reader
//         auto batch = arrow_reader->ReadRecordBatch(0).ValueOrDie();
//         std::shared_ptr<arrow::Table> arrow_table = arrow::Table::FromRecordBatches({batch}).ValueOrDie();

//         // Free the memory used by the result
//         free_result(result);

//         return arrow_table;
//     }
//     else
//     {
//         return nullptr;
//     }
// }

// py::object queryToArrowObject(const std::string & queryStr)
// {
//     std::shared_ptr<arrow::Table> table = queryToArrow(queryStr);
//     // Wrap the table in a pyarrow.Table object
//     auto py_table = arrow::py::wrap_table(table);

//     // Use py::reinterpret_borrow to convert PyObject* to py::object
//     return py::reinterpret_borrow<py::object>(py_table);
// }

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
    std::cerr << "out->schema()->ToString() = " << out->schema()->ToString() << std::endl;

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
        "queryToBytes",
        [](const std::string & queryStr, const std::string & format) -> py::memoryview
        {
            auto result = queryToBuffer(queryStr, format);
            py::memoryview memview = py::memoryview::from_memory(result->buf, result->len);
            // free_result(result);
            return memview;
        },
        "Execute SQL query and return bytes");
    // m.def("queryToArrow", [](const std::string & queryStr) -> std::string
    // {
    //     auto table = queryToArrow(queryStr);
    //     if (table)
    //     {
    //         // // Create a Python Capsule object
    //         // py::capsule arrow_table_capsule(
    //         //     table.get(),
    //         //     [](void * ptr)
    //         //     {
    //         //         // Free the shared_ptr to the arrow::Table
    //         //         arrow::Table * table = reinterpret_cast<arrow::Table *>(ptr);
    //         //         delete table;
    //         //     });
    //         // Create a pyarrow.lib.Table object from the arrow::Table Capsule
    //         py::module pyarrow = py::module::import("pyarrow");
    //         auto arrow_table = pyarrow.attr("lib").attr("Table").attr("_import_from_c")("example", "arrow_table");
    //         // Convert the pyarrow.lib.Table object to a pyarrow.Buffer object
    //         auto arrow_buffer = arrow_table.attr("to_batches")().attr("serialize")();
    //         // Convert the pyarrow.Buffer object to a Python bytes object
    //         auto py_bytes = arrow_buffer.attr("to_pybytes")();
    //         // Convert the Python bytes object to a std::string
    //         return py_bytes.cast<std::string>();
    //     }
    //     else
    //     {
    //         return {};
    //     }
    // }
    // , "Execute SQL query and return Arrow Table");
    // m.def(
    //     "queryToArrowObject",
    //     &queryToArrowObject,
    //     py::return_value_policy::take_ownership);
}

#endif // PY_TEST_MAIN
