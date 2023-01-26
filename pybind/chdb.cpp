#include "chdb.h"

#include <iostream>
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

namespace py = pybind11;

bool inside_main = true;

class query_result
{
private:
    local_result * result;
    py::memoryview * memview;

public:
    query_result(local_result * result) : result(result)
    {
        if (result)
        {
            memview = new py::memoryview(py::memoryview::from_memory(result->buf, result->len, true));
        }
        else
        {
            memview = new py::memoryview(py::memoryview::from_memory(nullptr, 0, true));
        }
    }
    ~query_result()
    {
        free_result(result);
        delete memview;
    }

    char * data() { return result->buf; }
    size_t size() { return result->len; }
    py::memoryview get_memview() { return *memview; }
};

local_result * queryToBuffer(const std::string & queryStr, const std::string & format = "CSV")
{
    char * argv_[] = {(char *)"clickhouse", (char *)"--multiquery"};

    std::vector<char *> argv(argv_, argv_ + sizeof(argv_) / sizeof(argv_[0]));
    std::unique_ptr<char[]> formatStr(new char[format.size() + 17]);
    snprintf(formatStr.get(), format.size() + 17, "--output-format=%s", format.c_str());
    argv.push_back(formatStr.get());
    std::unique_ptr<char[]> qStr(new char[queryStr.size() + 11]);
    snprintf(qStr.get(), queryStr.size() + 11, "--query=%s", queryStr.c_str());
    argv.push_back(qStr.get());
    return query_stable(argv.size(), argv.data());
}

std::unique_ptr<query_result> query(const std::string & queryStr, const std::string & format = "CSV")
{
    return std::make_unique<query_result>(queryToBuffer(queryStr, format));
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
    m.doc() = "My module for query function";

    py::class_<query_result>(m, "query_result")
        .def(py::init<local_result *>())
        .def("data", &query_result::data)
        .def("size", &query_result::size)
        .def("get_memview", &query_result::get_memview);

    m.def("query", &query, "A function which queries Clickhouse and returns a query_result object");
}

#endif // PY_TEST_MAIN
