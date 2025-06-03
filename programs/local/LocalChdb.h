#pragma once

#include <cstddef>
#include <string_view>
#include "config.h"

#    include <filesystem>
#    include <iostream>
#    include <sstream>
#    include <arrow/api.h>
#    include <arrow/io/api.h>
#    include <arrow/ipc/api.h>
#    include <fmt/core.h>
#    include <pybind11/gil.h>
#    include <pybind11/pybind11.h>
#    include <pybind11/pytypes.h>
#    include <pybind11/stl.h>
#    include "chdb.h"

namespace py = pybind11;

class __attribute__((visibility("default"))) local_result_wrapper;
class __attribute__((visibility("default"))) connection_wrapper;
class __attribute__((visibility("default"))) cursor_wrapper;
class __attribute__((visibility("default"))) memoryview_wrapper;
class __attribute__((visibility("default"))) query_result;
class __attribute__((visibility("default"))) streaming_query_result;

class connection_wrapper
{
private:
    chdb_conn ** conn;
    std::string db_path;
    bool is_memory_db;
    bool is_readonly;

public:
    explicit connection_wrapper(const std::string & conn_str);
    chdb_conn * get_conn() { return *conn; }
    ~connection_wrapper();
    cursor_wrapper * cursor();
    void commit();
    void close();
    query_result * query(const std::string & query_str, const std::string & format = "CSV");
    streaming_query_result * send_query(const std::string & query_str, const std::string & format = "CSV");
    query_result * streaming_fetch_result(streaming_query_result * streaming_result);
    void streaming_cancel_query(streaming_query_result * streaming_result);

    // Move the private methods declarations here
    std::pair<std::string, std::map<std::string, std::string>> parse_connection_string(const std::string & conn_str);
    std::vector<std::string> build_clickhouse_args(const std::string & path, const std::map<std::string, std::string> & params);
};

class local_result_wrapper
{
private:
    local_result_v2 * result;
    bool keep_buf; // background server mode will handle buf in ClickHouse engine

public:
    local_result_wrapper(local_result_v2 * result) : result(result), keep_buf(false) { }
    local_result_wrapper(local_result_v2 * result, bool keep_buf) : result(result), keep_buf(keep_buf) { }
    ~local_result_wrapper()
    {
        if (keep_buf)
        {
            if (!result)
                return;

            result->_vec = nullptr;
            delete[] result->error_message;
            delete result;
        }
        else
        {
            free_result_v2(result);
        }
    }
    char * data()
    {
        if (result == nullptr)
        {
            return nullptr;
        }
        return result->buf;
    }
    size_t size()
    {
        if (result == nullptr)
        {
            return 0;
        }
        return result->len;
    }
    py::bytes bytes()
    {
        if (result == nullptr)
        {
            return py::bytes();
        }
        return py::bytes(result->buf, result->len);
    }
    py::str str()
    {
        if (result == nullptr)
        {
            return py::str();
        }
        return py::str(result->buf, result->len);
    }
    // Query statistics
    size_t rows_read()
    {
        if (result == nullptr)
        {
            return 0;
        }
        return result->rows_read;
    }
    size_t bytes_read()
    {
        if (result == nullptr)
        {
            return 0;
        }
        return result->bytes_read;
    }
    size_t storage_rows_read()
    {
        if (result == nullptr)
        {
            return 0;
        }
        return 0;
    }
    size_t storage_bytes_read()
    {
        if (result == nullptr)
        {
            return 0;
        }
        return 0;
    }
    double elapsed()
    {
        if (result == nullptr)
        {
            return 0;
        }
        return result->elapsed;
    }
    bool has_error()
    {
        if (result == nullptr)
        {
            return false;
        }
        return result->error_message != nullptr;
    }
    py::str error_message()
    {
        if (has_error())
        {
            return py::str(result->error_message);
        }
        return py::str();
    }
};

class query_result
{
private:
    std::shared_ptr<local_result_wrapper> result_wrapper;

public:
    query_result(local_result_v2 * result) : result_wrapper(std::make_shared<local_result_wrapper>(result)) { }
    query_result(local_result_v2 * result, bool keep_buf) : result_wrapper(std::make_shared<local_result_wrapper>(result, keep_buf)) { }
    ~query_result() = default;
    char * data() { return result_wrapper->data(); }
    py::bytes bytes() { return result_wrapper->bytes(); }
    py::str str() { return result_wrapper->str(); }
    size_t size() { return result_wrapper->size(); }
    size_t rows_read() { return result_wrapper->rows_read(); }
    size_t bytes_read() { return result_wrapper->bytes_read(); }
    size_t storgae_rows_read() { return result_wrapper->storage_rows_read(); }
    size_t storage_bytes_read() { return result_wrapper->storage_bytes_read(); }
    double elapsed() { return result_wrapper->elapsed(); }
    bool has_error() { return result_wrapper->has_error(); }
    py::str error_message() { return result_wrapper->error_message(); }
    memoryview_wrapper * get_memview();
};

class streaming_query_result
{
private:
    chdb_streaming_result * result;

public:
    streaming_query_result(chdb_streaming_result * result_) : result(result_) {}
    ~streaming_query_result()
    {
        chdb_destroy_result(result);
    }
    bool has_error() { return chdb_streaming_result_error(result) != nullptr; }
    py::str error_message() { return chdb_streaming_result_error(result); }
    chdb_streaming_result * get_result() { return result; }
};

class memoryview_wrapper
{
private:
    std::shared_ptr<local_result_wrapper> result_wrapper;

public:
    explicit memoryview_wrapper(std::shared_ptr<local_result_wrapper> result) : result_wrapper(result)
    {
        // std::cerr << "memoryview_wrapper::memoryview_wrapper" << this->result->bytes() << std::endl;
    }
    ~memoryview_wrapper() = default;

    size_t size()
    {
        if (result_wrapper == nullptr)
        {
            return 0;
        }
        return result_wrapper->size();
    }

    py::bytes bytes() { return result_wrapper->bytes(); }

    void release() { }

    py::memoryview view()
    {
        if (result_wrapper != nullptr)
        {
            return py::memoryview(py::memoryview::from_memory(result_wrapper->data(), result_wrapper->size(), true));
        }
        else
        {
            return py::memoryview(py::memoryview::from_memory(nullptr, 0, true));
        }
    }
};

class cursor_wrapper
{
private:
    connection_wrapper * conn;
    local_result_v2 * current_result;

    void release_result()
    {
        if (current_result)
        {
            if (current_result->_vec)
            {
                auto * vec = reinterpret_cast<std::vector<char> *>(current_result->_vec);
                delete vec;
                current_result->_vec = nullptr;
            }
            free_result_v2(current_result);

            current_result = nullptr;
        }
    }

public:
    explicit cursor_wrapper(connection_wrapper * connection) : conn(connection), current_result(nullptr) { }

    ~cursor_wrapper() { release_result(); }

    void execute(const std::string & query_str);

    void commit()
    {
        // do nothing
    }

    void close() { release_result(); }

    py::memoryview get_memview()
    {
        if (current_result == nullptr)
        {
            return py::memoryview(py::memoryview::from_memory(nullptr, 0, true));
        }
        return py::memoryview(py::memoryview::from_memory(current_result->buf, current_result->len, true));
    }

    size_t data_size()
    {
        if (current_result == nullptr)
        {
            return 0;
        }
        return current_result->len;
    }

    size_t rows_read()
    {
        if (current_result == nullptr)
        {
            return 0;
        }
        return current_result->rows_read;
    }

    size_t bytes_read()
    {
        if (current_result == nullptr)
        {
            return 0;
        }
        return current_result->bytes_read;
    }

    double elapsed()
    {
        if (current_result == nullptr)
        {
            return 0;
        }
        return current_result->elapsed;
    }

    bool has_error()
    {
        if (current_result == nullptr)
        {
            return false;
        }
        return current_result->error_message != nullptr;
    }

    py::str error_message()
    {
        if (has_error())
        {
            return py::str(current_result->error_message);
        }
        return py::str();
    }
};
