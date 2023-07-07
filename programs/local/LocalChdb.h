#pragma once

#include "chdb.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

namespace py = pybind11;

class local_result_wrapper;
class __attribute__((visibility("default"))) memoryview_wrapper;
class __attribute__((visibility("default"))) query_result;


class local_result_wrapper
{
private:
    local_result * result;

public:
    local_result_wrapper(local_result * result) : result(result) { }
    ~local_result_wrapper()
    {
        free_result(result);
        delete result;
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
};

class query_result
{
private:
    std::shared_ptr<local_result_wrapper> result_wrapper;

public:
    query_result(local_result * result) : result_wrapper(std::make_shared<local_result_wrapper>(result)) { }
    ~query_result() { }
    char * data() { return result_wrapper->data(); }
    py::bytes bytes() { return result_wrapper->bytes(); }
    py::str str() { return result_wrapper->str(); }
    size_t size() { return result_wrapper->size(); }
    memoryview_wrapper * get_memview();
};

class memoryview_wrapper
{
private:
    std::shared_ptr<local_result_wrapper> result_wrapper;

public:
    memoryview_wrapper(std::shared_ptr<local_result_wrapper> result) : result_wrapper(result)
    {
        // std::cerr << "memoryview_wrapper::memoryview_wrapper" << this->result->bytes() << std::endl;
    }
    ~memoryview_wrapper() { }

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
