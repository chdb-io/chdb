#include "PythonUtils.h"
#include "config.h"

#include <cstddef>
#include <pybind11/gil.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>
#include <pybind11/detail/non_limited_api.h>
#include <utf8proc.h>
#include <Columns/ColumnString.h>
#include <Common/logger_useful.h>

namespace DB
{

const char * ConvertPyUnicodeToUtf8(const void * input, int kind, size_t codepoint_cnt, size_t & output_size)
{
    if (input == nullptr)
    {
        return nullptr;
    }

    char * output_buffer = new char[codepoint_cnt * 4 + 1]; // Allocate buffer based on calculated size
    char * target = output_buffer;
    size_t total_size = 0;

    // Encode each Unicode codepoint to UTF-8 using utf8proc
    switch (kind)
    {
        case 1: {
            const auto * start = static_cast<const uint8_t *>(input);
            for (size_t i = 0; i < codepoint_cnt; ++i)
            {
                int sz = utf8proc_encode_char(start[i], reinterpret_cast<utf8proc_uint8_t *>(target));
                target += sz;
                total_size += sz;
            }
            break;
        }
        case 2: {
            const auto * start = static_cast<const uint16_t *>(input);
            for (size_t i = 0; i < codepoint_cnt; ++i)
            {
                int sz = utf8proc_encode_char(start[i], reinterpret_cast<utf8proc_uint8_t *>(target));
                target += sz;
                total_size += sz;
            }
            break;
        }
        case 4: {
            const auto * start = static_cast<const uint32_t *>(input);
            for (size_t i = 0; i < codepoint_cnt; ++i)
            {
                int sz = utf8proc_encode_char(start[i], reinterpret_cast<utf8proc_uint8_t *>(target));
                target += sz;
                total_size += sz;
            }
            break;
        }
    }

    output_buffer[total_size] = '\0'; // Null-terminate the output string
    output_size = total_size;
    return output_buffer;
}

size_t
ConvertPyUnicodeToUtf8(const void * input, int kind, size_t codepoint_cnt, ColumnString::Offsets & offsets, ColumnString::Chars & chars)
{
    if (input == nullptr)
    {
        return 0;
    }

    // Estimate the maximum buffer size required for the UTF-8 output
    // Buffers is reserved from the caller, so we can safely resize it and memory will not be wasted
    size_t estimated_size = codepoint_cnt * 4 + 1; // Allocate buffer for UTF-8 output
    size_t chars_cursor = chars.size();
    size_t target_size = chars_cursor + estimated_size;
    chars.resize(target_size);

    // Resize the character buffer to accommodate the UTF-8 string
    chars.resize(chars_cursor + estimated_size + 1); // +1 for null terminator

    size_t offset = chars_cursor;
    switch (kind)
    {
        case 1: { // Latin1/ASCII
            const auto * start = static_cast<const uint8_t *>(input);
            for (size_t i = 0; i < codepoint_cnt; ++i)
            {
                auto sz = utf8proc_encode_char(start[i], reinterpret_cast<utf8proc_uint8_t *>(&chars[offset]));
                offset += sz;
            }
            break;
        }
        case 2: { // UTF-16
            const auto * start = static_cast<const uint16_t *>(input);
            for (size_t i = 0; i < codepoint_cnt; ++i)
            {
                auto sz = utf8proc_encode_char(start[i], reinterpret_cast<utf8proc_uint8_t *>(&chars[offset]));
                offset += sz;
            }
            break;
        }
        case 4: { // UTF-32
            const auto * start = static_cast<const uint32_t *>(input);
            for (size_t i = 0; i < codepoint_cnt; ++i)
            {
                auto sz = utf8proc_encode_char(start[i], reinterpret_cast<utf8proc_uint8_t *>(&chars[offset]));
                offset += sz;
            }
            break;
        }
    }

    chars[offset++] = '\0'; // Null terminate the output string
    offsets.push_back(offset); // Include the null terminator in the offset
    chars.resize(offset); // Resize to the actual used size, including null terminator

    return offset; // Return the number of bytes written, not including the null terminator
}

int PyString_AsStringAndSize(PyObject * ob, char ** charpp, Py_ssize_t * sizep)
{
    // always convert it to utf8, but this case is rare, here goes the slow path
    py::gil_scoped_acquire acquire;
    if (PyUnicode_Check(ob))
    {
        *charpp = const_cast<char *>(pybind11::non_limited_api::PyUnicode_AsUTF8AndSize(ob, sizep));
        if (*charpp == nullptr)
        {
            return -1;
        }
        return 0;
    }
    else
    {
        return PyBytes_AsStringAndSize(ob, charpp, sizep);
    }
}

void FillColumnString(PyObject * obj, ColumnString * column)
{
    // Simplified implementation using stable API only
    if (!PyUnicode_Check(obj))
    {
        return;
    }

    // Use stable API to get UTF-8 representation
    Py_ssize_t size;
    const char * data = pybind11::non_limited_api::PyUnicode_AsUTF8AndSize(obj, &size);
    if (data != nullptr && size >= 0)
    {
        column->insertData(data, static_cast<size_t>(size));
    }
    else
    {
        // Fallback for error cases
        py::gil_scoped_acquire acquire;
        char * fallback_data = nullptr;
        Py_ssize_t fallback_size = 0;
        if (PyString_AsStringAndSize(obj, &fallback_data, &fallback_size) == 0 && fallback_data != nullptr)
        {
            column->insertData(fallback_data, static_cast<size_t>(fallback_size));
        }
        else
        {
            throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED, "Failed to convert Python unicode object to UTF-8");
        }
    }
}

const char * GetPyUtf8StrData(PyObject * obj, size_t & buf_len)
{
    if (!PyUnicode_Check(obj))
    {
        buf_len = 0;
        return nullptr;
    }

    // Use stable API approach - always convert to UTF-8
    Py_ssize_t size;
    const char * data = pybind11::non_limited_api::PyUnicode_AsUTF8AndSize(obj, &size);
    if (data == nullptr || size < 0)
    {
        // Fallback using PyString_AsStringAndSize
        py::gil_scoped_acquire acquire;
        char * fallback_data = nullptr;
        Py_ssize_t fallback_size = 0;
        if (PyString_AsStringAndSize(obj, &fallback_data, &fallback_size) != 0 || fallback_data == nullptr)
        {
            throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED, "Failed to convert Python unicode object to UTF-8");
        }
        buf_len = static_cast<size_t>(fallback_size);
        return fallback_data;
    }

    buf_len = static_cast<size_t>(size);
    return data;
}

bool _isInheritsFromPyReader(const py::handle & obj)
{
    try
    {
        // Check directly if obj is an instance of a class named "PyReader"
        if (py::str(obj.attr("__class__").attr("__name__")).cast<std::string>() == "PyReader")
            return true;

        // Check the direct base classes of obj's class for "PyReader"
        py::tuple bases = obj.attr("__class__").attr("__bases__");
        for (auto base : bases)
            if (py::str(base.attr("__name__")).cast<std::string>() == "PyReader")
                return true;
    }
    catch (const py::error_already_set &)
    {
        // Ignore the exception, and return false
    }

    return false;
}

// Will try to get the ref of py::array from pandas Series, or PyArrow Table
// without import numpy or pyarrow. Just from class name for now.
const void * tryGetPyArray(const py::object & obj, py::handle & result, py::handle & tmp, std::string & type_name, size_t & row_count)
{
    py::gil_scoped_acquire acquire;
    type_name = py::str(obj.attr("__class__").attr("__name__")).cast<std::string>();
    if (type_name == "ndarray")
    {
        // Return the handle of py::array directly
        row_count = py::len(obj);
        py::array array = obj.cast<py::array>();
        result = array;
        return array.data();
    }
    else if (type_name == "Series")
    {
        // Try to get the handle of py::array from pandas Series
        py::array array = obj.attr("values");
        row_count = py::len(obj);
        // if element type is bytes or object, we need to convert it to string
        // chdb todo: need more type check
        if (row_count > 0)
        {
            auto elem_type = obj.attr("__getitem__")(0).attr("__class__").attr("__name__").cast<std::string>();
            if (elem_type == "str" || elem_type == "unicode")
            {
                result = array;
                return array.data();
            }
            if (elem_type == "bytes" || elem_type == "object")
            {
                // chdb todo: better handle for bytes and object type
                auto str_obj = obj.attr("astype")(py::dtype("str"));
                array = str_obj.attr("values");
                result = array;
                tmp = array;
                tmp.inc_ref();
                return array.data();
            }
        }

        result = array;
        return array.data();
    }
    else if (type_name == "Table")
    {
        // Try to get the handle of py::array from PyArrow Table
        py::array array = obj.attr("to_pandas")();
        row_count = py::len(obj);
        result = array;
        tmp = array;
        tmp.inc_ref();
        return array.data();
    }
    else if (type_name == "ChunkedArray")
    {
        // Try to get the handle of py::array from PyArrow ChunkedArray
        py::array array = obj.attr("to_numpy")();
        row_count = py::len(obj);
        result = array;
        tmp = array;
        tmp.inc_ref();
        return array.data();
    }
    else if (type_name == "list")
    {
        // Just set the row count for list
        row_count = py::len(obj);
        result = obj;
        return obj.ptr();
    }

    // chdb todo: maybe convert list to py::array?

    return nullptr;
}
}
