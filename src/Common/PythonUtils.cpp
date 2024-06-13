#include <cstddef>

#include <pybind11/gil.h>
#include <pybind11/pytypes.h>
#include <unicode/bytestream.h>
#include <unicode/unistr.h>
#include <Common/PythonUtils.h>
#include <Common/logger_useful.h>
#include "Columns/ColumnString.h"

namespace DB
{

const char * ConvertPyUnicodeToUtf8(const void * input, int kind, size_t codepoint_cnt, size_t & output_size)
{
    if (input == nullptr)
        return nullptr;

    char * output_buffer = new char[4 * codepoint_cnt]; // Allocate buffer for UTF-8 output

    size_t real_size = 0;

    switch (kind)
    {
        case 1: { // Handle 1-byte characters (Latin1/ASCII equivalent in ICU)
            const char * start = (const char *)input;
            const char * end = start + codepoint_cnt;
            char code_unit;
            char * target = output_buffer;
            int32_t append_size = 0;

            while (start < end)
            {
                code_unit = *start++;
                U8_APPEND_UNSAFE(target, append_size, code_unit);
            }
            real_size += append_size;
            output_buffer[real_size] = '\0'; // Null terminate the output string
            // LOG_DEBUG(&Poco::Logger::get("PythonUtils"), "Coverted 1byte String: {}", output_buffer);
            break;
        }
        case 2: { // Handle 2-byte characters (UTF-16 equivalent)
            const UChar * start = (const UChar *)input;
            const UChar * end = start + codepoint_cnt;
            UChar code_unit;
            char * target = output_buffer;
            int32_t append_size = 0;

            while (start < end)
            {
                code_unit = *start++;
                U8_APPEND_UNSAFE(target, append_size, code_unit);
            }
            real_size += append_size;
            output_buffer[real_size] = '\0'; // Null terminate the output string
            // LOG_DEBUG(&Poco::Logger::get("PythonUtils"), "Coverted 2byte String: {}", output_buffer);
            break;
        }
        case 4: { // Handle 4-byte characters (Assume UCS-4/UTF-32)
            const UInt32 * start = (const UInt32 *)input;
            const UInt32 * end = start + codepoint_cnt;
            UInt32 code_unit;
            char * target = output_buffer;
            int32_t append_size = 0;

            while (start < end)
            {
                code_unit = *start++;
                U8_APPEND_UNSAFE(target, append_size, code_unit);
            }
            real_size += append_size;
            output_buffer[real_size] = '\0'; // Null terminate the output string
            // LOG_DEBUG(&Poco::Logger::get("PythonUtils"), "Coverted 4byte String: {}", output_buffer);
            break;
        }
        default:
            delete[] output_buffer; // Clean up memory allocation if kind is unsupported
            throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Unsupported unicode kind {}", kind);
    }

    output_size = real_size;
    return output_buffer;
}

size_t
ConvertPyUnicodeToUtf8(const void * input, int kind, size_t codepoint_cnt, ColumnString::Offsets & offsets, ColumnString::Chars & chars)
{
    if (input == nullptr)
        return 0;

    size_t estimated_size = codepoint_cnt * 4 + 1; // Allocate buffer for UTF-8 output
    size_t chars_cursor = chars.size();
    size_t target_size = chars_cursor + estimated_size;
    chars.resize(target_size);

    switch (kind)
    {
        case 1: { // Handle 1-byte characters (Latin1/ASCII equivalent in ICU)
            const char * start = (const char *)input;
            const char * end = start + codepoint_cnt;
            char code_unit;
            int32_t append_size = 0;

            while (start < end)
            {
                code_unit = *start++;
                U8_APPEND_UNSAFE(chars.data(), chars_cursor, code_unit);
            }
            break;
        }
        case 2: { // Handle 2-byte characters (UTF-16 equivalent)
            const UChar * start = (const UChar *)input;
            const UChar * end = start + codepoint_cnt;
            UChar code_unit;
            int32_t append_size = 0;

            while (start < end)
            {
                code_unit = *start++;
                U8_APPEND_UNSAFE(chars.data(), chars_cursor, code_unit);
            }
            break;
        }
        case 4: { // Handle 4-byte characters (Assume UCS-4/UTF-32)
            const UInt32 * start = (const UInt32 *)input;
            const UInt32 * end = start + codepoint_cnt;
            UInt32 code_unit;
            int32_t append_size = 0;

            while (start < end)
            {
                code_unit = *start++;
                U8_APPEND_UNSAFE(chars.data(), chars_cursor, code_unit);
            }
            break;
        }
        default:
            throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Unsupported unicode kind {}", kind);
    }

    chars[chars_cursor++] = '\0'; // Null terminate the output string and increase the cursor
    offsets.push_back(chars_cursor);
    chars.resize_assume_reserved(chars_cursor);

    return chars_cursor;
}

void FillColumnString(PyObject * obj, ColumnString * column)
{
    ColumnString::Offsets & offsets = column->getOffsets();
    ColumnString::Chars & chars = column->getChars();
    if (PyUnicode_IS_COMPACT_ASCII(obj))
    {
        const char * data = reinterpret_cast<const char *>(PyUnicode_1BYTE_DATA(obj));
        size_t unicode_len = PyUnicode_GET_LENGTH(obj);
        column->insertData(data, unicode_len);
    }
    else
    {
        PyCompactUnicodeObject * unicode = reinterpret_cast<PyCompactUnicodeObject *>(obj);
        if (unicode->utf8 != nullptr)
        {
            // It's utf8 string, treat it like ASCII
            const char * data = reinterpret_cast<const char *>(unicode->utf8);
            column->insertData(data, unicode->utf8_length);
        }
        else if (PyUnicode_IS_COMPACT(obj))
        {
            auto kind = PyUnicode_KIND(obj);
            const char * data;
            size_t codepoint_cnt;

            if (kind == PyUnicode_1BYTE_KIND)
                data = reinterpret_cast<const char *>(PyUnicode_1BYTE_DATA(obj));
            else if (kind == PyUnicode_2BYTE_KIND)
                data = reinterpret_cast<const char *>(PyUnicode_2BYTE_DATA(obj));
            else if (kind == PyUnicode_4BYTE_KIND)
                data = reinterpret_cast<const char *>(PyUnicode_4BYTE_DATA(obj));
            else
                throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Unsupported unicode kind {}", kind);
            codepoint_cnt = PyUnicode_GET_LENGTH(obj);
            ConvertPyUnicodeToUtf8(data, kind, codepoint_cnt, offsets, chars);
        }
        else
        {
            // always convert it to utf8, but this case is rare, here goes the slow path
            py::gil_scoped_acquire acquire;
            Py_ssize_t bytes_size = -1;
            const char * data = PyUnicode_AsUTF8AndSize(obj, &bytes_size);
            if (bytes_size < 0)
                throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED, "Failed to convert Python unicode object to UTF-8");
            column->insertData(data, bytes_size);
        }
    }
}


const char * GetPyUtf8StrData(PyObject * obj, size_t & buf_len)
{
    // See: https://github.com/python/cpython/blob/3.9/Include/cpython/unicodeobject.h#L81
    if (PyUnicode_IS_COMPACT_ASCII(obj))
    {
        const char * data = reinterpret_cast<const char *>(PyUnicode_1BYTE_DATA(obj));
        buf_len = PyUnicode_GET_LENGTH(obj);
        return data;
    }
    else
    {
        PyCompactUnicodeObject * unicode = reinterpret_cast<PyCompactUnicodeObject *>(obj);
        if (unicode->utf8 != nullptr)
        {
            // It's utf8 string, treat it like ASCII
            const char * data = reinterpret_cast<const char *>(unicode->utf8);
            buf_len = unicode->utf8_length;
            return data;
        }
        else if (PyUnicode_IS_COMPACT(obj))
        {
            auto kind = PyUnicode_KIND(obj);
            /// We could not use the implementation provided by CPython like below because it requires GIL holded by the caller
            // if (kind == PyUnicode_1BYTE_KIND || kind == PyUnicode_2BYTE_KIND || kind == PyUnicode_4BYTE_KIND)
            // {
            //     // always convert it to utf8
            //     const char * data = PyUnicode_AsUTF8AndSize(obj, &unicode->utf8_length);
            //     buf_len = unicode->utf8_length;
            //     // set the utf8 buffer back
            //     unicode->utf8 = const_cast<char *>(data);
            //     return data;
            // }
            const char * data;
            size_t codepoint_cnt;

            if (kind == PyUnicode_1BYTE_KIND)
                data = reinterpret_cast<const char *>(PyUnicode_1BYTE_DATA(obj));
            else if (kind == PyUnicode_2BYTE_KIND)
                data = reinterpret_cast<const char *>(PyUnicode_2BYTE_DATA(obj));
            else if (kind == PyUnicode_4BYTE_KIND)
                data = reinterpret_cast<const char *>(PyUnicode_4BYTE_DATA(obj));
            else
                throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Unsupported unicode kind {}", kind);
            // always convert it to utf8, and we can't use as function provided by CPython because it requires GIL
            // holded by the caller. So we have to do it manually with libicu
            codepoint_cnt = PyUnicode_GET_LENGTH(obj);
            data = ConvertPyUnicodeToUtf8(data, kind, codepoint_cnt, buf_len);
            // set the utf8 buffer back like PyUnicode_AsUTF8AndSize does, so that we can reuse it
            // and also we can avoid the memory leak
            unicode->utf8 = const_cast<char *>(data);
            unicode->utf8_length = buf_len;
            return data;
        }
        else
        {
            // always convert it to utf8, but this case is rare, here goes the slow path
            py::gil_scoped_acquire acquire;
            // PyUnicode_AsUTF8AndSize caches the UTF-8 encoded string in the unicodeobject
            // and subsequent calls will return the same string.  The memory is released
            // when the unicodeobject is deallocated.
            Py_ssize_t bytes_size = -1;
            const char * data = PyUnicode_AsUTF8AndSize(obj, &bytes_size);
            if (bytes_size < 0)
                throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED, "Failed to convert Python unicode object to UTF-8");
            buf_len = bytes_size;
            return data;
        }
    }
}

bool _isInheritsFromPyReader(const py::handle & obj)
{
    // Check directly if obj is an instance of a class named "PyReader"
    if (py::str(obj.attr("__class__").attr("__name__")).cast<std::string>() == "PyReader")
        return true;

    // Check the direct base classes of obj's class for "PyReader"
    py::tuple bases = obj.attr("__class__").attr("__bases__");
    for (auto base : bases)
        if (py::str(base.attr("__name__")).cast<std::string>() == "PyReader")
            return true;

    return false;
}

// Will try to get the ref of py::array from pandas Series, or PyArrow Table
// without import numpy or pyarrow. Just from class name for now.
const void * tryGetPyArray(const py::object & obj, py::handle & result, std::string & type_name, size_t & row_count)
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
        result = array;
        return array.data();
    }
    else if (type_name == "Table")
    {
        // Try to get the handle of py::array from PyArrow Table
        py::array array = obj.attr("to_pandas")();
        row_count = py::len(obj);
        result = array;
        return array.data();
    }

    // chdb todo: maybe convert list to py::array?

    return nullptr;
}
}
