#include <unicode/bytestream.h>
#include "Common/logger_useful.h"
#include <Common/PythonUtils.h>

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

const char * GetPyUtf8StrData(const py::handle & obj, size_t & buf_len)
{
    // See: https://github.com/python/cpython/blob/3.9/Include/cpython/unicodeobject.h#L81
    if (PyUnicode_IS_COMPACT_ASCII(obj.ptr()))
    {
        const char * data = reinterpret_cast<const char *>(PyUnicode_1BYTE_DATA(obj.ptr()));
        buf_len = PyUnicode_GET_LENGTH(obj.ptr());
        return data;
    }
    else
    {
        PyCompactUnicodeObject * unicode = reinterpret_cast<PyCompactUnicodeObject *>(obj.ptr());
        if (unicode->utf8 != nullptr)
        {
            // It's utf8 string, treat it like ASCII
            const char * data = reinterpret_cast<const char *>(unicode->utf8);
            buf_len = unicode->utf8_length;
            return data;
        }
        else if (PyUnicode_IS_COMPACT(obj.ptr()))
        {
            auto kind = PyUnicode_KIND(obj.ptr());
            // if (kind == PyUnicode_1BYTE_KIND || kind == PyUnicode_2BYTE_KIND || kind == PyUnicode_4BYTE_KIND)
            // {
            //     // always convert it to utf8
            //     const char * data = PyUnicode_AsUTF8AndSize(obj.ptr(), &unicode->utf8_length);
            //     buf_len = unicode->utf8_length;
            //     // set the utf8 buffer back
            //     unicode->utf8 = const_cast<char *>(data);
            //     return data;
            // }
            const char * data;
            size_t codepoint_cnt;

            if (kind == PyUnicode_1BYTE_KIND)
                data = reinterpret_cast<const char *>(PyUnicode_1BYTE_DATA(obj.ptr()));
            else if (kind == PyUnicode_2BYTE_KIND)
                data = reinterpret_cast<const char *>(PyUnicode_2BYTE_DATA(obj.ptr()));
            else if (kind == PyUnicode_4BYTE_KIND)
                data = reinterpret_cast<const char *>(PyUnicode_4BYTE_DATA(obj.ptr()));
            else
                throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Unsupported unicode kind {}", kind);
            // always convert it to utf8, and we can't use as function provided by CPython because it requires GIL
            // holded by the caller. So we have to do it manually with libicu
            codepoint_cnt = PyUnicode_GET_LENGTH(obj.ptr());
            data = ConvertPyUnicodeToUtf8(data, kind, codepoint_cnt, buf_len);
            unicode->utf8 = const_cast<char *>(data);
            unicode->utf8_length = buf_len;
            return data;
        }
        else
        {
            // always convert it to utf8, but this case is rare, here goes the slow path
            py::gil_scoped_acquire acquire;
            const char * data = PyUnicode_AsUTF8AndSize(obj.ptr(), &unicode->utf8_length);
            buf_len = unicode->utf8_length;
            // set the utf8 buffer back
            unicode->utf8 = const_cast<char *>(data);
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
}
