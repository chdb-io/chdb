#pragma once

#include <Core/Block.h>

#include <Core/ExternalResultDescription.h>
#include <Processors/ISource.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <Poco/Logger.h>

namespace DB
{
class PyReader;

namespace py = pybind11;
class PythonSource : public ISource
{
public:
    PythonSource(py::object reader_, const Block & sample_block_, UInt64 max_block_size_);
    ~PythonSource() override
    {
        // Acquire the GIL before destroying the reader object
        py::gil_scoped_acquire acquire;
        reader.dec_ref();
        reader.release();
    }

    String getName() const override { return "Python"; }
    Chunk generate() override;

    static const char * getPyUtf8StrData(const py::handle & obj, size_t & buf_len)
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
            if (unicode->utf8)
            {
                // It's utf8 string, treat it like ASCII
                const char * data = reinterpret_cast<const char *>(unicode->utf8);
                buf_len = unicode->utf8_length;
                return data;
            }
            else if (PyUnicode_IS_COMPACT(obj.ptr()))
            {
                auto kind = PyUnicode_KIND(obj.ptr());
                if (kind == PyUnicode_1BYTE_KIND || kind == PyUnicode_2BYTE_KIND || kind == PyUnicode_4BYTE_KIND)
                {
                    // always convert it to utf8
                    const char * data = PyUnicode_AsUTF8AndSize(obj.ptr(), &unicode->utf8_length);
                    buf_len = unicode->utf8_length;
                    // set the utf8 buffer back
                    unicode->utf8 = const_cast<char *>(data);
                    return data;
                }
                else
                {
                    throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Unsupported unicode kind {}", kind);
                }
            }
            else
            {
                // always convert it to utf8
                const char * data = PyUnicode_AsUTF8AndSize(obj.ptr(), &unicode->utf8_length);
                buf_len = unicode->utf8_length;
                // set the utf8 buffer back
                unicode->utf8 = const_cast<char *>(data);
                return data;
            }
        }
    }

private:
    py::object reader;
    Block sample_block;
    const UInt64 max_block_size;
    Poco::Logger * logger = &Poco::Logger::get("TableFunctionPython");
    ExternalResultDescription description;
};
}
