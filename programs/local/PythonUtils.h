#pragma once

#include "config.h"
#include "PybindWrapper.h"

#include <Columns/ColumnString.h>
#include <Columns/IColumn.h>
#include <DataTypes/Serializations/SerializationNumber.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <Common/Exception.h>

namespace DB
{

namespace ErrorCodes
{
extern const int PY_EXCEPTION_OCCURED;
}

namespace py = pybind11;


struct ColumnWrapper
{
    void * buf; // we may modify the data when cast it to PyObject **, so we need a non-const pointer
    size_t row_count;
    py::handle data;
    py::handle tmp; // hold some tmp data like hits['Title'].astype("str")
    DataTypePtr dest_type;
    std::string py_type; //py::handle type, eg. numpy.ndarray;
    std::string row_format;
    std::string encoding; // utf8, utf16, utf32, etc.
    std::string name;

    ~ColumnWrapper()
    {
        py::gil_scoped_acquire acquire;
        if (!tmp.is_none())
        {
            tmp.dec_ref();
        }
    }
};

using PyObjectVec = std::vector<py::object>;
using PyObjectVecPtr = std::shared_ptr<PyObjectVec>;
using PyColumnVec = std::vector<ColumnWrapper>;
using PyColumnVecPtr = std::shared_ptr<PyColumnVec>;

// Template wrapper function to handle any return type
template <typename Func, typename... Args>
auto execWithGIL(Func func, Args &&... args) -> decltype(func(std::forward<Args>(args)...))
{
    py::gil_scoped_acquire acquire;
    return func(std::forward<Args>(args)...);
}

void FillColumnString(PyObject * obj, ColumnString * column);

// Helper function to check if an object's class is or inherits from PyReader with a maximum depth
bool _isInheritsFromPyReader(const py::handle & obj);

inline bool isInheritsFromPyReader(const py::object & obj)
{
    return execWithGIL([&]() { return _isInheritsFromPyReader(obj); });
}

// Helper function to check if object is a PyArrow Table
inline bool isPyarrowTable(const py::object & obj)
{
    chassert(py::gil_check());
    auto dict = py::module_::import("sys").attr("modules");
    if (!dict.contains(py::str("pyarrow")))
        return false;

    return py::isinstance(obj, py::module_::import("pyarrow").attr("Table"));
}

inline bool hasGetItem(const py::object & obj)
{
    chassert(py::gil_check());
    return py::hasattr(obj, "__getitem__");
}

// Specific wrappers for common use cases
inline auto castToPyList(const py::object & obj)
{
    return execWithGIL([&]() { return obj.cast<py::list>(); });
}

inline std::string castToStr(const py::object & obj)
{
    return execWithGIL([&]() { return py::str(obj).cast<std::string>(); });
}

inline std::string getPyType(const py::object & obj)
{
    return execWithGIL([&]() { return obj.get_type().attr("__name__").cast<std::string>(); });
}

template <typename T>
inline std::vector<T> castToVector(const py::object & obj)
{
    return execWithGIL([&]() { return obj.cast<std::vector<T>>(); });
}

inline std::vector<py::handle> castToPyHandleVector(const py::handle obj)
{
    return execWithGIL([&]() { return obj.cast<std::vector<py::handle>>(); });
}

template <typename T>
inline std::shared_ptr<std::vector<T>> castToSharedPtrVector(const py::object & obj)
{
    return execWithGIL([&]() { return std::make_shared<std::vector<T>>(obj.cast<std::vector<T>>()); });
}

inline size_t getObjectLength(const py::object & obj)
{
    return execWithGIL([&]() { return py::len(obj); });
}

inline py::object getValueByKey(const py::object & obj, const std::string & key)
{
    return execWithGIL([&]() { return obj[py::str(key)]; });
}

inline size_t getLengthOfValueByKey(const py::object & obj, const std::string & key)
{
    return execWithGIL([&]() { return py::len(obj[py::str(key)]); });
}

template <typename T>
inline T castObject(const py::object & obj)
{
    return execWithGIL([&]() { return obj.cast<T>(); });
}

inline bool hasAttribute(const py::object & obj, const char * attr_name)
{
    return execWithGIL([&]() { return py::hasattr(obj, attr_name); });
}

inline std::string getStringAttribute(const py::object & obj, const char * attr_name)
{
    return execWithGIL([&]() { return obj.attr(attr_name).cast<std::string>(); });
}

template <typename T>
inline bool isInstanceOf(const py::object & obj)
{
    return execWithGIL([&]() { return py::isinstance<T>(obj); });
}

inline size_t getPythonObjectLength(const py::object & obj)
{
    return execWithGIL([&]() { return py::len(obj); });
}

inline py::object getAttribute(const py::object & obj, const char * name)
{
    return execWithGIL([&]() { return obj.attr(name); });
}

inline py::object callMethod(const py::object & obj, const char * method_name)
{
    return execWithGIL([&]() { return obj.attr(method_name)(); });
}

inline std::vector<py::object> readData(const py::object & data_source, const std::vector<std::string> & names, size_t cursor, size_t count)
{
    return execWithGIL([&]() { return data_source.attr("read")(names, cursor, count).cast<std::vector<py::object>>(); });
}

const void * tryGetPyArray(const py::object & obj, py::handle & result, py::handle & tmp, std::string & type_name, size_t & row_count);

} // namespace DB
