#include "ArrowStreamWrapper.h"
#include "PyArrowTable.h"
#include "PybindWrapper.h"
#include "PythonImporter.h"

#include <Common/Exception.h>
#include <base/defines.h>
#include <pybind11/gil.h>
#include <unordered_set>

namespace DB
{

namespace ErrorCodes
{
extern const int PY_EXCEPTION_OCCURED;
}

}

namespace py = pybind11;
using namespace DB;

namespace CHDB
{

/// ArrowSchemaWrapper implementation
ArrowSchemaWrapper::~ArrowSchemaWrapper()
{
    if (arrow_schema.release)
    {
        arrow_schema.release(&arrow_schema);
    }
}

ArrowSchemaWrapper::ArrowSchemaWrapper(ArrowSchemaWrapper && other) noexcept
    : arrow_schema(other.arrow_schema)
{
    other.arrow_schema.release = nullptr;
}

ArrowSchemaWrapper & ArrowSchemaWrapper::operator=(ArrowSchemaWrapper && other) noexcept
{
    if (this != &other)
    {
        if (arrow_schema.release)
        {
            arrow_schema.release(&arrow_schema);
        }
        arrow_schema = other.arrow_schema;
        other.arrow_schema.release = nullptr;
    }
    return *this;
}

/// ArrowArrayWrapper implementation
ArrowArrayWrapper::~ArrowArrayWrapper()
{
    if (arrow_array.release)
    {
        arrow_array.release(&arrow_array);
    }
}

ArrowArrayWrapper::ArrowArrayWrapper(ArrowArrayWrapper && other) noexcept
    : arrow_array(other.arrow_array)
{
    other.arrow_array.release = nullptr;
}

ArrowArrayWrapper & ArrowArrayWrapper::operator=(ArrowArrayWrapper && other) noexcept
{
    if (this != &other)
    {
        if (arrow_array.release)
        {
            arrow_array.release(&arrow_array);
        }
        arrow_array = other.arrow_array;
        other.arrow_array.release = nullptr;
    }
    return *this;
}

/// ArrowArrayStreamWrapper implementation
ArrowArrayStreamWrapper::~ArrowArrayStreamWrapper()
{
    if (arrow_array_stream.release)
    {
        arrow_array_stream.release(&arrow_array_stream);
    }
}

ArrowArrayStreamWrapper::ArrowArrayStreamWrapper(ArrowArrayStreamWrapper&& other) noexcept
    : arrow_array_stream(other.arrow_array_stream)
{
    other.arrow_array_stream.release = nullptr;
}

ArrowArrayStreamWrapper & ArrowArrayStreamWrapper::operator=(ArrowArrayStreamWrapper && other) noexcept
{
    if (this != &other)
    {
        if (arrow_array_stream.release)
        {
            arrow_array_stream.release(&arrow_array_stream);
        }
        arrow_array_stream = other.arrow_array_stream;
        other.arrow_array_stream.release = nullptr;
    }
    return *this;
}

void ArrowArrayStreamWrapper::getSchema(ArrowSchemaWrapper& schema)
{
    if (!isValid())
    {
        throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED, "ArrowArrayStream is not valid");
    }

    if (arrow_array_stream.get_schema(&arrow_array_stream, &schema.arrow_schema) != 0)
    {
        throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED,
                        "Failed to get schema from ArrowArrayStream: {}", getError());
    }

    if (!schema.arrow_schema.release)
    {
        throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED, "Released schema returned from ArrowArrayStream");
    }
}

std::unique_ptr<ArrowArrayWrapper> ArrowArrayStreamWrapper::getNextChunk()
{
    chassert(isValid());

    auto chunk = std::make_unique<ArrowArrayWrapper>();

    /// Get next non-empty chunk, skipping empty ones
    do
    {
        chunk->reset();
        if (arrow_array_stream.get_next(&arrow_array_stream, &chunk->arrow_array) != 0)
        {
            throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED,
                            "Failed to get next chunk from ArrowArrayStream: {}", getError());
        }

        /// Check if we've reached the end of the stream
        if (!chunk->arrow_array.release)
        {
            return nullptr;
        }
    }
    while (chunk->arrow_array.length == 0);

    return chunk;
}

const char* ArrowArrayStreamWrapper::getError()
{
    if (!isValid())
    {
        return "ArrowArrayStream is not valid";
    }

    return arrow_array_stream.get_last_error(&arrow_array_stream);
}

std::unique_ptr<ArrowArrayStreamWrapper> PyArrowStreamFactory::createFromPyObject(
    py::object & py_obj,
    const Names & column_names)
{
    py::gil_scoped_acquire acquire;

    try
    {
        auto arrow_object_type = PyArrowTable::getArrowType(py_obj);

        switch (arrow_object_type)
        {
            case PyArrowObjectType::Table:
                return createFromTable(py_obj, column_names);
            default:
                throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED,
                    "Unsupported PyArrow object type: {}", arrow_object_type);
        }
    }
    catch (const py::error_already_set & e)
    {
        throw Exception(ErrorCodes::PY_EXCEPTION_OCCURED,
                        "Failed to convert PyArrow object to arrow array stream: {}", e.what());
    }
}

std::unique_ptr<ArrowArrayStreamWrapper> PyArrowStreamFactory::createFromTable(
    py::object & table,
    const Names & column_names)
{
    chassert(py::gil_check());

    py::handle table_handle(table);
    auto & import_cache = PythonImporter::ImportCache();
    auto arrow_dataset = import_cache.pyarrow.dataset().attr("dataset");

	auto dataset = arrow_dataset(table_handle);
	py::object arrow_scanner = dataset.attr("__class__").attr("scanner");

	py::dict kwargs;
	if (!column_names.empty()) {
        ArrowSchemaWrapper schema;
        auto obj_schema = table_handle.attr("schema");
        auto export_to_c = obj_schema.attr("_export_to_c");
        export_to_c(reinterpret_cast<uint64_t>(&schema.arrow_schema));

        /// Get available column names from schema
        std::unordered_set<std::string> available_columns;
        if (schema.arrow_schema.n_children > 0 && schema.arrow_schema.children)
        {
            for (int64_t i = 0; i < schema.arrow_schema.n_children; ++i)
            {
                if (schema.arrow_schema.children[i] && schema.arrow_schema.children[i]->name)
                {
                    available_columns.insert(schema.arrow_schema.children[i]->name);
                }
            }
        }

        /// Only add column names that exist in the schema
        py::list projection_list;
        for (const auto & name : column_names)
        {
            if (available_columns.contains(name))
            {
                projection_list.append(name);
            }
        }

        /// Only set columns if we have valid projections
        if (projection_list.size() > 0)
        {
            kwargs["columns"] = projection_list;
        }
	}

	auto scanner = arrow_scanner(dataset, **kwargs);

    auto record_batches = scanner.attr("to_reader")();
	auto res = std::make_unique<ArrowArrayStreamWrapper>();
	auto export_to_c = record_batches.attr("_export_to_c");
	export_to_c(reinterpret_cast<uint64_t>(&res->arrow_array_stream));
	return res;
}

} // namespace CHDB
