#include "ObjectToPython.h"
#include "FieldToPython.h"

#include <Columns/ColumnObject.h>
#include <Columns/ColumnDynamic.h>
#include <DataTypes/DataTypeObject.h>
#include <base/defines.h>

namespace DB
{
namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}
}

namespace CHDB
{

using namespace DB;
namespace py = pybind11;

struct PathElements
{
    explicit PathElements(const String & path)
    {
        const char * start = path.data();
        const char * end = start + path.size();
        const char * pos = start;
        const char * last_dot_pos = pos - 1;
        for (pos = start; pos != end; ++pos)
        {
            if (*pos == '.')
            {
                elements.emplace_back(last_dot_pos + 1, size_t(pos - last_dot_pos - 1));
                last_dot_pos = pos;
            }
        }

        elements.emplace_back(last_dot_pos + 1, size_t(pos - last_dot_pos - 1));
    }

    size_t size() const { return elements.size(); }

    std::vector<std::string_view> elements;
};

py::object convertObjectToPython(
    const IColumn & column,
    const DataTypePtr & type,
    size_t index)
{
    const auto & column_object = typeid_cast<const ColumnObject &>(column);
    const auto & typed_paths = column_object.getTypedPaths();
    const auto & dynamic_paths = column_object.getDynamicPaths();
    const auto & shared_data_offsets = column_object.getSharedDataOffsets();
    const auto [shared_data_paths, shared_data_values] = column_object.getSharedDataPathsAndValues();

    size_t shared_data_offset = shared_data_offsets[static_cast<ssize_t>(index) - 1];
    size_t shared_data_end = shared_data_offsets[static_cast<ssize_t>(index)];

    const auto & object_type = typeid_cast<const DataTypeObject &>(type);
    const auto & specific_typed_paths = object_type.getTypedPaths();
    const auto & dynamic_data_type = object_type.getDynamicType();

    std::vector<std::pair<String, py::object>> path_values;
    path_values.reserve(typed_paths.size() + dynamic_paths.size() + (shared_data_end - shared_data_offset));

    for (const auto & [path, column_ptr] : typed_paths)
    {
        auto iter = specific_typed_paths.find(path);
        if (iter == specific_typed_paths.end())
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Path {} not found in typed paths", path);

        const auto & specific_data_type = iter->second;
        auto python_value = convertFieldToPython(*column_ptr, specific_data_type, index);
        path_values.emplace_back(path, python_value);
    }

    for (const auto & [path, dynamic_column] : dynamic_paths)
    {
        if (!dynamic_column->isNullAt(index))
        {
            auto python_value = convertFieldToPython(*dynamic_column, dynamic_data_type, index);
            path_values.emplace_back(path, python_value);
        }
    }

    size_t index_in_shared_data_values = shared_data_offset;
    for (size_t i = shared_data_offset; i != shared_data_end; ++i)
    {
        auto path = shared_data_paths->getDataAt(i).toString();

        auto tmp_dynamic_column = ColumnDynamic::create();
        tmp_dynamic_column->reserve(1);
        ColumnObject::deserializeValueFromSharedData(shared_data_values, index_in_shared_data_values++, *tmp_dynamic_column);

        auto python_value = convertFieldToPython(*tmp_dynamic_column, dynamic_data_type, 0);
        path_values.emplace_back(path, python_value);
    }

    py::dict result;

    for (const auto & [path, value] : path_values)
    {
        PathElements path_elements(path);

        if (path_elements.size() == 1)
        {
            String key(path_elements.elements[0]);
            result[key.c_str()] = value;
        }
        else
        {
            py::dict * current_dict = &result;

            for (size_t i = 0; i < path_elements.size() - 1; ++i)
            {
                String key(path_elements.elements[i]);

                if (current_dict->contains(key.c_str()))
                {
                    py::object nested = (*current_dict)[key.c_str()];
                    current_dict = &nested.cast<py::dict &>();
                }
                else
                {
                    py::dict new_dict;
                    (*current_dict)[key.c_str()] = new_dict;
                    current_dict = &new_dict;
                }
            }

            chassert(current_dict);
            String final_key(path_elements.elements[path_elements.size() - 1]);
            (*current_dict)[final_key.c_str()] = value;
        }
    }

    return result;
}

} // namespace CHDB
