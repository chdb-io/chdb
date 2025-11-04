#include "NumpyNestedTypes.h"
#include "NumpyArray.h"
#include "FieldToPython.h"

#include <Columns/ColumnArray.h>
#include <Columns/ColumnNullable.h>
#include <Columns/ColumnTuple.h>
#include <Columns/ColumnMap.h>
#include <Columns/ColumnObject.h>
#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypeTuple.h>
#include <DataTypes/DataTypeMap.h>
#include <DataTypes/DataTypeObject.h>
#include <Common/typeid_cast.h>
#include <Common/Exception.h>
#include <DataTypes/DataTypeVariant.h>
#include <DataTypes/DataTypeDynamic.h>
#include <Columns/ColumnVariant.h>
#include <Columns/ColumnDynamic.h>
#include <Processors/Formats/Impl/CHColumnToArrowColumn.h>
#include <pybind11/pybind11.h>

namespace CHDB
{

using namespace DB;
namespace py = pybind11;

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
    extern const int NOT_IMPLEMENTED;
}

template <typename ColumnType>
struct ColumnTraits;

template <>
struct ColumnTraits<ColumnArray>
{
    using DataType = DataTypeArray;

    static py::object convertElement(const ColumnArray * column, const DataTypePtr & data_type, size_t index)
    {
        const auto & offsets = column->getOffsets();
        const auto & nested_column = column->getDataPtr();

        size_t start_offset = (index == 0) ? 0 : offsets[index - 1];
        size_t end_offset = offsets[index];
        size_t array_size = end_offset - start_offset;

        NumpyArray numpy_array(data_type);
        numpy_array.init(array_size);
        numpy_array.append(nested_column, start_offset, array_size);

        return numpy_array.toArray();
    }
};

template <>
struct ColumnTraits<ColumnTuple>
{
    using DataType = DataTypeTuple;

    static py::object convertElement(const ColumnTuple * column, const DataTypePtr & data_type, size_t index)
    {
        const auto * tuple_data_type = typeid_cast<const DataType *>(data_type.get());
        if (!tuple_data_type)
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected DataTypeTuple");

        const auto & element_types = tuple_data_type->getElements();
        size_t tuple_size = column->tupleSize();

        NumpyArray numpy_array({});
        numpy_array.init(tuple_size);

        for (size_t i = 0; i < tuple_size; ++i)
        {
            numpy_array.append(column->getColumn(i), element_types[i], index);
        }

        return numpy_array.toArray();
    }
};

template <>
struct ColumnTraits<ColumnMap>
{
    using DataType = DataTypeMap;

    static py::object convertElement(const ColumnMap * column, const DataTypePtr & data_type, size_t index)
    {
        return convertFieldToPython(*column, data_type, index);
    }
};

template <>
struct ColumnTraits<ColumnObject>
{
    using DataType = DataTypeObject;

    static py::object convertElement(const ColumnObject * column, const DataTypePtr & data_type, size_t index)
    {
        return convertFieldToPython(*column, data_type, index);
    }
};

template <>
struct ColumnTraits<ColumnVariant>
{
    using DataType = DataTypeVariant;

    static py::object convertElement(const ColumnVariant * column, const DataTypePtr & data_type, size_t index)
    {
        return convertFieldToPython(*column, data_type, index);
    }
};

template <>
struct ColumnTraits<ColumnDynamic>
{
    using DataType = DataTypeDynamic;

    static py::object convertElement(const ColumnDynamic * column, const DataTypePtr & data_type, size_t index)
    {
        return convertFieldToPython(*column, data_type, index);
    }
};

template <typename ColumnType>
bool CHNestedColumnToNumpyArray(NumpyAppendData & append_data, const DataTypePtr & data_type)
{
    bool has_null = false;
    const IColumn * data_column = &append_data.column;
    const ColumnNullable * nullable_column = nullptr;

    if (const auto * nullable = typeid_cast<const ColumnNullable *>(&append_data.column))
    {
        nullable_column = nullable;
        data_column = &nullable->getNestedColumn();
    }

    const auto * typed_column = typeid_cast<const ColumnType *>(data_column);
    if (!typed_column)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected specific column type");

    auto * dest_ptr = reinterpret_cast<py::object *>(append_data.target_data);
    auto * mask_ptr = append_data.target_mask;

    for (size_t i = append_data.src_offset; i < append_data.src_offset + append_data.src_count; i++)
    {
        size_t offset = append_data.dest_offset + i;
        if (nullable_column && nullable_column->isNullAt(i))
        {
            dest_ptr[offset] = py::none();
            mask_ptr[offset] = true;
            has_null = true;
        }
        else
        {
            dest_ptr[offset] = ColumnTraits<ColumnType>::convertElement(typed_column, data_type, i);
            mask_ptr[offset] = false;
        }
    }

    return has_null;
}

bool CHColumnArrayToNumpyArray(NumpyAppendData & append_data, const DataTypePtr & data_type)
{
    return CHNestedColumnToNumpyArray<ColumnArray>(append_data, data_type);
}

bool CHColumnTupleToNumpyArray(NumpyAppendData & append_data, const DataTypePtr & data_type)
{
    return CHNestedColumnToNumpyArray<ColumnTuple>(append_data, data_type);
}

bool CHColumnMapToNumpyArray(NumpyAppendData & append_data, const DataTypePtr & data_type)
{
    return CHNestedColumnToNumpyArray<ColumnMap>(append_data, data_type);
}

bool CHColumnObjectToNumpyArray(NumpyAppendData & append_data, const DataTypePtr & data_type)
{
    return CHNestedColumnToNumpyArray<ColumnObject>(append_data, data_type);
}

bool CHColumnVariantToNumpyArray(NumpyAppendData & append_data, const DataTypePtr & data_type)
{
    return CHNestedColumnToNumpyArray<ColumnVariant>(append_data, data_type);
}

bool CHColumnDynamicToNumpyArray(NumpyAppendData & append_data, const DataTypePtr & data_type)
{
    return CHNestedColumnToNumpyArray<ColumnDynamic>(append_data, data_type);
}

} // namespace CHDB
