#pragma once

#include "PybindWrapper.h"
#include "PythonUtils.h"
#include <DataTypes/IDataType.h>

namespace CHDB {

class PandasScan
{
public:
    static DB::ColumnPtr scanColumn(
        const DB::ColumnWrapper & col_wrap,
        const size_t cursor,
        const size_t count,
        const DB::FormatSettings & format_settings);

    static DB::ColumnPtr scanObject(
        const DB::ColumnWrapper & col_wrap,
        const size_t cursor,
        const size_t count,
        const DB::FormatSettings & format_settings);

    static void scanObject(
        const size_t cursor,
        const size_t count,
        const DB::FormatSettings & format_settings,
        const void * buf,
        DB::MutableColumnPtr & column);

private:
    static void innerCheck(const DB::ColumnWrapper & col_wrap);

    static void innerScanObject(
        const size_t cursor,
        const size_t count,
        const DB::FormatSettings & format_settings,
        DB::SerializationPtr & serialization,
        PyObject ** objects,
        DB::MutableColumnPtr & column,
        DB::WhichDataType which = DB::WhichDataType(DB::TypeIndex::Object));

    template <typename T>
    static void innerScanFloat(
        const size_t cursor,
        const size_t count,
        const T * ptr,
        DB::MutableColumnPtr & column);

    template <typename T>
    static void innerScanNumeric(
        const size_t cursor,
        const size_t count,
        const T * data_ptr,
        const bool * mask_ptr,
        DB::MutableColumnPtr & column);

    static void innerScanDateTime64(
        const size_t cursor,
        const size_t count,
        const Int64 * ptr,
        DB::MutableColumnPtr & column);
};

} // namespace CHDB
