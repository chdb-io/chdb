#pragma once

#include "PybindWrapper.h"
#include "PythonUtils.h"

namespace CHDB {

class PandasScan {
public:
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
        DB::MutableColumnPtr & column);
};

} // namespace CHDB
