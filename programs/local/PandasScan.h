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
};

} // namespace CHDB
