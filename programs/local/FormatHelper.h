#pragma once

#include <base/types.h>

namespace CHDB {

void SetCurrentFormat(const char * format, size_t format_len);

bool isJSONSupported();

} // namespace CHDB
