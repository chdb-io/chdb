#pragma once
#include <Core/Types.h>
#include <string_view>

namespace DB::Statistics
{
UInt64 stringHash64(std::string_view view);
}
