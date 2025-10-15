#include "FormatHelper.h"

#include <algorithm>
#include <cctype>
#include <base/types.h>

namespace CHDB {

static bool is_json_supported = true;

void SetCurrentFormat(const char * format, size_t format_len)
{
    if (format)
    {
        String lower_format{format, format_len};
        std::transform(lower_format.begin(), lower_format.end(), lower_format.begin(), ::tolower);

        is_json_supported
            = !(lower_format == "arrow" || lower_format == "parquet" || lower_format == "arrowstream" || lower_format == "protobuf"
                || lower_format == "protobuflist" || lower_format == "protobufsingle");

        return;
    }

    is_json_supported = true;
}

bool isJSONSupported()
{
    return is_json_supported;
}

} // namespace CHDB
