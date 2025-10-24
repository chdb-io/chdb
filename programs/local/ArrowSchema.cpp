#include "ArrowSchema.h"

#include <Processors/Formats/Impl/ArrowColumnToCHColumn.h>
#include <Formats/FormatFactory.h>
#include <arrow/c/bridge.h>

namespace DB
{

namespace ErrorCodes
{
extern const int BAD_ARGUMENTS;
}

}

using namespace DB;

namespace CHDB
{

ArrowSchemaWrapper::~ArrowSchemaWrapper()
{
    if (arrow_schema.release != nullptr)
    {
        arrow_schema.release(&arrow_schema);
        chassert(!arrow_schema.release);
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

void ArrowSchemaWrapper::convertArrowSchema(
    ArrowSchemaWrapper & schema,
    NamesAndTypesList & names_and_types,
    ContextPtr & context)
{
    if (!schema.arrow_schema.release)
    {
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "ArrowSchema is already released");
    }

    /// Import ArrowSchema to arrow::Schema
    auto arrow_schema_result = arrow::ImportSchema(&schema.arrow_schema);
    if (!arrow_schema_result.ok())
    {
        throw Exception(ErrorCodes::BAD_ARGUMENTS,
                        "Failed to import Arrow schema: {}", arrow_schema_result.status().message());
    }

    const auto & arrow_schema = arrow_schema_result.ValueOrDie();

    const auto format_settings = getFormatSettings(context);

    /// Convert Arrow schema to ClickHouse header
    auto block = ArrowColumnToCHColumn::arrowSchemaToCHHeader(
        *arrow_schema,
        nullptr,
        "Arrow",
        format_settings,
        format_settings.arrow.skip_columns_with_unsupported_types_in_schema_inference,
        format_settings.schema_inference_make_columns_nullable != 0,
        false,
        format_settings.parquet.allow_geoparquet_parser);

    for (const auto & column : block)
    {
        names_and_types.emplace_back(column.name, column.type);
    }
}

} // namespace CHDB
