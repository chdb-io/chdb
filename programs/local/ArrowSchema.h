#pragma once

#include <Core/NamesAndTypes.h>
#include <Interpreters/Context.h>
#include <arrow/c/abi.h>

namespace CHDB
{

/// Wrapper for Arrow C Data Interface structures with RAII resource management
class ArrowSchemaWrapper
{
public:
    ArrowSchema arrow_schema;

    ArrowSchemaWrapper() {
        arrow_schema.release = nullptr;
    }

    ~ArrowSchemaWrapper();

    /// Non-copyable but moveable
    ArrowSchemaWrapper(const ArrowSchemaWrapper &) = delete;
    ArrowSchemaWrapper & operator=(const ArrowSchemaWrapper &) = delete;
    ArrowSchemaWrapper(ArrowSchemaWrapper && other) noexcept;
    ArrowSchemaWrapper & operator=(ArrowSchemaWrapper && other) noexcept;

	static void convertArrowSchema(
		ArrowSchemaWrapper & schema,
		DB::NamesAndTypesList & names_and_types,
		DB::ContextPtr & context);
};

} // namespace CHDB
