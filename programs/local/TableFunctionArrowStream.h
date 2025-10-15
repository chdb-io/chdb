#pragma once

#include "ArrowStreamRegistry.h"

#include <TableFunctions/ITableFunction.h>
#include <Storages/ColumnsDescription.h>
#include <Poco/Logger.h>

namespace DB
{

class TableFunctionFactory;
void registerTableFunctionArrowStream(TableFunctionFactory & factory);

class TableFunctionArrowStream : public ITableFunction
{
public:
    static constexpr auto name = "arrowstream";
    std::string getName() const override { return name; }

private:
    Poco::Logger * logger = &Poco::Logger::get("TableFunctionArrowStream");

    StoragePtr executeImpl(
        const ASTPtr & ast_function,
        ContextPtr context,
        const std::string & table_name,
        ColumnsDescription cached_columns,
        bool is_insert_query) const override;

    const char * getStorageEngineName() const override { return "ArrowStream"; }

    void parseArguments(const ASTPtr & ast_function, ContextPtr context) override;

    ColumnsDescription getActualTableStructure(ContextPtr context, bool is_insert_query) const override;

    String stream_name;
    CHDB::ArrowStreamRegistry::ArrowStreamInfo stream_info;
};

}
