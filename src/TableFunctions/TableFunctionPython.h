#pragma once

#include <Storages/StoragePython.h>
#include <TableFunctions/ITableFunction.h>
#include <Poco/Logger.h>
#include "Storages/ColumnsDescription.h"

namespace DB
{

class TableFunctionPython : public ITableFunction
{
public:
    static constexpr auto name = "python";
    std::string getName() const override { return name; }

private:
    Poco::Logger * logger = &Poco::Logger::get("TableFunctionPython");
    StoragePtr executeImpl(
        const ASTPtr & ast_function,
        ContextPtr context,
        const std::string & table_name,
        ColumnsDescription cached_columns,
        bool is_insert_query) const override;
    const char * getStorageTypeName() const override { return "Python"; }

    void parseArguments(const ASTPtr & ast_function, ContextPtr context) override;

    ColumnsDescription getActualTableStructure(ContextPtr context, bool is_insert_query) const override;
    std::shared_ptr<PyReader> reader;
};

}
