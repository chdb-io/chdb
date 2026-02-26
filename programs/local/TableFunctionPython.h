#pragma once

#include "DataSourceWrapper.h"

#include <Storages/ColumnsDescription.h>
#include <TableFunctions/ITableFunction.h>
#include <Poco/Logger.h>

namespace DB
{

class TableFunctionFactory;
void registerTableFunctionPython(TableFunctionFactory & factory);

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
    const char * getStorageEngineName() const override { return ""; }

    void parseArguments(const ASTPtr & ast_function, ContextPtr context) override;

    ColumnsDescription getActualTableStructure(ContextPtr context, bool is_insert_query) const override;

    bool is_pandas_df = false;
    mutable CHDB::DataSourceWrapperPtr data_source_wrapper;
};

}
