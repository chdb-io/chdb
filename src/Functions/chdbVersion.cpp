#if defined(CHDB_VERSION_STRING)

#include <DataTypes/DataTypeString.h>
#include <Functions/FunctionFactory.h>
#include <Functions/FunctionConstantBase.h>

namespace DB
{

namespace
{
    /// chdb() - returns the current chdb version as a string.
    class FunctionChdbVersion : public FunctionConstantBase<FunctionChdbVersion, String, DataTypeString>
    {
    public:
        static constexpr auto name = "chdb";
        static FunctionPtr create(ContextPtr context) { return std::make_shared<FunctionChdbVersion>(context); }
        explicit FunctionChdbVersion(ContextPtr context) : FunctionConstantBase(CHDB_VERSION_STRING, context->isDistributed()) {}
    };
}
    
REGISTER_FUNCTION(ChdbVersion)
{
    factory.registerFunction<FunctionChdbVersion>(
        {
        R"(
Returns the version of chDB.  The result type is String.
        )",
        Documentation::Examples{{"chdb", "SELECT chdb()"}},
        Documentation::Categories{"String"}
        }, FunctionFactory::CaseInsensitive);
}
}
#endif
