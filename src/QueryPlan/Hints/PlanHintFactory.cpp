#include <QueryPlan/Hints/PlanHintFactory.h>
#include <Common/IFactoryWithAliases.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>


namespace DB
{


namespace ErrorCodes
{
    extern const int UNKNOWN_HINT;
    extern const int LOGICAL_ERROR;
}

void PlanHintFactory::registerPlanHint(const
                                       std::string & name,
                                       Value creator,
                                       CaseSensitiveness case_sensitiveness)
{
    if (!hints.emplace(name, creator).second)
        throw Exception("PlanHintFactory: the hint name '" + name + "' is not unique",
                        ErrorCodes::LOGICAL_ERROR);

    String hint_name_lowercase = Poco::toLower(name);
    if (isAlias(name) || isAlias(hint_name_lowercase))
        throw Exception("PlanHintFactory: the hint name '" + name + "' is already registered as alias",
                        ErrorCodes::LOGICAL_ERROR);

    if (case_sensitiveness == CaseInsensitive
        && !case_insensitive_hints.emplace(hint_name_lowercase, creator).second)
        throw Exception("PlanHintFactory: the case insensitive hint name '" + name + "' is not unique",
                        ErrorCodes::LOGICAL_ERROR);
}

PlanHintPtr PlanHintFactory::get(
    const std::string & name,
    const ContextMutablePtr context,
    const SqlHint & hint) const
{
    auto res = tryGet(name, context, hint);
    if (!res)
    {
        throw Exception("Unknown hint " + name, ErrorCodes::UNKNOWN_HINT);
    }
    return res;
}

PlanHintPtr PlanHintFactory::tryGet(
    const std::string & name_param,
    const ContextMutablePtr context, const SqlHint & hint) const
{
    String name = getAliasToOrName(name_param);
    auto it = hints.find(name);
    if (hints.end() != it)
        return it->second(hint, context);

    it = case_insensitive_hints.find(Poco::toLower(name));
    if (case_insensitive_hints.end() != it)
        return it->second(hint, context);

    return {};
}

PlanHintFactory & PlanHintFactory::instance()
{
    static PlanHintFactory ret;
    return ret;
}


}
