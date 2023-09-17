#pragma once

#include <Common/IFactoryWithAliases.h>
#include <Interpreters/Context.h>
#include <Parsers/IAST.h>
#include <Processors/QueryPlan/Hints/IPlanHint.h>

namespace DB
{

using PlanHintCreator = std::function<PlanHintPtr(const SqlHint &, const ContextMutablePtr)>;

class PlanHintFactory : private boost::noncopyable,
                              public IFactoryWithAliases<PlanHintCreator>

{
public:
    static PlanHintFactory & instance();

    /// Throws an exception if not found.
    PlanHintPtr get(const String & name, const ContextMutablePtr context, const SqlHint & hint) const;

    /// Returns nullptr if not found.
    PlanHintPtr tryGet(const String & name, const ContextMutablePtr context, const SqlHint & hint) const;

    /// Register a hint by its name.
    /// No locking, you must register all hints before usage of get.
    void registerPlanHint(
        const std::string & name,
        Value creator,
        CaseSensitiveness case_sensitiveness = CaseSensitive);

private:
    using HintCreators = std::unordered_map<std::string, Value>;

    HintCreators hints;
    HintCreators case_insensitive_hints;

    const HintCreators & getMap() const override { return hints; }
    const HintCreators & getCaseInsensitiveMap() const override { return case_insensitive_hints; }

    String getFactoryName() const override { return "PlanHintFactory"; }
};

}
