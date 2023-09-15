#include <QueryPlan/Hints/registerHints.h>
#include <QueryPlan/Hints/PlanHintFactory.h>


namespace DB
{

void registerHintBroadcastJoin(PlanHintFactory & factory);
void registerHintRepartitionJoin(PlanHintFactory & factory);
void registerHintLeading(PlanHintFactory & factory);

void registerHints()
{
    auto & factory = PlanHintFactory::instance();

    registerHintBroadcastJoin(factory);
    registerHintLeading(factory);
    registerHintRepartitionJoin(factory);


}

}
