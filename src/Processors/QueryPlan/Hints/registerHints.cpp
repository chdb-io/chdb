#include <Processors/QueryPlan/Hints/registerHints.h>
#include <Processors/QueryPlan/Hints/PlanHintFactory.h>


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
