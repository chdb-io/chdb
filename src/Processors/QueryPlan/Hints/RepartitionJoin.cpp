#include <Processors/QueryPlan/Hints/RepartitionJoin.h>
#include <Processors/QueryPlan/Hints/PlanHintFactory.h>


namespace DB
{

void registerHintRepartitionJoin(PlanHintFactory & factory)
{
    factory.registerPlanHint(RepartitionJoin::name, &RepartitionJoin::create, PlanHintFactory::CaseInsensitive);
}

}
