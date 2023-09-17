#include <Processors/QueryPlan/Hints/BroadcastJoin.h>
#include <Processors/QueryPlan/Hints/PlanHintFactory.h>


namespace DB
{

void registerHintBroadcastJoin(PlanHintFactory & factory)
{
    factory.registerPlanHint(BroadcastJoin::name, &BroadcastJoin::create, PlanHintFactory::CaseInsensitive);
}

}
