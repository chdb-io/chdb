#include <QueryPlan/Hints/BroadcastJoin.h>
#include <QueryPlan/Hints/PlanHintFactory.h>


namespace DB
{

void registerHintBroadcastJoin(PlanHintFactory & factory)
{
    factory.registerPlanHint(BroadcastJoin::name, &BroadcastJoin::create, PlanHintFactory::CaseInsensitive);
}

}
