/*
 * Copyright (2022) Bytedance Ltd. and/or its affiliates
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <Optimizer/Rule/Transformation/InlineCTE.h>

#include <Optimizer/Cascades/CascadesOptimizer.h>
#include <Optimizer/Iterative/IterativeRewriter.h>
#include <Optimizer/Rewriter/PredicatePushdown.h>
#include <Optimizer/Rule/Patterns.h>
#include <Optimizer/Rule/Rules.h>
#include <QueryPlan/CTERefStep.h>

namespace DB
{
PatternPtr InlineCTE::getPattern() const
{
    return Patterns::cte().result();
}

TransformResult InlineCTE::transformImpl(PlanNodePtr node, const Captures &, RuleContext & context)
{
    const auto * cte_step = dynamic_cast<const CTERefStep *>(node->getStep().get());
    if (cte_step->hasFilter())
        return {}; // InlineCTEWithFilter
    return {cte_step->toInlinedPlanNode(context.cte_info, context.context)};
}

PatternPtr InlineCTEWithFilter::getPattern() const
{
    return Patterns::filter().withSingle(Patterns::cte()).result();
}

TransformResult InlineCTEWithFilter::transformImpl(PlanNodePtr node, const Captures &, RuleContext & context)
{
    auto cte = node->getChildren()[0];
    const auto * cte_step = dynamic_cast<const CTERefStep *>(cte->getStep().get());
    if (!cte_step->hasFilter())
        return {}; // InlineCTE

    auto inlined_plan
        = PlanNodeBase::createPlanNode(node->getId(), node->getStep(), {cte_step->toInlinedPlanNode(context.cte_info, context.context)});
    return {predicatePushDown(inlined_plan, context.cte_info, context.context)};
}

PlanNodePtr InlineCTEWithFilter::predicatePushDown(const PlanNodePtr & node, CTEInfo & cte_info, ContextMutablePtr & context)
{
    static Rewriters rewriters
        = {std::make_shared<PredicatePushdown>(),
           std::make_shared<IterativeRewriter>(Rules::inlineProjectionRules(), "InlineProjection"),
           std::make_shared<IterativeRewriter>(Rules::normalizeExpressionRules(), "NormalizeExpression"),
           std::make_shared<IterativeRewriter>(Rules::swapPredicateRules(), "SwapPredicate"),
           std::make_shared<IterativeRewriter>(Rules::simplifyExpressionRules(), "SimplifyExpression"),
           std::make_shared<IterativeRewriter>(Rules::removeRedundantRules(), "RemoveRedundant")};

    QueryPlan sub_plan{node, cte_info, context->getPlanNodeIdAllocator()};
    for (auto & rewriter : rewriters)
        rewriter->rewrite(sub_plan, context);
    return sub_plan.getPlanNode();
}
}
