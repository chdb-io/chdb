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

#include <QueryPlan/PlanBuilder.h>
#include <QueryPlan/planning_common.h>
#include <QueryPlan/ProjectionStep.h>
#include <Optimizer/makeCastFunction.h>

namespace DB
{
void PlanBuilder::addStep(QueryPlanStepPtr step, PlanNodes children)
{
    plan = plan->addStep(id_allocator->nextId(), std::move(step), std::move(children));
}

Names PlanBuilder::translateToSymbols(ASTs & expressions) const
{
    Names symbols;
    symbols.reserve(expressions.size());
    for (auto & expr : expressions)
        symbols.push_back(translateToSymbol(expr));
    return symbols;
}

Names PlanBuilder::translateToUniqueSymbols(ASTs & expressions) const
{
    Names symbols;
    NameSet exists;
    for (auto & expr : expressions)
    {
        auto symbol = translateToSymbol(expr);
        if (exists.emplace(symbol).second)
            symbols.push_back(symbol);
    }
    return symbols;
}

Names PlanBuilder::applyProjection(ASTs & expressions)
{
    Names output_symbols;
    Assignments assignments;
    NameToType types;
    putIdentities(getOutputNamesAndTypes(), assignments, types);

    for (auto & expr : expressions)
    {
        String symbol = symbol_allocator->newSymbol(expr);
        assignments.emplace_back(symbol, translate(expr));
        types[symbol] = analysis.getExpressionType(expr);
        output_symbols.push_back(symbol);
    }

    auto project = std::make_shared<ProjectionStep>(getCurrentDataStream(), assignments, types);
    addStep(std::move(project));
    return output_symbols;
}

void PlanBuilder::appendProjection(ASTs & expressions)
{
    Assignments assignments;
    NameToType types;
    putIdentities(getOutputNamesAndTypes(), assignments, types);
    bool has_new_projection = false;
    AstToSymbol expression_to_symbols = createScopeAwaredASTMap<String>(analysis);

    for (auto & expr : expressions)
    {
        if (expression_to_symbols.find(expr) == expression_to_symbols.end() && !canTranslateToSymbol(expr))
        {
            String symbol = symbol_allocator->newSymbol(expr);
            assignments.emplace_back(symbol, translate(expr));
            types[symbol] = analysis.getExpressionType(expr);
            expression_to_symbols[expr] = symbol;
            has_new_projection = true;
        }
    }

    if (has_new_projection)
    {
        auto project = std::make_shared<ProjectionStep>(getCurrentDataStream(), assignments, types);
        addStep(std::move(project));
        withAdditionalMappings(expression_to_symbols);
    }
}

Names PlanBuilder::projectExpressionsWithCoercion(const ExpressionsAndTypes & expression_and_types)
{
    Assignments assignments;
    NameToType output_types;
    Names output_symbols;
    bool do_project = false;

    putIdentities(getCurrentDataStream().header, assignments, output_types);

    for (const auto & [expr, cast_type]: expression_and_types)
    {
        auto rewritten_expr = translation->translate(expr);

        // if an expression has been translated and no type coercion happens, just skip it
        if (!cast_type && rewritten_expr->as<ASTIdentifier>())
        {
            output_symbols.push_back(rewritten_expr->as<ASTIdentifier>()->name());
            continue;
        }

        if (cast_type)
        {
            rewritten_expr = makeCastFunction(rewritten_expr, cast_type);
        }

        auto output_symbol = symbol_allocator->newSymbol(rewritten_expr);

        assignments.emplace_back(output_symbol, rewritten_expr);
        output_types[output_symbol] = cast_type ? cast_type : analysis.getExpressionType(expr);
        output_symbols.push_back(output_symbol);
        do_project = true;
    }

    if (do_project)
    {
        auto casting_step = std::make_shared<ProjectionStep>(getCurrentDataStream(), assignments, output_types);
        addStep(std::move(casting_step));
    }

    return output_symbols;
}

}
