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

#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <Interpreters/predicateExpressionsUtils.h>
#include <Optimizer/ImplementSetOperation.h>
#include <Optimizer/Rule/Patterns.h>
#include <Optimizer/Rule/Rewrite/ImplementSetOperationRules.h>
#include <Optimizer/Utils.h>
#include <Parsers/ASTFunction.h>
#include <Processors/QueryPlan/ExceptStep.h>
#include <Processors/QueryPlan/FilterStep.h>
#include <Processors/QueryPlan/IntersectStep.h>

namespace DB
{
namespace ErrorCodes
{
    extern const int NOT_IMPLEMENTED;
}

PatternPtr ImplementExceptRule::getPattern() const
{
    return Patterns::except().result();
}

TransformResult ImplementExceptRule::transformImpl(PlanNodePtr node, const Captures &, RuleContext & rule_context)
{
    auto & context = *rule_context.context;
    if (!context.getSettingsRef().enable_setoperation_to_agg)
    {
        return {};
    }
    SetOperationNodeTranslator translator{context};

    const auto * step = dynamic_cast<const ExceptStep *>(node->getStep().get());

    /**
    * Converts EXCEPT DISTINCT queries into UNION ALL..GROUP BY...WHERE
    * E.g.:
    *     SELECT a FROM foo
    *     EXCEPT DISTINCT
    *     SELECT x FROM bar
    * =>
    *     SELECT a
    *     FROM
    *     (
    *         SELECT a,
    *         sum(foo_marker) AS foo_count,
    *         sum(bar_marker) AS bar_count
    *         FROM
    *         (
    *             SELECT a, 1 as foo_marker, 0 as bar_marker
    *             FROM foo
    *             UNION ALL
    *             SELECT x, 0 as foo_marker, 1 as bar_marker
    *             FROM bar
    *         ) T1
    *     GROUP BY a
    *     ) T2
    *     WHERE foo_count >= 1 AND bar_count = 0;
    */
    if (step->isDistinct())
    {
        auto translator_result = translator.makeSetContainmentPlanForDistinct(*node);

        // intersect predicate: the row must be present in every source
        ASTs greaters;
        greaters.emplace_back(makeASTFunction(
            "greaterOrEquals", ASTs{std::make_shared<ASTIdentifier>(translator_result.count_symbols[0]), std::make_shared<ASTLiteral>(1)}));
        for (size_t index = 1; index < translator_result.count_symbols.size(); ++index)
        {
            greaters.emplace_back(makeASTFunction(
                "equals", ASTs{std::make_shared<ASTIdentifier>(translator_result.count_symbols[index]), std::make_shared<ASTLiteral>(0)}));
        }

        auto predicate = composeAnd(greaters);

        auto filter_step = std::make_shared<FilterStep>(translator_result.plan_node->getStep()->getOutputStream(), predicate);
        PlanNodes children{translator_result.plan_node};
        PlanNodePtr filter_node = std::make_shared<FilterNode>(context.nextNodeId(), std::move(filter_step), children);
        return filter_node;
    }


    /**
     * Implement EXCEPT ALL using union, window and filter.
     * <p>
     * Transforms:
     * <pre>
     * - Except all
     *   output: a, b
     *     - Source1 (a1, b1)
     *     - Source2 (a2, b2)
     *     - Source3 (a3, b3)
     * </pre>
     * Into:
     * <pre>
     * - Project (prune helper symbols)
     *   output: a, b
     *     - Filter (row_number <= greatest(greatest(count1 - count2, 0) - count3, 0))
     *         - Window (partition by a, b)
     *           count1 <- count(marker1)
     *           count2 <- count(marker2)
     *           count3 <- count(marker3)
     *           row_number <- row_number()
     *               - Union
     *                 output: a, b, marker1, marker2, marker3
     *                   - Project (marker1 <- true, marker2 <- null, marker3 <- null)
     *                       - Source1 (a1, b1)
     *                   - Project (marker1 <- null, marker2 <- true, marker3 <- null)
     *                       - Source2 (a2, b2)
     *                   - Project (marker1 <- null, marker2 <- null, marker3 <- true)
     *                       - Source3 (a3, b3)
     * </pre>
     *
     * For example, the following SQL:
     *
     * <pre>
     * SELECT s_store_sk FROM store
     * Except ALL
     * SELECT s_store_sk FROM store
     * </pre>
     *
     * will be rewritten to:
     *
     * <pre>
     *
     * SELECT *
     * FROM
     * (
     *     SELECT
     *         s_store_sk,
     *         sum(a) OVER (PARTITION BY s_store_sk) AS t1,
     *         sum(b) OVER (PARTITION BY s_store_sk) AS t2,
     *         row_number() OVER (PARTITION BY s_store_sk) AS t3
     *     FROM
     *     (
     *          SELECT
     *             s_store_sk,
     *             a,
     *             b
     *          FROM
     *          (
     *              SELECT
     *                  s_store_sk,
     *                  1 AS a,
     *                  0 AS b
     *              FROM store
     *              UNION ALL
     *              SELECT
     *                  s_store_sk,
     *                  0 AS a,
     *                  1 AS b
     *              FROM store
     *          )
     *      )
     * )
     * WHERE (t1 >=1 and t2 = 0)
     *
     * </pre>
     */
    auto translator_result = translator.makeSetContainmentPlanForDistinctAll(*node);

    Utils::checkState(!translator_result.count_symbols.empty(), "ExceptNode translation result has no count symbols");

    // filter rows so that expected number of rows remains
    ASTPtr remove_extra_rows = makeASTFunction("greaterOrEquals", ASTs{std::make_shared<ASTIdentifier>(translator_result.count_symbols[0]), std::make_shared<ASTLiteral>(1u)});
    for (size_t i = 1; i < translator_result.count_symbols.size(); i++)
    {
        ASTPtr except_check = makeASTFunction("equals", ASTs{std::make_shared<ASTIdentifier>(translator_result.count_symbols[i]), std::make_shared<ASTLiteral>(0u)});
        remove_extra_rows = makeASTFunction("and", ASTs{remove_extra_rows, except_check});
    }

    auto filter_step = std::make_shared<FilterStep>(translator_result.plan_node->getStep()->getOutputStream(), remove_extra_rows);
    auto filter_node = PlanNodeBase::createPlanNode(
        context.getPlanNodeIdAllocator()->nextId(), std::move(filter_step), PlanNodes{translator_result.plan_node});

    // prune helper symbols
    Assignments assignments;
    NameToType name_to_type;
    for (const auto & item : node->getStep()->getOutputStream().header)
    {
        assignments.emplace_back(item.name, std::make_shared<ASTIdentifier>(item.name));
        name_to_type[item.name] = item.type;
    }

    auto project_step = std::make_shared<ProjectionStep>(filter_node->getStep()->getOutputStream(), assignments, name_to_type);
    auto project_node
        = PlanNodeBase::createPlanNode(context.getPlanNodeIdAllocator()->nextId(), std::move(project_step), PlanNodes{filter_node});
    return project_node;
}

PatternPtr ImplementIntersectRule::getPattern() const
{
    return Patterns::intersect().result();
}

TransformResult ImplementIntersectRule::transformImpl(PlanNodePtr node, const Captures &, RuleContext & rule_context)
{
    auto & context = *rule_context.context;
    if (!context.getSettingsRef().enable_setoperation_to_agg)
    {
        return {};
    }
    SetOperationNodeTranslator translator{context};

    const auto * step = dynamic_cast<const IntersectStep *>(node->getStep().get());

    /**
     * Converts INTERSECT DISTINCT queries into UNION ALL..GROUP BY...WHERE
     * E.g.:
     *     SELECT a FROM foo
     *     INTERSECT DISTINCT
     *     SELECT x FROM bar
     * =>
     *     SELECT a
     *     FROM
     *     (
     *         SELECT a,
     *         sum(foo_marker) AS foo_count,
     *         sum(bar_marker) AS bar_count
     *         FROM
     *         (
     *             SELECT a, 1 as foo_marker, 0 as bar_marker
     *             FROM foo
     *             UNION ALL
     *             SELECT x, 0 as foo_marker, 1 as bar_marker
     *             FROM bar
     *         ) T1
     *     GROUP BY a
     *     ) T2
     *     WHERE foo_count >= 1 AND bar_count >= 1;
     * </pre>
     */
    if (step->isDistinct())
    {
        auto translator_result = translator.makeSetContainmentPlanForDistinct(*node);

        // intersect predicate: the row must be present in every source
        ASTs greaters;
        for (const auto & item : translator_result.count_symbols)
            greaters.emplace_back(
                makeASTFunction("greaterOrEquals", ASTs{std::make_shared<ASTIdentifier>(item), std::make_shared<ASTLiteral>(1)}));

        auto predicate = composeAnd(greaters);

        auto filter_step = std::make_shared<FilterStep>(translator_result.plan_node->getStep()->getOutputStream(), predicate);
        PlanNodes children{translator_result.plan_node};
        PlanNodePtr filter_node = std::make_shared<FilterNode>(context.nextNodeId(), std::move(filter_step), children);
        return filter_node;
    }


    /**
     * Implement INTERSECT ALL using union, window and filter.
     * <p>
     * Transforms:
     * <pre>
     * - Intersect all
     *   output: a, b
     *     - Source1 (a1, b1)
     *     - Source2 (a2, b2)
     *     - Source3 (a3, b3)
     * </pre>
     * Into:
     * <pre>
     * - Project (prune helper symbols)
     *   output: a, b
     *     - Filter (row_number <= least(least(count1, count2), count3))
     *         - Window (partition by a, b)
     *           count1 <- count(marker1)
     *           count2 <- count(marker2)
     *           count3 <- count(marker3)
     *           row_number <- row_number()
     *               - Union
     *                 output: a, b, marker1, marker2, marker3
     *                   - Project (marker1 <- true, marker2 <- null, marker3 <- null)
     *                       - Source1 (a1, b1)
     *                   - Project (marker1 <- null, marker2 <- true, marker3 <- null)
     *                       - Source2 (a2, b2)
     *                   - Project (marker1 <- null, marker2 <- null, marker3 <- true)
     *                       - Source3 (a3, b3)
     * </pre>
     *
     * For example, the following SQL:
     *
     * <pre>
     * SELECT s_store_sk FROM store
     * Intersect ALL
     * SELECT s_store_sk FROM store
     * </pre>
     *
     * will be rewritten to:
     *
     * <pre>
     *
     * SELECT *
     * FROM
     * (
     *     SELECT
     *         s_store_sk,
     *         sum(a) OVER (PARTITION BY s_store_sk) AS t1,
     *         sum(b) OVER (PARTITION BY s_store_sk) AS t2,
     *         row_number() OVER (PARTITION BY s_store_sk) AS t3
     *     FROM
     *     (
     *          SELECT
     *             s_store_sk,
     *             a,
     *             b
     *          FROM
     *          (
     *              SELECT
     *                  s_store_sk,
     *                  1 AS a,
     *                  0 AS b
     *              FROM store
     *              UNION ALL
     *              SELECT
     *                  s_store_sk,
     *                  0 AS a,
     *                  1 AS b
     *              FROM store
     *          )
     *      )
     * )
     * WHERE t1 >=1 and t2 >=1 and t1 >= row_number() 
     *
     * </pre>
     */
    auto translator_result = translator.makeSetContainmentPlanForDistinctAll(*node);

    Utils::checkState(!translator_result.count_symbols.empty(), "IntersectNode translation result has no count symbols");

    // filter rows so that expected number of rows remains
    ASTPtr remove_extra_rows = makeASTFunction("greaterOrEquals", ASTs{std::make_shared<ASTIdentifier>(translator_result.count_symbols[0]), std::make_shared<ASTLiteral>(1u)});
    for (size_t i = 1; i < translator_result.count_symbols.size(); i++)
    {
        ASTPtr intersect_check = makeASTFunction("greaterOrEquals", ASTs{std::make_shared<ASTIdentifier>(translator_result.count_symbols[i]), std::make_shared<ASTLiteral>(1u)});
        remove_extra_rows = makeASTFunction("and", ASTs{remove_extra_rows, intersect_check});
    }

    ASTPtr row_number_check 
        = makeASTFunction("greaterOrEquals", ASTs{std::make_shared<ASTIdentifier>(translator_result.count_symbols[0]), std::make_shared<ASTIdentifier>(translator_result.row_number_symbol.value())});
    
    remove_extra_rows = makeASTFunction("and", ASTs{remove_extra_rows, row_number_check});
    auto filter_step = std::make_shared<FilterStep>(translator_result.plan_node->getStep()->getOutputStream(), remove_extra_rows);
    auto filter_node = PlanNodeBase::createPlanNode(
        context.getPlanNodeIdAllocator()->nextId(), std::move(filter_step), PlanNodes{translator_result.plan_node});

    // prune helper symbols
    Assignments assignments;
    NameToType name_to_type;
    for (const auto & item : node->getStep()->getOutputStream().header)
    {
        assignments.emplace_back(item.name, std::make_shared<ASTIdentifier>(item.name));
        name_to_type[item.name] = item.type;
    }

    auto project_step = std::make_shared<ProjectionStep>(filter_node->getStep()->getOutputStream(), assignments, name_to_type);
    auto project_node
        = PlanNodeBase::createPlanNode(context.getPlanNodeIdAllocator()->nextId(), std::move(project_step), PlanNodes{filter_node});
    return project_node;
}

}
