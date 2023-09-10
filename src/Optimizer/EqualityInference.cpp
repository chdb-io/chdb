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

#include <Optimizer/EqualityInference.h>
#include <Optimizer/ExpressionDeterminism.h>
#include <Optimizer/ExpressionRewriter.h>
#include <Optimizer/PredicateUtils.h>
#include <Parsers/ASTFunction.h>

namespace DB
{
EqualityInference EqualityInference::newInstance(const ConstASTPtr & predicate, ContextMutablePtr & context)
{
    return newInstance(std::vector{predicate}, context);
}

EqualityInference EqualityInference::newInstance(const std::vector<ConstASTPtr> & predicates, ContextMutablePtr & context)
{
    DisjointSet equalities;
    for (const auto & predicate : predicates)
    {
        auto conjuncts = PredicateUtils::extractConjuncts(predicate);
        for (auto & conjunct : conjuncts)
        {
            if (isInferenceCandidate(conjunct, context))
            {
                const auto & fun = conjunct->as<ASTFunction &>();
                ASTPtr left = fun.arguments->getChildren()[0];
                ASTPtr right = fun.arguments->getChildren()[1];
                equalities.findAndUnion(ConstHashAST::make(left), ConstHashAST::make(right));
            }
        }
    }

    ConstASTSet derived_expressions;

    auto equality_sets = makeEqualitySets(equalities);
    ConstASTMap canonical_mappings;

    for (auto & equality_set : equality_sets)
    {
        for (const auto & value : equality_set.second)
        {
            canonical_mappings[value] = equality_set.first;
        }
    }

    return EqualityInference(equality_sets, std::move(canonical_mappings), {});
}

bool EqualityInference::isInferenceCandidate(const ConstASTPtr & predicate, ContextMutablePtr & context)
{
    if (predicate->as<ASTFunction>())
    {
        const auto & fun = predicate->as<ASTFunction &>();
        if (fun.name == "equals" && !mayReturnNullOnNonNullInput(fun) && ExpressionDeterminism::isDeterministic(predicate, context))
        {
            // We should only consider equalities that have distinct left and right components
            if (fun.arguments->getChildren()[0]->getColumnName() != fun.arguments->getChildren()[1]->getColumnName())
            {
                return true;
            }

            if (fun.arguments->getChildren()[0]->getType() != fun.arguments->getChildren()[1]->getType())
            {
                return true;
            }
        }
    }
    return false;
}

// TODO: this should look at whether the return type of the function is Nullable
bool EqualityInference::mayReturnNullOnNonNullInput(const ASTFunction & predicate)
{
    // for example: a = upper(name), we consider arguments with function all may result null value.
    for (auto & argument : predicate.arguments->getChildren())
    {
        if (argument->as<ASTFunction>())
        {
            return true;
        }
    }
    return false;
}

EqualityASTMap<ConstASTSet> EqualityInference::makeEqualitySets(DisjointSet equalities)
{
    EqualityASTMap<ConstASTSet> equality_sets;
    auto equivalent_classes = equalities.getEquivalentClasses();
    for (auto & equivalent_class : equivalent_classes)
    {
        auto key = getMin(equivalent_class);
        equality_sets.emplace(key, std::move(equivalent_class));
    }
    return equality_sets;
}

ConstHashAST EqualityInference::getMin(ConstASTSet & equivalences)
{
    auto min_iter = std::min_element(equivalences.begin(), equivalences.end(), Utils::ConstASTPtrOrdering());
    return *min_iter;
}

std::vector<ConstASTPtr> EqualityInference::nonInferrableConjuncts(const ConstASTPtr & expression, ContextMutablePtr & context)
{
    std::vector<ConstASTPtr> non_inferrable_conjuncts;
    auto conjuncts = PredicateUtils::extractConjuncts(expression);
    for (auto & conjunct : conjuncts)
    {
        if (!isInferenceCandidate(conjunct, context))
        {
            non_inferrable_conjuncts.emplace_back(conjunct);
        }
    }
    return non_inferrable_conjuncts;
}

ASTPtr EqualityInference::rewrite(const ConstASTPtr & expression, const std::set<String> & scope)
{
    return rewrite(expression, scope, true, true);
}

ASTPtr EqualityInference::rewrite(const ConstASTPtr & expression, const std::set<String> & scope, bool contains, bool allow_full_replacement)
{
    ConstASTSet sub_expressions = SubExpressionExtractor::extract(expression);
    if (!allow_full_replacement)
    {
        ConstASTSet sub_expressions_remove_itself;
        for (const auto & sub_expression : sub_expressions)
        {
            if (!ASTEquality::compareTree(sub_expression, expression))
            {
                sub_expressions_remove_itself.emplace(sub_expression);
            }
        }
        sub_expressions = sub_expressions_remove_itself;
    }
    ConstASTMap expression_remap;
    for (const auto & sub_expression : sub_expressions)
    {
        auto canonical = getScopedCanonical(sub_expression, scope, contains);
        if (canonical != nullptr)
        {
            expression_remap[sub_expression] = canonical;
        }
    }

    // Perform a naive single-pass traversal to try to rewrite non-compliant portions of the tree. Prefers to replace
    // larger subtrees over smaller subtrees
    // TODO: this rewrite can probably be made more sophisticated
    ASTPtr rewritten = ExpressionRewriter::rewrite(expression, expression_remap);
    if (contains)
    {
        if (!isScoped(rewritten, scope))
        {
            // If the rewritten is still not compliant with the symbol scope, just give up
            return nullptr;
        }
        return rewritten;
    }
    else
    {
        if (!isNotScoped(rewritten, scope))
        {
            // If the rewritten is still not compliant with the symbol scope, just give up
            return nullptr;
        }
        return rewritten;
    }
}

ConstHashAST EqualityInference::getScopedCanonical(const ConstHashAST & expression, const std::set<String> & scope, bool contains)
{
    if (!canonical_map.contains(expression))
    {
        return nullptr;
    }

    auto & canonical_index = canonical_map[expression];
    const ConstASTSet & equivalences = equality_sets[canonical_index];
    if (expression->as<ASTIdentifier>())
    {
        bool in_scope = false;
        for (const auto & equivalence : equivalences)
        {
            if (contains && scope.contains(equivalence->getColumnName()))
            {
                in_scope = true;
            }
        }

        if (!in_scope)
        {
            return nullptr;
        }
    }

    ConstASTSet candidates;
    for (const auto & equivalence : equivalences)
    {
        if (isScoped(equivalence, scope))
        {
            candidates.emplace(equivalence);
        }
    }

    return getCanonical(candidates);
}

ConstHashAST EqualityInference::getCanonical(ConstASTSet & equivalences)
{
    if (equivalences.empty())
        return nullptr;
    return getMin(equivalences);
}

bool EqualityInference::isScoped(const ConstASTPtr & equivalence, const std::set<String> & scope)
{
    std::set<String> symbols = SymbolsExtractor::extract(equivalence);
    return std::all_of(symbols.begin(), symbols.end(), [&](auto & symbol) { return scope.contains(symbol); });
}

bool EqualityInference::isNotScoped(const ConstASTPtr & equivalence, const std::set<String> & scope)
{
    std::set<String> symbols = SymbolsExtractor::extract(equivalence);
    return std::none_of(symbols.begin(), symbols.end(), [&](auto & symbol) { return scope.contains(symbol); });
}

EqualityPartition EqualityInference::partitionedBy(const std::set<String>& scope)
{
    std::vector<ConstASTPtr> scope_equalities;
    std::vector<ConstASTPtr> scope_complement_equalities;
    std::vector<ConstASTPtr> scope_straddling_equalities;

    for (auto & equality_set : equality_sets)
    {
        ConstASTSet scope_expressions;
        ConstASTSet scope_complement_expressions;
        ConstASTSet scope_straddling_expressions;

        ConstASTSet & values = equality_set.second;
        // Try to push each non-derived expression into one side of the scope
        std::vector<ConstASTPtr> candidates;
        for (const auto & value : values)
        {
            if (!derived_expressions.contains(value))
            {
                candidates.emplace_back(value);
            }
        }
        for (auto & candidate : candidates)
        {
            ConstASTPtr scope_rewritten = rewrite(candidate, scope, true, false);
            if (scope_rewritten != nullptr)
            {
                scope_expressions.emplace(scope_rewritten);
            }
            ConstASTPtr scope_complement_rewritten = rewrite(candidate, scope, false, false);
            if (scope_complement_rewritten != nullptr)
            {
                scope_complement_expressions.emplace(scope_complement_rewritten);
            }
            if (scope_rewritten == nullptr && scope_complement_rewritten == nullptr)
            {
                scope_straddling_expressions.emplace(candidate);
            }
        }

        // Compile the equality expressions on each side of the scope
        auto matching_canonical = getCanonical(scope_expressions);
        if (scope_expressions.size() >= 2)
        {
            for (const auto & scope_expression : scope_expressions)
            {
                if (scope_expression != matching_canonical)
                {
                    ASTPtr expression = makeASTFunction("equals", ASTs{matching_canonical->clone(), scope_expression->clone()});
                    scope_equalities.emplace_back(expression);
                }
            }
        }
        auto complement_canonical = getCanonical(scope_complement_expressions);
        if (scope_complement_expressions.size() >= 2)
        {
            for (const auto & scope_complement_expression : scope_complement_expressions)
            {
                if (scope_complement_expression != complement_canonical)
                {
                    ASTPtr expression
                        = makeASTFunction("equals", ASTs{complement_canonical->clone(), scope_complement_expression->clone()});
                    scope_complement_equalities.emplace_back(expression);
                }
            }
        }

        // Compile the scope straddling equality expressions
        std::vector<ConstASTPtr> connecting_expressions;
        connecting_expressions.emplace_back(matching_canonical);
        connecting_expressions.emplace_back(complement_canonical);
        for (const auto & scope_straddling_expression : scope_straddling_expressions)
        {
            connecting_expressions.emplace_back(scope_straddling_expression);
        }

        ConstASTSet connecting_expressions_remove_null;
        for (auto & connecting_expression : connecting_expressions)
        {
            if (connecting_expression != nullptr)
            {
                connecting_expressions_remove_null.emplace(connecting_expression);
            }
        }

        auto connecting_canonical = getCanonical(connecting_expressions_remove_null);
        if (connecting_canonical != nullptr)
        {
            for (const auto & connecting_expression_remove_null : connecting_expressions_remove_null)
            {
                if (connecting_expression_remove_null != connecting_canonical)
                {
                    ASTPtr expression
                        = makeASTFunction("equals", ASTs{connecting_canonical->clone(), connecting_expression_remove_null->clone()});
                    scope_straddling_equalities.emplace_back(expression);
                }
            }
        }
    }
    return EqualityPartition(std::move(scope_equalities), std::move(scope_complement_equalities), std::move(scope_straddling_equalities));
}

bool DisjointSet::findAndUnion(const ConstHashAST & element_1, const ConstHashAST & element_2)
{
    auto find_element_1 = find(element_1);
    auto find_element_2 = find(element_2);
    return union_(find_element_1, find_element_2);
}

ConstHashAST DisjointSet::find(ConstHashAST element)
{
    if (!map.contains(element))
    {
        map[element] = Entry{};
        return element;
    }
    return findInternal(element);
}

std::vector<ConstASTSet> DisjointSet::getEquivalentClasses()
{
    EqualityASTMap<ConstASTSet> root_to_tree_elements;
    for (auto & entry : map)
    {
        const ConstHashAST & node = entry.first;
        auto root = findInternal(node);
        if (root_to_tree_elements.contains(root))
        {
            ConstASTSet & value = root_to_tree_elements[root];
            value.emplace(node);
        }
        else
        {
            ConstASTSet value;
            value.emplace(node);
            root_to_tree_elements.emplace(root, value);
        }
    }
    std::vector<ConstASTSet> result;
    for (auto & element : root_to_tree_elements)
    {
        result.emplace_back(element.second);
    }
    return result;
}

bool DisjointSet::union_(ConstHashAST & element_1, ConstHashAST & element_2)
{
    if (PredicateUtils::equals(element_1, element_2))
    {
        return false;
    }
    Entry & entry1 = map[element_1];
    Entry & entry2 = map[element_2];
    int entry1_rank = entry1.getRank();
    int entry2_rank = entry2.getRank();

    if (entry1_rank < 0)
    {
        throw Exception("Rank < 0", ErrorCodes::LOGICAL_ERROR);
    }
    if (entry2_rank < 0)
    {
        throw Exception("Rank < 0", ErrorCodes::LOGICAL_ERROR);
    }

    if (entry1_rank < entry2_rank)
    {
        // make root1 child of root2
        entry1.setParent(element_2);
    }
    else
    {
        if (entry1_rank == entry2_rank)
        {
            // increment rank of root1 when both side were equally deep
            entry1.increaseRank();
        }
        // make root2 child of root1
        entry2.setParent(element_1);
    }
    return true;
}

ConstHashAST DisjointSet::findInternal(const ConstHashAST & element) // NOLINT(misc-no-recursion)
{
    Entry & entry = map[element];
    if (entry.getParent() == nullptr || entry.getParent() == element)
    {
        return element;
    }
    else
    {
        ConstHashAST parent = entry.getParent();
        ConstHashAST root = findInternal(parent);
        entry.setParent(root);
        return root;
    }
}

}
