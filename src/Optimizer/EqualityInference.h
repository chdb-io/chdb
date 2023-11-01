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

#pragma once

#include <map>
#include <set>
#include <utility>

#include <Interpreters/Context.h>
#include <Optimizer/ExpressionExtractor.h>
#include <Optimizer/PredicateUtils.h>
#include <Optimizer/SymbolsExtractor.h>
#include <Optimizer/EqualityASTMap.h>
#include <Optimizer/Utils.h>
#include <Parsers/IAST_fwd.h>
#include "ConstHashAST.h"

namespace DB
{
struct DisjointSet;
struct EqualityPartition;
using ConstASTMap = EqualityASTMap<ConstHashAST>;
using ConstASTSet = EqualityASTSet;
/**
 * Makes equality based inferences to rewrite Expressions and
 * generate equality sets in terms of specified symbol scopes.
 */
class EqualityInference
{
public:
    static EqualityInference newInstance(const ConstASTPtr & predicates, ContextMutablePtr & context);
    static EqualityInference newInstance(const std::vector<ConstASTPtr> & predicates, ContextMutablePtr & context);

    /**
     * Determines whether an predicate may be successfully applied to the equality inference
     *
     * for example:
     *
     * p_partkey = ps_partkey is a reasonable candidate.
     * r_name = r_name is not a candidate.
     */
    static bool isInferenceCandidate(const ConstASTPtr & predicate, ContextMutablePtr & context);
    static bool mayReturnNullOnNonNullInput(const ASTFunction & predicate);
    static EqualityASTMap<ConstASTSet>
    makeEqualitySets(DisjointSet equalities);
    static ConstHashAST getMin(ConstASTSet & equivalence);

    /**
     * Provides a convenience Iterable of Expression conjuncts which have not been added to the inference
     */
    static std::vector<ConstASTPtr> nonInferrableConjuncts(const ConstASTPtr & expression, ContextMutablePtr & context);

    /**
     * Attempts to rewrite an Expression in terms of the symbols allowed by the symbol scope
     * given the known equalities. Returns null if unsuccessful.
     */
    ASTPtr rewrite(const ConstASTPtr & expression, const std::set<String> & scope);
    ASTPtr rewrite(const ConstASTPtr & expression, const std::set<String> & scope, bool contains, bool allow_full_replacement);
    ConstHashAST getScopedCanonical(const ConstHashAST & expression, const std::set<String> & scope, bool contains);
    static ConstHashAST getCanonical(ConstASTSet & equivalence);
    static bool isScoped(const ConstASTPtr & expression, const std::set<String> & scope);
    static bool isNotScoped(const ConstASTPtr & expression, const std::set<String> & scope);
    EqualityPartition partitionedBy(const std::set<String>& scope);

private:
    EqualityInference(
        EqualityASTMap<ConstASTSet> equality_sets_,
        ConstASTMap canonical_map_,
        ConstASTSet derived_expressions_)
        : equality_sets(std::move(equality_sets_))
        , canonical_map(std::move(canonical_map_))
        , derived_expressions(std::move(derived_expressions_))
    {
    }
    // Indexed by canonical expression
    EqualityASTMap<ConstASTSet> equality_sets;
    // Map each known expression to canonical expression
    ConstASTMap canonical_map;
    ConstASTSet derived_expressions;
};


struct DisjointSet
{
public:
    bool findAndUnion(const ConstHashAST & element_1, const ConstHashAST & element_2);
    ConstHashAST find(ConstHashAST element);
    std::vector<ConstASTSet> getEquivalentClasses();

private:
    class Entry
    {
    public:
        explicit Entry(ConstHashAST predicate = {}, int rank_ = 0) : parent(std::move(predicate)), rank(rank_) { }
        ConstHashAST getParent() { return parent; }
        void setParent(ConstHashAST predicate)
        {
            parent = std::move(predicate);
            rank = -1;
        }
        int getRank() const { return rank; }
        void increaseRank()
        {
            if (parent == nullptr)
            {
                rank++;
            }
        }

    private:
        ConstHashAST parent;
        int rank;
    };

    EqualityASTMap<Entry> map;

    bool union_(ConstHashAST & element_1, ConstHashAST & element_2);
    ConstHashAST findInternal(const ConstHashAST & element);
};

/**
 * Dumps the inference equalities as equality expressions that are partitioned
 * by the symbol scope. All stored equalities are returned in a compact set and
 * will be classified into three groups as determined by the symbol scope:
 *
 * <ol>
 * <li>equalities that fit entirely within the symbol scope</li>
 * <li>equalities that fit entirely outside of the symbol scope</li>
 * <li>equalities that straddle the symbol scope</li>
 * </ol>
 *
 * <pre>
 * Example:
 *   Stored Equalities:
 *     a = b = c
 *     d = e = f = g
 *
 *   Symbol Scope:
 *     a, b, d, e
 *
 *   Output EqualityPartition:
 *     Scope Equalities:
 *       a = b
 *       d = e
 *     Complement Scope Equalities
 *       f = g
 *     Scope Straddling Equalities
 *       a = c
 *       d = f
 * </pre>
 */
struct EqualityPartition
{
public:
    EqualityPartition(
        std::vector<ConstASTPtr> scope_equalities_,
        std::vector<ConstASTPtr> scope_complement_equalities_,
        std::vector<ConstASTPtr> scope_straddling_equalities_)
        : scope_equalities(std::move(scope_equalities_))
        , scope_complement_equalities(std::move(scope_complement_equalities_))
        , scope_straddling_equalities(std::move(scope_straddling_equalities_))
    {
    }
    const std::vector<ConstASTPtr> & getScopeEqualities() const { return scope_equalities; }
    const std::vector<ConstASTPtr> & getScopeComplementEqualities() const { return scope_complement_equalities; }
    const std::vector<ConstASTPtr> & getScopeStraddlingEqualities() const { return scope_straddling_equalities; }

private:
    std::vector<ConstASTPtr> scope_equalities;
    std::vector<ConstASTPtr> scope_complement_equalities;
    std::vector<ConstASTPtr> scope_straddling_equalities;
};

}
