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

#include <Processors/QueryPlan/PlanVisitor.h>
#include <Processors/QueryPlan/QueryPlan.h>
#include <Processors/QueryPlan/SimplePlanVisitor.h>
#include "QueryPlan/PlanNode.h"

namespace DB
{
class PlanPattern
{
public:
    static bool isSimpleQuery(QueryPlan & plan);
    static bool hasCrossJoin(QueryPlan & plan);
    static bool hasOuterJoin(QueryPlan & plan);
    static size_t maxJoinSize(QueryPlan & plan, ContextMutablePtr & context);
};

class SimpleQueryPlanPatternVisitor : public SimplePlanVisitor<Void>
{
public:
    explicit SimpleQueryPlanPatternVisitor(CTEInfo & cte_info) : SimplePlanVisitor(cte_info) { }

    bool isSimpleQuery() const { return simple_query; }

    Void visitJoinNode(JoinNode &, Void &) override;
    Void visitApplyNode(ApplyNode &, Void &) override;
    Void visitIntersectNode(IntersectNode &, Void &) override;
    Void visitExceptNode(ExceptNode &, Void &) override;
    Void visitCTERefNode(CTERefNode &, Void &) override;

private:
    bool simple_query = true;
};

class CrossJoinPlanPatternVisitor : public SimplePlanVisitor<Void>
{
public:
    explicit CrossJoinPlanPatternVisitor(CTEInfo & cte_info) : SimplePlanVisitor(cte_info) { }

    bool hasCrossJoin() const { return has_cross_join; }

    Void visitJoinNode(JoinNode &, Void &) override;

private:
    bool has_cross_join = false;
};

class OuterJoinPlanPatternVisitor : public SimplePlanVisitor<Void>
{
public:
    explicit OuterJoinPlanPatternVisitor(CTEInfo & cte_info) : SimplePlanVisitor(cte_info) { }

    bool hasOuterJoin() const { return has_outer_join; }

    Void visitJoinNode(JoinNode &, Void &) override;

private:
    bool has_outer_join = false;
};

class GetMaxJoinSizeVisitor : public PlanNodeVisitor<size_t, Void>
{
public:
    explicit GetMaxJoinSizeVisitor(ContextMutablePtr context_, CTEInfo & cte_info) : context(context_), cte_helper(cte_info) { }
    size_t visitJoinNode(JoinNode &, Void &) override;
    size_t visitPlanNode(PlanNodeBase & node, Void & c) override
    {
        for (const auto & child : node.getChildren())
            VisitorUtil::accept(*child, *this, c);
        return 1;
    }
    size_t visitCTERefNode(CTERefNode & node, Void & c) override
    {
        const auto * cte_step = dynamic_cast<const CTERefStep *>(node.getStep().get());
        auto cte_id = cte_step->getId();
        cte_helper.accept(cte_id, *this, c);
        return 1;
    }

    size_t getMaxSize() const { return max_size; }

private:
    ContextMutablePtr context;
    CTEPreorderVisitHelper cte_helper;
    size_t max_size = 0;
};

}
