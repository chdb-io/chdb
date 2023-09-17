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

#include <Processors/QueryPlan/PlanNode.h>

#include <memory>
#include <utility>

namespace DB
{

PlanNodePtr PlanNodeBase::getNodeById(PlanNodeId node_id) const
{
    PlanNodes stack;
    stack.push_back(std::const_pointer_cast<PlanNodeBase>(this->shared_from_this()));

    while (!stack.empty())
    {
        auto node = stack.back();
        stack.pop_back();
        if (node->getId() == node_id)
            return node;

        for (auto & child: node->getChildren())
            stack.push_back(child);
    }

    return nullptr;
}

#define PLAN_NODE_DEF(TYPE) \
template class PlanNode<TYPE##Step>;
APPLY_STEP_TYPES(PLAN_NODE_DEF)
PLAN_NODE_DEF(Any)
#undef PLAN_NODE_DEF

}
