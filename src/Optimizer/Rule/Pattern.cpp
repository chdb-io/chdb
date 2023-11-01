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

#include <Optimizer/Rule/Pattern.h>

#include <Optimizer/Rule/Patterns.h>

namespace DB
{
std::optional<Match> Pattern::match(const PlanNodePtr & node, Captures & captures) const
{
    if (!previous)
        return accept(node, captures);

    if (auto prev_res = previous->match(node, captures))
        return accept(node, prev_res->captures);
    else
        return {};
}

PatternRawPtrs Pattern::getChildrenPatterns() const
{
    PatternRawPtrs children_patterns;
    PatternRawPtr pattern = this;
    while (pattern->getPrevious())
    {
        if (const auto * with_pattern = dynamic_cast<const WithPattern *>(pattern))
        {
            children_patterns = with_pattern->getSubPatterns();
        }
        pattern = pattern->getPrevious();
    }

    if (const auto * type_of_pattern = dynamic_cast<const TypeOfPattern *>(pattern))
    {
        if (type_of_pattern->type == IQueryPlanStep::Type::Tree)
            return {type_of_pattern};
    }
    return children_patterns;
}

// By convention, the head pattern is a TypeOfPattern,
// which indicates which plan node type this pattern is targeted to
IQueryPlanStep::Type Pattern::getTargetType() const
{
    PatternRawPtr pattern = this;
    while (pattern->getPrevious())
        pattern = pattern->getPrevious();

    if (const auto * type_of_pattern = dynamic_cast<const TypeOfPattern *>(pattern))
        return type_of_pattern->type;
    else
        throw Exception("Head pattern must be a TypeOfPattern, illegal pattern found: " + toString(), ErrorCodes::LOGICAL_ERROR);
}

String Pattern::toString() const
{
    PatternPrinter printer;
    this->accept(printer);
    return printer.formatted_str.str();
}

std::optional<Match> TypeOfPattern::accept(const PlanNodePtr & node, Captures & captures) const
{
    if (type == IQueryPlanStep::Type::Any || type == IQueryPlanStep::Type::Tree || type == node->getStep()->getType())
        return {Match{std::move(captures)}};
    else
        return {};
}

void TypeOfPattern::accept(PatternVisitor & pattern_visitor) const
{
    pattern_visitor.visitTypeOfPattern(*this);
}

std::optional<Match> CapturePattern::accept(const PlanNodePtr & node, Captures & captures) const
{
    captures.insert(std::make_pair(capture, property(node)));
    return {Match{std::move(captures)}};
}

void CapturePattern::accept(PatternVisitor & pattern_visitor) const
{
    pattern_visitor.visitCapturePattern(*this);
}

std::optional<Match> FilterPattern::accept(const PlanNodePtr & node, Captures & captures) const
{
    if (predicate(node, captures))
        return {Match{std::move(captures)}};
    else
        return {};
}

void FilterPattern::accept(PatternVisitor & pattern_visitor) const
{
    pattern_visitor.visitFilterPattern(*this);
}

std::optional<Match> WithPattern::accept(const PlanNodePtr & node, Captures & captures) const
{
    const PlanNodes & sub_nodes = node->getChildren();

    if (quantifier == PatternQuantifier::EMPTY)
    {
        if (sub_nodes.empty())
            return {Match{std::move(captures)}};
        else
            return {};
    }
    else if (quantifier == PatternQuantifier::SINGLE)
    {
        if (sub_nodes.size() == 1)
            return sub_patterns[0]->match(sub_nodes[0], captures);
        else
            return {};
    }
    else
    {
        bool matched = false;
        size_t index = 0;
        for (const PlanNodePtr & subNode : sub_nodes)
        {
            Captures sub_captures = captures;
            const auto * pattern = sub_patterns[0].get();
            if (sub_patterns.size() > index)
            {
                pattern = sub_patterns[index].get();
                index++;
            }
            std::optional<Match> subResult = pattern->match(subNode, sub_captures);

            if (subResult)
            {
                captures = std::move(subResult->captures);
                matched = true;
            }
            else if (quantifier == PatternQuantifier::ALL)
            {
                matched = false;
                break;
            }
        }

        if (matched)
            return {Match{std::move(captures)}};
        else
            return {};
    }
}

PatternRawPtrs WithPattern::getSubPatterns() const
{
    PatternRawPtrs res;
    for (const auto & sub: sub_patterns)
        res.emplace_back(sub.get());
    return res;
}

void WithPattern::accept(PatternVisitor & pattern_visitor) const
{
    pattern_visitor.visitWithPattern(*this);
}

void PatternPrinter::appendLine(const std::string & str)
{
    if (first)
        first = false;
    else
        formatted_str << '\n';

    formatted_str << std::string(level, '\t') << str;
}

void PatternPrinter::visitWithPattern(const WithPattern & pattern)
{
    visitPrevious(pattern);

    std::string withType = ([](const WithPattern & pattern_) -> std::string {
        switch (pattern_.getQuantifier())
        {
            case PatternQuantifier::EMPTY:
                return "empty";
            case PatternQuantifier::SINGLE:
                return "single";
            case PatternQuantifier::ANY:
                return "any";
            case PatternQuantifier::ALL:
                return "all";
        }

        throw Exception("Unknown with type", ErrorCodes::LOGICAL_ERROR);
    })(pattern);

    appendLine("with " + withType + ":");

    for (auto & sub_pattern : pattern.getSubPatterns())
    {
        level++;
        sub_pattern->accept(*this);
        level--;
    }
}

void PatternPrinter::visitTypeOfPattern(const TypeOfPattern & pattern)
{
    visitPrevious(pattern);
#define PRINT_STEP_TYPE(ITEM) \
    case IQueryPlanStep::Type::ITEM: \
        appendLine("typeOf: " #ITEM); \
        break;

    switch (pattern.type)
    {
        case IQueryPlanStep::Type::Any:
            appendLine("typeOf: Any");
            break;
            APPLY_STEP_TYPES(PRINT_STEP_TYPE)
        default:
            break;
    }
#undef PRINT_STEP_TYPE
}

void PatternPrinter::visitCapturePattern(const CapturePattern & pattern)
{
    visitPrevious(pattern);
    appendLine("capture " + pattern.name + " as: " + pattern.capture.desc);
}

void PatternPrinter::visitFilterPattern(const FilterPattern & pattern)
{
    visitPrevious(pattern);
    appendLine("filter by: " + pattern.name);
}

}
