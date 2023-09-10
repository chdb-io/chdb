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

#include <Optimizer/Rule/Match.h>

#include <functional>
#include <memory>
#include <utility>

/**
 * Pattern matching is used to match a plan node with a specified
 * structure(`WithPattern`) or by some criteria(`FilterPattern`),
 * and retrieve properties of a plan node(`CapturePattern`).
 */
namespace DB
{
class Pattern;
using PatternPtr = std::unique_ptr<Pattern>;
using PatternPtrs = std::vector<PatternPtr>;
using PatternRawPtr = const Pattern *;
using PatternRawPtrs = std::vector<PatternRawPtr>;
class PatternVisitor;
using PatternProperty = std::function<std::any(const PlanNodePtr &)>;
using PatternPredicate = std::function<bool(const PlanNodePtr &, const Captures &)>;
enum class PatternQuantifier;

class Pattern
{
public:
    virtual ~Pattern() = default;

    IQueryPlanStep::Type getTargetType() const;
    String toString() const;

    bool matches(const PlanNodePtr & node) const { return match(node).has_value(); }

    std::optional<Match> match(const PlanNodePtr & node) const
    {
        Captures captures;
        return match(node, captures);
    }

    std::optional<Match> match(const PlanNodePtr & node, Captures & captures) const;
    virtual std::optional<Match> accept(const PlanNodePtr & node, Captures & captures) const = 0;
    virtual void accept(PatternVisitor & pattern_visitor) const = 0;
    PatternRawPtr getPrevious() const { return previous.get(); }

    PatternRawPtrs getChildrenPatterns() const;

protected:
    Pattern() = default;
    explicit Pattern(PatternPtr previous_) : previous(std::move(previous_)){}

private:
    PatternPtr previous;
};

class TypeOfPattern : public Pattern
{
public:
    explicit TypeOfPattern(IQueryPlanStep::Type type_) : Pattern(), type(type_){}
    TypeOfPattern(IQueryPlanStep::Type type_, PatternPtr previous) : Pattern(std::move(previous)), type(type_){}
    std::optional<Match> accept(const PlanNodePtr & node, Captures & captures) const override;
    void accept(PatternVisitor & pattern_visitor) const override;

    IQueryPlanStep::Type type;
};

class CapturePattern : public Pattern
{
public:
    CapturePattern(std::string name_, PatternProperty property_, Capture capture_, PatternPtr previous)
        : Pattern(std::move(previous)), name(std::move(name_)), property(std::move(property_)), capture(std::move(capture_)){}
    std::optional<Match> accept(const PlanNodePtr & node, Captures & captures) const override;
    void accept(PatternVisitor & pattern_visitor) const override;

    std::string name;
    PatternProperty property;
    Capture capture;
};

class FilterPattern : public Pattern
{
public:
    FilterPattern(std::string name_, PatternPredicate predicate_, PatternPtr previous)
        : Pattern(std::move(previous)), name(std::move(name_)), predicate(std::move(predicate_)){}
    std::optional<Match> accept(const PlanNodePtr & node, Captures & captures) const override;
    void accept(PatternVisitor & pattern_visitor) const override;

    std::string name;
    PatternPredicate predicate;
};

enum class PatternQuantifier
{
    EMPTY,
    SINGLE,
    ANY,
    ALL
};

class WithPattern : public Pattern
{
public:
    WithPattern(PatternQuantifier quantifier_, PatternPtr sub_pattern, PatternPtr previous)
        : Pattern(std::move(previous)), quantifier(quantifier_)
    {
        sub_patterns.emplace_back(std::move(sub_pattern));
    }

    WithPattern(PatternQuantifier quantifier_, PatternPtrs sub_patterns_, PatternPtr previous)
        : Pattern(std::move(previous)), quantifier(quantifier_), sub_patterns{std::move(sub_patterns_)} {}
    std::optional<Match> accept(const PlanNodePtr & node, Captures & captures) const override;
    void accept(PatternVisitor & pattern_visitor) const override;

    PatternQuantifier getQuantifier() const { return quantifier; }
    PatternRawPtrs getSubPatterns() const;

private:
    PatternQuantifier quantifier;
    PatternPtrs sub_patterns;
};

class PatternVisitor
{
public:
    virtual ~PatternVisitor() = default;
    virtual void visitTypeOfPattern(const TypeOfPattern & pattern) = 0;
    virtual void visitCapturePattern(const CapturePattern & pattern) = 0;
    virtual void visitFilterPattern(const FilterPattern & pattern) = 0;
    virtual void visitWithPattern(const WithPattern & pattern) = 0;

    void visitPrevious(const Pattern & pattern)
    {
        if (const auto * prev = pattern.getPrevious())
            prev->accept(*this);
    }
};

class PatternPrinter : public PatternVisitor
{
public:
    void visitTypeOfPattern(const TypeOfPattern & pattern) override;
    void visitCapturePattern(const CapturePattern & pattern) override;
    void visitFilterPattern(const FilterPattern & pattern) override;
    void visitWithPattern(const WithPattern & pattern) override;

    void appendLine(const std::string & str);

    std::stringstream formatted_str;
    int level = 0;
    bool first = true;
};

}
