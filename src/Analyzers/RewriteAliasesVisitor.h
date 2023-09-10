#pragma once

#include <Interpreters/Context.h>
#include <Interpreters/InDepthNodeVisitor.h>
#include <Interpreters/QueryAliasesVisitor.h>
#include <Interpreters/QueryNormalizer.h>
#include <Parsers/ASTSelectQuery.h>

namespace DB
{

struct RewriteAliases
{
    using TypeToVisit = ASTSelectQuery;

    void visit(ASTSelectQuery &, ASTPtr & ast)
    {
        QueryAliasesVisitor::Data aliases;
        QueryAliasesVisitor(aliases).visit(ast);

        QueryNormalizer::Data normalizer_data(
            aliases,
            {},
            false,
            context->getSettingsRef(),
            true,
            context,
            nullptr,
            nullptr,
            /* rewrite_map_col */ false,
            /* aliases_rewrite_scope */ true);
        QueryNormalizer(normalizer_data).visit(ast);
    }

    RewriteAliases(ContextPtr context_) : context(std::move(context_))
    {
    }

    ContextPtr context;
};

using RewriteAliasesMatcher = OneTypeMatcher<RewriteAliases>;
using RewriteAliasesVisitor = InDepthNodeVisitor<RewriteAliasesMatcher, true>;

}
