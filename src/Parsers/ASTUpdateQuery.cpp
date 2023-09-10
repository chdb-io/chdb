#include <Parsers/ASTUpdateQuery.h>
#include <Common/quoteString.h>

namespace DB
{

String ASTUpdateQuery::getID(char) const
{
    return "UpdateQuery";
}

ASTPtr ASTUpdateQuery::clone() const
{
    auto res = std::make_shared<ASTUpdateQuery>(*this);
    res->children.clear();

    res->assignment_list = assignment_list->clone();
    res->where_condition = where_condition->clone();

    if (order_by_expr)
        res->order_by_expr = order_by_expr->clone();
    if (limit_value)
        res->limit_value = limit_value->clone();
    if (limit_offset)
        res->limit_offset = limit_offset->clone();

    return res;
}

void ASTUpdateQuery::formatQueryImpl(const FormatSettings & settings, FormatState & state, FormatStateStacked frame) const
{
     settings.ostr << (settings.hilite ? hilite_keyword : "") << "UPDATE " << (settings.hilite ? hilite_none : "");

    if (!database.empty())
        settings.ostr << backQuoteIfNeed(database) << ".";
    
    settings.ostr << backQuoteIfNeed(table);

    settings.ostr << (settings.hilite ? hilite_keyword : "") << " SET " << (settings.hilite ? hilite_none : "");

    assignment_list->formatImpl(settings, state, frame);

    settings.ostr << (settings.hilite ? hilite_keyword : "") << " WHERE " << (settings.hilite ? hilite_none : "");
    where_condition->formatImpl(settings, state, frame);

    if (order_by_expr)
    {
        settings.ostr << (settings.hilite ? hilite_keyword : "") << " ORDER BY " << (settings.hilite ? hilite_none : "");
        order_by_expr->formatImpl(settings, state, frame);
    }

    if (limit_value)
    {
        settings.ostr << (settings.hilite ? hilite_keyword : "") << " LIMIT " << (settings.hilite ? hilite_none : "");

        if (limit_offset)
        {
            limit_offset->formatImpl(settings, state, frame);
            settings.ostr << ", ";
        }
        limit_value->formatImpl(settings, state, frame);
    }
}

}

