#pragma once

#include <Parsers/ASTQueryWithTableAndOutput.h>

namespace DB
{

/*
* Unique table update statement.
* UPDATE table_name SET assignment_list WHERE where_condition [ORDER BY ...]  [LIMIT ...]  
*/

class ASTUpdateQuery : public ASTQueryWithTableAndOutput
{
public:

    String getID(char) const override;
    ASTPtr clone() const override;

    ASTPtr assignment_list;
    ASTPtr where_condition;

    // optional expressions
    ASTPtr order_by_expr;
    ASTPtr limit_value;
    ASTPtr limit_offset;

protected:
    void formatQueryImpl(const FormatSettings & settings, FormatState & state, FormatStateStacked frame) const override;
};

}
