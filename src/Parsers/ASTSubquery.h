#pragma once

#include <Parsers/ASTWithAlias.h>


namespace DB
{


/** SELECT subquery
  */
class ASTSubquery : public ASTWithAlias
{
public:
    // Stored the name when the subquery is defined in WITH clause. For example:
    // WITH (SELECT 1) AS a SELECT * FROM a AS b; cte_name will be `a`.
    std::string cte_name;

    // Stored the database name when the subquery is defined by a view. For example:
    // CREATE VIEW db1.v1 AS SELECT number FROM system.numbers LIMIT 10;
    // SELECT * FROM v1; database_of_view will be `db1`; cte_name will be `v1`.
    std::string database_of_view;
    /** Get the text that identifies this element. */
    String getID(char) const override { return "Subquery"; }

    ASTPtr clone() const override
    {
        auto clone = std::make_shared<ASTSubquery>(*this);
        clone->cloneChildren();
        return clone;
    }

    void updateTreeHashImpl(SipHash & hash_state) const override;
    bool isWithClause() const { return !cte_name.empty(); }
    String getAliasOrColumnName() const override;
    String tryGetAlias() const override;

protected:
    void formatImplWithoutAlias(const FormatSettings & settings, FormatState & state, FormatStateStacked frame) const override;
    void appendColumnNameImpl(WriteBuffer & ostr) const override;
};

}
