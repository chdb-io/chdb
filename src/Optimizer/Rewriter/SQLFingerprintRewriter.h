#include <memory>
#include <Optimizer/SimpleExpressionRewriter.h>
#include <Parsers/ASTLiteral.h>
#include <Parsers/IAST_fwd.h>

namespace DB
{
class SQLFingerprintRewriter : public SimpleExpressionRewriter<Void>
{
public:
    ASTPtr visitASTLiteral(ASTPtr & node, Void & context) override
    {
        auto literal_ptr = std::dynamic_pointer_cast<ASTLiteral>(node);
        literal_ptr->updateField("?");
        return visitNode(node, context);
    }

    ASTPtr visitASTSelectQuery(ASTPtr & node, Void & context) override
    {
        auto select_ptr = std::dynamic_pointer_cast<ASTSelectQuery>(node);
        select_ptr->removeSettingsAndOutputFormat();
        return visitNode(node, context);
    }

};
}
