#pragma once

#include <Parsers/IAST.h>

namespace DB
{
class ASTFunction;

class SimpleFunctionVisitor
{
public:

    static void visit(ASTPtr & ast);
    static void visit(ASTFunction * func);
};


}
