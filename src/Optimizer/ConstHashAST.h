#pragma once

#include <Analyzers/ASTEquals.h>
#include <Parsers/IAST.h>
#include <Parsers/IAST_fwd.h>


namespace DB
{
// Hash-Augmented Const AST
// ensure it is immutable
class ConstHashAST
{
public:
    ConstHashAST(UInt64 hashcode_, ConstASTPtr ptr_) : hashcode(hashcode_), ptr(std::move(ptr_)) { }
    ConstHashAST() : hashcode(0), ptr(nullptr) { }
    ConstHashAST(nullptr_t) : hashcode(0), ptr(nullptr) { }

    static ConstHashAST make(ConstASTPtr ptr_)
    {
        auto hashcode = ASTEquality::ASTHash()(ptr_);
        return ConstHashAST(hashcode, std::move(ptr_));
    }

    const ConstASTPtr & getPtr() const { return ptr; }

    // make it use like ConstAST
    const IAST * get() const { return ptr.get(); }
    const IAST & operator*() const { return *ptr; }
    const IAST * operator->() const { return ptr.get(); }
    operator ConstASTPtr() const { return ptr; }

    bool operator==(const ConstHashAST & x) const { return ptr == x.ptr; }
    bool operator!=(const ConstHashAST & x) const { return ptr != x.ptr; }

    UInt64 hash() const { return hashcode; }

private:
    UInt64 hashcode = 0;
    ConstASTPtr ptr;
};


}
