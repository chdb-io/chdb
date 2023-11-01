#pragma once
#include <map>
#include <set>
#include <utility>

#include <Interpreters/Context.h>
#include <Optimizer/ConstHashAST.h>
#include <Optimizer/OptimizerMetrics.h>
#include <Optimizer/Utils.h>
#include <Parsers/IAST_fwd.h>

namespace DB
{
namespace ErrorCodes
{
    extern const int INCORRECT_DATA;
}
namespace EqAST
{
    struct Hash
    {
        UInt64 operator()(const ConstHashAST & key) const
        {
            return key.hash();
        }
    };

    struct Equal
    {
        bool operator()(const ConstHashAST & left, const ConstHashAST & right) const
        {
            if (left.getPtr() == right.getPtr())
                return true;

            if (left.hash() != right.hash())
                return false;

            return ASTEquality::ASTEquals()(left.getPtr(), right.getPtr());
        }
    };
}


template <typename Value>
class EqualityASTMap
{
public:
    EqualityASTMap() = default;

    EqualityASTMap(const EqualityASTMap &) = default;
    EqualityASTMap(EqualityASTMap &&) = default;

    using Key = ConstHashAST;
    using OriKey = ConstASTPtr;

    template <typename V>
    auto emplace(const Key & key, V && v)
    {
        // translator.translate(k);
        return container.emplace(key, std::forward<V>(v));
    }

    template <typename V>
    auto emplace(const OriKey & key, V && v)
    {
        return container.emplace(ConstHashAST::make(key), std::forward<V>(v));
    }

    Value & operator[](const Key & key)
    {
        return container[key];
    }

    Value & operator[](const OriKey & key)
    {
        return container[ConstHashAST::make(key)];
    }

    const Value & at(const Key & key) const
    {
        return container.at(key);
    }

    const Value & at(const OriKey & key) const
    {
        return container.at(ConstHashAST::make(key));
    }

    bool empty() const { return container.empty(); }

    template <typename K>
    bool count(const K & k) const
    {
        return contains(k);
    }

    bool contains(const Key & key) const
    {
        return container.contains(key);
    }

    bool contains(const OriKey & key) const
    {
        return container.contains(ConstHashAST::make(key));
    }

    auto begin() const { return container.begin(); }
    auto end() const { return container.end(); }

    auto begin() { return container.begin(); }
    auto end() { return container.end(); }
    size_t size() { return container.size(); }
    auto find(const Key & key)
    {
        return container.find(key);
    }
    auto find(const Key & key) const
    {
        return container.find(key);
    }

    auto find(const OriKey & key)
    {
        return container.find(ConstHashAST::make(key));
    }
    auto find(const OriKey & key) const
    {
        return container.find(ConstHashAST::make(key));
    }


private:
    std::unordered_map<Key, Value, EqAST::Hash, EqAST::Equal> container;
};

class EqualityASTSet
{
public:
    using Key = ConstHashAST;
    using OriKey = ConstASTPtr;
    auto emplace(const Key & key) { return container.emplace(key); }
    auto emplace(const OriKey & key) { return container.emplace(ConstHashAST::make(key)); }

    template <typename Iter>
    EqualityASTSet(Iter begin, Iter end) : EqualityASTSet()
    {
        for (auto iter = begin; iter != end; ++iter)
        {
            emplace(ConstHashAST::make(*iter));
        }
    }

    EqualityASTSet() = default;

    EqualityASTSet(EqualityASTSet &&) = default;
    EqualityASTSet(const EqualityASTSet &) = default;

    EqualityASTSet & operator=(const EqualityASTSet & right) = default;
    EqualityASTSet & operator=(EqualityASTSet && right) = default;

    bool empty() const { return container.empty(); }


    template <typename K>
    bool count(const K & k) const
    {
        return contains(k);
    }

    bool contains(const Key & key) const
    {
        auto res = container.contains(key);

        return res;
    }

    bool contains(const OriKey & key) const
    {
        auto res = container.contains(ConstHashAST::make(key));
        return res;
    }

    auto begin() const
    {
        return container.cbegin();
    }

    auto end() const { return container.cend(); }
    size_t size() { return container.size(); }

    template <typename Iter>
    auto erase(Iter iter) -> Iter
    {
        return container.erase(iter);
    }

private:
    std::unordered_set<Key, EqAST::Hash, EqAST::Equal> container;
};

}
