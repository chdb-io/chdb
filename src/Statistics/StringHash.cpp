#include <Statistics/StringHash.h>

#include <city.h>

namespace DB::Statistics
{

// this hash function impls the behaviour of cityHash64(toString(col)) in sql
// so FixedString and String are hashed the same way
UInt64 stringHash64(std::string_view view)
{
    // trim tail \0
    UInt64 tail = view.size();
    for (; tail > 0; --tail)
    {
        if (view[tail - 1] != '\0')
            break;
    }
    view = view.substr(0, tail);
    return CityHash_v1_0_2::CityHash64(view.data(), view.size());
}

}