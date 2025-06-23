#include <unistd.h>
#include <Core/UUID.h>
#include <Common/thread_local_rng.h>
#include <Common/SipHash.h>


namespace DB
{

namespace UUIDHelpers
{
    UUID generateV4()
    {
        UUID uuid;
        getHighBytes(uuid) = (thread_local_rng() & 0xffffffffffff0fffull) | 0x0000000000004000ull;
        getLowBytes(uuid) = (thread_local_rng() & 0x3fffffffffffffffull) | 0x8000000000000000ull;

        return uuid;
    }

    UUID makeUUIDv4FromHash(const String & string)
    {
        UInt128 hash = sipHash128(string.data(), string.size());

        UUID uuid;
        getHighBytes(uuid) = (hash.items[HighBytes] & 0xffffffffffff0fffull) | 0x0000000000004000ull;
        getLowBytes(uuid) = (hash.items[LowBytes] & 0x3fffffffffffffffull) | 0x8000000000000000ull;

        return uuid;
    }

    /// chdb: generate UUID from process id. For testing purposes.
    UUID generate_from_pid()
    {
        UInt128 res{0, 0};
        res.items[1] = (res.items[1] & 0xffffffffffff0fffull) | (static_cast<UInt64>(getpid()));
        return UUID{res};
    }
}

}
