#pragma once

#include <Core/Types.h>


namespace DB
{

namespace UUIDHelpers
{
    /// Generate random UUID.
    UUID generateV4();

    /// Generate UUID from process id. For testing purposes.
    UUID generate_from_pid();

    const UUID Nil{};
}

}
