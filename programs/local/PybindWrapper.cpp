#include "PybindWrapper.h"

#include <pybind11/detail/non_limited_api.h>

#include <Common/Exception.h>

using namespace DB;

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

}

namespace pybind11 {

bool gil_check()
{
    return static_cast<bool>(pybind11::non_limited_api::PyGILState_Check());
}

void gil_assert()
{
    if (!gil_check())
        throw Exception(ErrorCodes::LOGICAL_ERROR,
                        "The GIL should be held for this operation, but it's not!");
}

} // namespace pybind11
