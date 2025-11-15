#pragma once

#include <mutex>

namespace DB
{
void registerInterpreters();

extern std::once_flag global_register_once_flag;
}
