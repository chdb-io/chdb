#pragma once
#include <stddef.h>

extern "C" {
struct local_result
{
    char * buf;
    size_t len;
};

local_result * query_stable(int argc, char ** argv);
void free_result(local_result * result);
}
