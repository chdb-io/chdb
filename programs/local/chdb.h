#pragma once
#include <stddef.h>

extern "C" {
typedef void* LocalServerPtr;

struct local_result
{
    char * buf;
    size_t len;
    char * error_message;
};

struct init_result
{
   LocalServerPtr local_server;
   char * error_message;
};

init_result * chdb_connect(int argc, char ** argv);
void chdb_disconnect(LocalServerPtr obj);
local_result * chdb_query(LocalServerPtr obj, char * query, char * format);
void chdb_free_result(LocalServerPtr obj, local_result * result);
}
