#include <stdlib.h>
#include <stdio.h>
// Stub for __real_free() function to make clickhouse-local linking possible
// this function will print error message and terminate program if it will be called
void __real_free(void *) __attribute__((noreturn));

void __real_free(void * ptr)
{
    fprintf(stderr, "ERROR: __real_free() function called with %p\n", ptr);
    exit(1);
}
