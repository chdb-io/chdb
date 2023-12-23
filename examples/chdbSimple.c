#include <dlfcn.h> // Include for dynamic loading
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Assuming chdb.h defines the structure and functions we need
#include "chdb.h"

int main()
{
    void * handle;
    struct local_result * (*query_stable)(int, char **);
    void (*free_result)(struct local_result *);

    // Load the libchdb.so from the parent directory
    handle = dlopen("../libchdb.so", RTLD_LAZY);
    if (!handle)
    {
        fprintf(stderr, "Error loading libchdb.so: %s\n", dlerror());
        exit(1);
    }

    // Get the query_stable function
    *(void **)(&query_stable) = dlsym(handle, "query_stable");
    if (!query_stable)
    {
        fprintf(stderr, "Could not find query_stable: %s\n", dlerror());
        dlclose(handle);
        exit(1);
    }

    // Get the free_result function
    *(void **)(&free_result) = dlsym(handle, "free_result");
    if (!free_result)
    {
        fprintf(stderr, "Could not find free_result: %s\n", dlerror());
        dlclose(handle);
        exit(1);
    }

    // Define query parameters
    const char * queryStr = "SELECT 'Hello libchdb.so from chdbSimple'"; // Replace with your query string

    // Prepare the argument array
    char * argv[10]; // Adjust array size as needed
    int argc = 0;
    argv[argc++] = "clickhouse";
    argv[argc++] = "--multiquery";
    argv[argc++] = "--output-format=CSV";

    // Add query string
    char query_arg[1000]; // Adjust size to fit actual query string length
    sprintf(query_arg, "--query=%s", queryStr);
    argv[argc++] = query_arg;

    // Call the query_stable function
    struct local_result * result = query_stable(argc, argv);

    // Print results (assuming it's a string type, adjust according to actual data type)
    if (result)
    {
        printf("Query Result: %s\n", result->buf);
        printf("Elapsed Time: %fs\n", result->elapsed);
        printf("Rows Read: %llu\n", (unsigned long long)result->rows_read);
        printf("Bytes Read: %llu\n", (unsigned long long)result->bytes_read);

        // Free the result
        free_result(result);
    }

    // Remember to close the library handle at the end
    dlclose(handle);
    return 0;
}
