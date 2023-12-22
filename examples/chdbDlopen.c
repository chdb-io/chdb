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
    const char * output_format = "CSV";
    const char * queryStr = "SELECT 'Hello libchdb.so from chdbDlopen'"; // Replace with your query string
    const char * path = ""; // Replace with the path if needed
    const char * udfPath = ""; // Replace with the UDF path if needed

    // Prepare the argument array
    char * argv[10]; // Adjust array size as needed
    int argc = 0;
    argv[argc++] = "clickhouse";
    argv[argc++] = "--multiquery";

    // Add parameters based on the output format
    if (strcmp(output_format, "Debug") == 0 || strcmp(output_format, "debug") == 0)
    {
        argv[argc++] = "--verbose";
        argv[argc++] = "--log-level=trace";
        argv[argc++] = "--output-format=CSV";
    }
    else
    {
        char output_format_arg[50];
        sprintf(output_format_arg, "--output-format=%s", output_format);
        argv[argc++] = strdup(output_format_arg); // Note: Memory allocated by strdup needs to be freed
    }

    // If path is not empty, add corresponding parameter
    if (path && *path)
    {
        char path_arg[100];
        sprintf(path_arg, "--path=%s", path);
        argv[argc++] = path_arg;
    }

    // Add query string
    char query_arg[1000]; // Adjust size to fit actual query string length
    sprintf(query_arg, "--query=%s", queryStr);
    argv[argc++] = query_arg;

    // If udfPath is not empty, add corresponding parameters
    if (udfPath && *udfPath)
    {
        argv[argc++] = "--";
        char udfPath_arg1[200]; // Adjust size to fit actual query string length
        sprintf(udfPath_arg1, "--user_scripts_path=%s", udfPath);
        argv[argc++] = udfPath_arg1; // Memory allocated by strdup needs to be freed

        char udfPath_arg2[200]; // Adjust size to fit actual query string length
        sprintf(udfPath_arg2, "--user_defined_executable_functions_config=%s/*.xml", udfPath);
        argv[argc++] = udfPath_arg2; // Memory allocated by strdup needs to be freed
    }

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
