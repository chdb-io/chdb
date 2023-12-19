#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chdb.h"

int main()
{
    // Define query parameters
    const char * output_format = "CSV";
    const char * queryStr = "SELECT 'Hello libchdb.so from chdbStub'"; // Replace with your query string
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
        printf("Rows Read: %lu\n", result->rows_read);
        printf("Bytes Read: %lu\n", result->bytes_read);

        // Free the result
        free_result(result);
    }

    return 0;
}
