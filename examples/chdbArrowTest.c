#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <errno.h>

#include "../programs/local/chdb.h"
#include "arrow_c_abi.h"

// Custom ArrowArrayStream implementation data
typedef struct CustomStreamData
{
    bool schema_sent;
    size_t current_row;
    size_t total_rows;
    size_t batch_size;
    char* last_error;
} CustomStreamData;

// Function to initialize CustomStreamData
static void init_custom_stream_data(CustomStreamData * data)
{
    data->schema_sent = false;
    data->current_row = 0;
    data->total_rows = 1000000;
    data->batch_size = 10000;
    data->last_error = NULL;
}

// Reset the stream to allow reading from the beginning
static void reset_custom_stream_data(CustomStreamData * data)
{
    data->current_row = 0;
    if (data->last_error) {
        free(data->last_error);
        data->last_error = NULL;
    }
}

// Release function prototypes
static void release_schema_child(struct ArrowSchema * s);
static void release_schema_main(struct ArrowSchema * s);
static void release_id_array(struct ArrowArray * arr);
static void release_string_array(struct ArrowArray * arr);
static void release_main_array(struct ArrowArray * arr);

// Helper function to find minimum of two values
static size_t min_size_t(size_t a, size_t b)
{
    return (a < b) ? a : b;
}

// Release function implementations
static void release_schema_child(struct ArrowSchema * s)
{
    s->release = NULL;
}

static void release_schema_main(struct ArrowSchema * s)
{
    if (s->children)
    {
        for (int64_t i = 0; i < s->n_children; i++)
        {
            if (s->children[i] && s->children[i]->release)
            {
                s->children[i]->release(s->children[i]);
            }
            free(s->children[i]);
        }
        free(s->children);
    }
    s->release = NULL;
}

static void release_id_array(struct ArrowArray * arr)
{
    if (arr->buffers)
    {
        free((void*)(uintptr_t)arr->buffers[1]);  // free data buffer
        free((void**)(uintptr_t)arr->buffers);
    }
    arr->release = NULL;
}

static void release_string_array(struct ArrowArray * arr)
{
    if (arr->buffers)
    {
        free((void*)(uintptr_t)arr->buffers[1]);  // free offset buffer
        free((void*)(uintptr_t)arr->buffers[2]);  // free data buffer
        free((void**)(uintptr_t)arr->buffers);
    }
    arr->release = NULL;
}

static void release_main_array(struct ArrowArray * arr)
{
    if (arr->children)
    {
        for (int64_t i = 0; i < arr->n_children; i++)
        {
            if (arr->children[i] && arr->children[i]->release)
            {
                arr->children[i]->release(arr->children[i]);
            }
            free(arr->children[i]);
        }
        free(arr->children);
    }
    if (arr->buffers) {
        free((void**)(uintptr_t)arr->buffers);
    }
    arr->release = NULL;
}

// Helper function to create schema with 2 columns: id(int64), value(string)
static void create_schema(struct ArrowSchema * schema)
{
    schema->format = "+s";  // struct format
    schema->name = NULL;
    schema->metadata = NULL;
    schema->flags = 0;
    schema->n_children = 2;
    schema->children = (struct ArrowSchema**)malloc(2 * sizeof(struct ArrowSchema*));
    schema->dictionary = NULL;
    schema->release = release_schema_main;

    // Field 0: id (int64)
    schema->children[0] = (struct ArrowSchema*)malloc(sizeof(struct ArrowSchema));
    schema->children[0]->format = "l";  // int64
    schema->children[0]->name = "id";
    schema->children[0]->metadata = NULL;
    schema->children[0]->flags = 0;
    schema->children[0]->n_children = 0;
    schema->children[0]->children = NULL;
    schema->children[0]->dictionary = NULL;
    schema->children[0]->release = release_schema_child;

    // Field 1: value (string)
    schema->children[1] = (struct ArrowSchema*)malloc(sizeof(struct ArrowSchema));
    schema->children[1]->format = "u";  // utf8 string
    schema->children[1]->name = "value";
    schema->children[1]->metadata = NULL;
    schema->children[1]->flags = 0;
    schema->children[1]->n_children = 0;
    schema->children[1]->children = NULL;
    schema->children[1]->dictionary = NULL;
    schema->children[1]->release = release_schema_child;
}

// Helper function to create a batch of data
static void create_batch(struct ArrowArray * array, size_t start_row, size_t batch_size)
{
    struct ArrowArray * id_array;
    struct ArrowArray * str_array;
    int64_t * id_data;
    int32_t * offsets;
    size_t total_str_len;
    char ** strings;
    char * str_data;
    size_t pos;
    size_t i;

    // Main array structure
    array->length = batch_size;
    array->null_count = 0;
    array->offset = 0;
    array->n_buffers = 1;
    array->n_children = 2;
    array->buffers = (const void **)malloc(1 * sizeof(void *));
    array->buffers[0] = NULL;  // validity buffer (no nulls)
    array->children = (struct ArrowArray **)malloc(2 * sizeof(struct ArrowArray *));
    array->dictionary = NULL;

    // Create id column (int64)
    array->children[0] = (struct ArrowArray *)malloc(sizeof(struct ArrowArray));
    id_array = array->children[0];
    id_array->length = batch_size;
    id_array->null_count = 0;
    id_array->offset = 0;
    id_array->n_buffers = 2;
    id_array->n_children = 0;
    id_array->buffers = (const void **)malloc(2 * sizeof(void *));
    id_array->buffers[0] = NULL;  // validity buffer

    // Allocate and fill id data
    id_data = (int64_t *)malloc(batch_size * sizeof(int64_t));
    for (i = 0; i < batch_size; i++)
        id_data[i] = start_row + i;

    id_array->buffers[1] = id_data;  // data buffer
    id_array->children = NULL;
    id_array->dictionary = NULL;
    id_array->release = release_id_array;

    // Create value column (string)
    array->children[1] = (struct ArrowArray *)malloc(sizeof(struct ArrowArray));
    str_array = array->children[1];
    str_array->length = batch_size;
    str_array->null_count = 0;
    str_array->offset = 0;
    str_array->n_buffers = 3;
    str_array->n_children = 0;
    str_array->buffers = (const void **)malloc(3 * sizeof(void *));
    str_array->buffers[0] = NULL;  // validity buffer

    // Create offset buffer (int32)
    offsets = (int32_t *)malloc((batch_size + 1) * sizeof(int32_t));
    offsets[0] = 0;

    // Calculate total string length and create strings
    total_str_len = 0;
    strings = (char **)malloc(batch_size * sizeof(char *));
    for (i = 0; i < batch_size; i++)
    {
        char buffer[64];
        size_t len;
        snprintf(buffer, sizeof(buffer), "value_%zu", start_row + i);
        len = strlen(buffer);
        strings[i] = (char*)malloc(len + 1);
        strcpy(strings[i], buffer);
        total_str_len += len;
        offsets[i + 1] = total_str_len;
    }
    str_array->buffers[1] = offsets;  // offset buffer

    // Create data buffer
    str_data = (char*)malloc(total_str_len);
    pos = 0;
    for (i = 0; i < batch_size; i++)
    {
        size_t len = strlen(strings[i]);
        memcpy(str_data + pos, strings[i], len);
        pos += len;
        free(strings[i]);
    }
    free(strings);
    str_array->buffers[2] = str_data;  // data buffer

    str_array->children = NULL;
    str_array->dictionary = NULL;
    str_array->release = release_string_array;

    // Main array release function
    array->release = release_main_array;
}

// Callback function to get schema
static int custom_get_schema(struct ArrowArrayStream* stream, struct ArrowSchema* out)
{
    (void)stream;  // Suppress unused parameter warning
    create_schema(out);
    return 0;
}

// Callback function to get next array
static int custom_get_next(struct ArrowArrayStream * stream, struct ArrowArray * out)
{
    CustomStreamData * data;
    size_t remaining_rows;
    size_t batch_size;

    data = (CustomStreamData *)stream->private_data;
    if (!data)
        return EINVAL;

    // Check if we've reached the end of the stream
    if (data->current_row >= data->total_rows)
    {
        // End of stream - set release to NULL to indicate no more data
        out->release = NULL;
        return 0;
    }

    // Calculate batch size for this iteration
    remaining_rows = data->total_rows - data->current_row;
    batch_size = min_size_t(data->batch_size, remaining_rows);

    // Create the batch
    create_batch(out, data->current_row, batch_size);

    data->current_row += batch_size;
    return 0;
}

// Callback function to get last error
static const char * custom_get_last_error(struct ArrowArrayStream * stream)
{
    CustomStreamData * data = (CustomStreamData *)stream->private_data;
    if (!data || !data->last_error)
        return NULL;

    return data->last_error;
}

// Callback function to release stream resources
static void custom_release(struct ArrowArrayStream * stream)
{
    if (stream->private_data)
    {
        CustomStreamData * data = (CustomStreamData *)stream->private_data;
        if (data->last_error)
        {
            free(data->last_error);
        }
        free(data);
        stream->private_data = NULL;
    }
    stream->release = NULL;
}

// Helper function to reset the ArrowArrayStream for reuse
static void reset_arrow_stream(struct ArrowArrayStream * stream)
{
    if (stream && stream->private_data)
    {
        CustomStreamData * data = (CustomStreamData *)stream->private_data;
        reset_custom_stream_data(data);
        printf("✓ ArrowArrayStream has been reset, ready for re-reading\n");
    }
}

//===--------------------------------------------------------------------===//
// Unit Test Utilities
//===--------------------------------------------------------------------===//

static void test_assert(bool condition, const char * test_name, const char * message)
{
    if (condition)
    {
        printf("✓ PASS: %s\n", test_name);
    }
    else
    {
        printf("✗ FAIL: %s", test_name);
        if (message && strlen(message) > 0)
        {
            printf(" - %s", message);
        }
        printf("\n");
        exit(1);
    }
}

static void test_assert_chdb_state(chdb_state state, const char * operation_name)
{
    char message[256];
    if (state == CHDBError)
    {
        strcpy(message, "Operation failed");
    }
    else
    {
        strcpy(message, "Unknown state");
    }

    test_assert(state == CHDBSuccess, operation_name, 
                state == CHDBError ? message : NULL);
}

static void test_assert_not_null(void * ptr, const char * test_name)
{
    test_assert(ptr != NULL, test_name, "Pointer is null");
}

static void test_assert_no_error(chdb_result * result, const char * query_name)
{
    char full_test_name[512];
    const char * error;

    snprintf(full_test_name, sizeof(full_test_name), "%s - Result is not null", query_name);
    test_assert_not_null(result, full_test_name);

    error = chdb_result_error(result);
    snprintf(full_test_name, sizeof(full_test_name), "%s - No query error", query_name);

    if (error)
    {
        char error_message[512];
        snprintf(error_message, sizeof(error_message), "Error: %s", error);
        test_assert(error == NULL, full_test_name, error_message);
    }
    else
    {
        test_assert(error == NULL, full_test_name, NULL);
    }
}

static void test_assert_query_result_contains(chdb_result * result, const char * expected_content, const char * query_name)
{
    char * buffer;
    char full_test_name[512];
    bool contains;

    test_assert_no_error(result, query_name);

    buffer = chdb_result_buffer(result);
    snprintf(full_test_name, sizeof(full_test_name), "%s - Result buffer is not null", query_name);
    test_assert_not_null(buffer, full_test_name);

    snprintf(full_test_name, sizeof(full_test_name), "%s - Result contains expected content", query_name);

    contains = strstr(buffer, expected_content) != NULL;
    if (!contains)
    {
        char error_message[1024];
        snprintf(error_message, sizeof(error_message), "Expected: %s, Actual: %s", expected_content, buffer);
        test_assert(contains, full_test_name, error_message);
    }
    else
    {
        test_assert(contains, full_test_name, NULL);
    }
}

static void test_assert_row_count(chdb_result * result, uint64_t expected_rows, const char * query_name)
{
    char * buffer;
    char full_test_name[512];
    char * result_str;
    char * end;
    uint64_t actual_rows;

    test_assert_no_error(result, query_name);

    buffer = chdb_result_buffer(result);
    snprintf(full_test_name, sizeof(full_test_name), "%s - Result buffer is not null", query_name);
    test_assert_not_null(buffer, full_test_name);

    /* Parse the count result (assuming CSV format with just the number) */
    result_str = (char*)malloc(strlen(buffer) + 1);
    strcpy(result_str, buffer);

    /* Remove trailing whitespace/newlines */
    end = result_str + strlen(result_str) - 1;
    while (end > result_str && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r' || *end == '\f' || *end == '\v')) {
        *end = '\0';
        end--;
    }

    actual_rows = strtoull(result_str, NULL, 10);

    snprintf(full_test_name, sizeof(full_test_name), "%s - Row count matches", query_name);

    if (actual_rows != expected_rows)
    {
        char error_message[256];
        snprintf(error_message, sizeof(error_message), "Expected: %llu, Actual: %llu",
                 (unsigned long long)expected_rows, (unsigned long long)actual_rows);
        test_assert(actual_rows == expected_rows, full_test_name, error_message);
    }
    else
    {
        test_assert(actual_rows == expected_rows, full_test_name, NULL);
    }

    free(result_str);
}

void test_arrow_scan(chdb_connection conn)
{
    struct ArrowArrayStream stream;
    struct ArrowArrayStream stream2;
    struct ArrowArrayStream stream3;
    CustomStreamData * stream_data;
    CustomStreamData * stream_data2;
    CustomStreamData * stream_data3;
    const char* table_name = "test_arrow_table";
    const char* non_exist_table_name = "non_exist_table";
    const char* table_name2 = "test_arrow_table_2";
    const char* table_name3 = "test_arrow_table_3";
    chdb_arrow_stream arrow_stream;
    chdb_arrow_stream arrow_stream2;
    chdb_arrow_stream arrow_stream3;
    chdb_state result;
    chdb_result * count_result;
    chdb_result * sample_result;
    chdb_result * last_result;
    chdb_result * count1_result;
    chdb_result * count2_result;
    chdb_result * count3_result;
    chdb_result * join_result;
    chdb_result * union_result;
    chdb_result * unregister_result;
    const char * error;
    char error_message[512];

    printf("\n=== Testing ArrowArrayStream Scan Functions ===\n");
    printf("Data specification: 1,000,000 rows × 2 columns (id: int64, value: string)\n");

    memset(&stream, 0, sizeof(stream));

    /* Create and initialize stream data */
    stream_data = (CustomStreamData*)malloc(sizeof(CustomStreamData));
    init_custom_stream_data(stream_data);

    /* Set up the ArrowArrayStream callbacks */
    stream.get_schema = custom_get_schema;
    stream.get_next = custom_get_next;
    stream.get_last_error = custom_get_last_error;
    stream.release = custom_release;
    stream.private_data = stream_data;

    printf("✓ ArrowArrayStream initialization completed\n");
    printf("Starting registration with chDB...\n");

    arrow_stream = (chdb_arrow_stream)&stream;
    result = chdb_arrow_scan(conn, table_name, arrow_stream);

    /* Test 1: Verify arrow registration succeeded */
    test_assert_chdb_state(result, "Register ArrowArrayStream to table: test_arrow_table");

    /* Test 2: Unregister non-existent table should handle gracefully */
    result = chdb_arrow_unregister_table(conn, non_exist_table_name);
    test_assert_chdb_state(result, "Unregister non-existent table: non_exist_table");

    /* Test 3: Count rows - should be exactly 1,000,000 */
    count_result = chdb_query(conn, "SELECT COUNT(*) as total_rows FROM arrowstream(test_arrow_table)", "CSV");
    test_assert_row_count(count_result, 1000000, "Count total rows");
    chdb_destroy_query_result(count_result);

    /* Test 4: Sample first 5 rows - should contain id=0,1,2,3,4 */
    reset_arrow_stream(&stream);
    sample_result = chdb_query(conn, "SELECT * FROM arrowstream(test_arrow_table) LIMIT 5", "CSV");
    test_assert_query_result_contains(sample_result, "0,\"value_0\"", "First 5 rows contain first row");
    test_assert_query_result_contains(sample_result, "4,\"value_4\"", "First 5 rows contain fifth row");
    chdb_destroy_query_result(sample_result);

    /* Test 5: Sample last 5 rows - should contain id=999999,999998,999997,999996,999995 */
    reset_arrow_stream(&stream);
    last_result = chdb_query(conn, "SELECT * FROM arrowstream(test_arrow_table) ORDER BY id DESC LIMIT 5", "CSV");
    test_assert_query_result_contains(last_result, "999999,\"value_999999\"", "Last 5 rows contain last row");
    test_assert_query_result_contains(last_result, "999995,\"value_999995\"", "Last 5 rows contain fifth row");
    chdb_destroy_query_result(last_result);

    /* Test 6: Multiple table registration tests */
    /* Create second ArrowArrayStream with different data (500,000 rows) */
    memset(&stream2, 0, sizeof(stream2));
    stream_data2 = (CustomStreamData *)malloc(sizeof(CustomStreamData));
    init_custom_stream_data(stream_data2);
    stream_data2->total_rows = 500000;  /* Different row count */
    stream_data2->current_row = 0;
    stream2.get_schema = custom_get_schema;
    stream2.get_next = custom_get_next;
    stream2.get_last_error = custom_get_last_error;
    stream2.release = custom_release;
    stream2.private_data = stream_data2;

    /* Create third ArrowArrayStream with different data (100,000 rows) */
    memset(&stream3, 0, sizeof(stream3));
    stream_data3 = (CustomStreamData *)malloc(sizeof(CustomStreamData));
    init_custom_stream_data(stream_data3);
    stream_data3->total_rows = 100000;  /* Different row count */
    stream_data3->current_row = 0;
    stream3.get_schema = custom_get_schema;
    stream3.get_next = custom_get_next;
    stream3.get_last_error = custom_get_last_error;
    stream3.release = custom_release;
    stream3.private_data = stream_data3;

    /* Register second table */
    arrow_stream2 = (chdb_arrow_stream)&stream2;
    result = chdb_arrow_scan(conn, table_name2, arrow_stream2);
    test_assert_chdb_state(result, "Register second ArrowArrayStream to table: test_arrow_table_2");

    /* Register third table */
    arrow_stream3 = (chdb_arrow_stream)&stream3;
    result = chdb_arrow_scan(conn, table_name3, arrow_stream3);
    test_assert_chdb_state(result, "Register third ArrowArrayStream to table: test_arrow_table_3");

    /* Test 6a: Verify each table has correct row counts */
    reset_arrow_stream(&stream);
    count1_result = chdb_query(conn, "SELECT COUNT(*) FROM arrowstream(test_arrow_table)", "CSV");
    test_assert_row_count(count1_result, 1000000, "First table row count");
    chdb_destroy_query_result(count1_result);

    reset_arrow_stream(&stream2);
    count2_result = chdb_query(conn, "SELECT COUNT(*) FROM arrowstream(test_arrow_table_2)", "CSV");
    test_assert_row_count(count2_result, 500000, "Second table row count");
    chdb_destroy_query_result(count2_result);

    reset_arrow_stream(&stream3);
    count3_result = chdb_query(conn, "SELECT COUNT(*) FROM arrowstream(test_arrow_table_3)", "CSV");
    test_assert_row_count(count3_result, 100000, "Third table row count");
    chdb_destroy_query_result(count3_result);

    /* Test 6b: Test cross-table JOIN query */
    reset_arrow_stream(&stream);
    reset_arrow_stream(&stream2);
    join_result = chdb_query(conn,
        "SELECT t1.id, t1.value, t2.value as value2 "
        "FROM arrowstream(test_arrow_table) t1 "
        "INNER JOIN arrowstream(test_arrow_table_2) t2 ON t1.id = t2.id "
        "WHERE t1.id < 5 ORDER BY t1.id", "CSV");
    test_assert_query_result_contains(join_result, "0,\"value_0\",\"value_0\"", "JOIN query contains expected data");
    test_assert_query_result_contains(join_result, "4,\"value_4\",\"value_4\"", "JOIN query contains fifth row");
    chdb_destroy_query_result(join_result);

    /* Test 6c: Test UNION query across multiple tables */
    reset_arrow_stream(&stream2);
    reset_arrow_stream(&stream3);
    union_result = chdb_query(conn,
        "SELECT COUNT(*) FROM ("
        "SELECT id FROM arrowstream(test_arrow_table_2) WHERE id < 10 "
        "UNION ALL "
        "SELECT id FROM arrowstream(test_arrow_table_3) WHERE id < 10"
        ")", "CSV");
    test_assert_row_count(union_result, 20, "UNION query row count");
    chdb_destroy_query_result(union_result);

    /* Cleanup additional tables */
    result = chdb_arrow_unregister_table(conn, table_name2);
    test_assert_chdb_state(result, "Unregister second ArrowArrayStream table");

    result = chdb_arrow_unregister_table(conn, table_name3);
    test_assert_chdb_state(result, "Unregister third ArrowArrayStream table");

    /* Test 7: Unregister original table should succeed */
    result = chdb_arrow_unregister_table(conn, table_name);
    test_assert_chdb_state(result, "Unregister ArrowArrayStream table: test_arrow_table");

    /* Test 8: Sample last 5 rows after unregister should fail */
    reset_arrow_stream(&stream);
    unregister_result = chdb_query(conn, "SELECT * FROM arrowstream(test_arrow_table) ORDER BY id DESC LIMIT 5", "CSV");
    error = chdb_result_error(unregister_result);

    if (error)
    {
        snprintf(error_message, sizeof(error_message), "Got expected error: %s", error);
        test_assert(error != NULL, "Query after unregister should fail", error_message);
    }
    else
    {
        test_assert(error != NULL, "Query after unregister should fail", "No error returned when error was expected");
    }
    chdb_destroy_query_result(unregister_result);
}

// Release function for array children in create_arrow_array
static void release_array_child_id(struct ArrowArray* a)
{
    if (a->buffers)
    {
        free((void*)(uintptr_t)a->buffers[1]); // id data
        free((void**)(uintptr_t)a->buffers);
    }
    free(a);
}

// Release function for array children (string) in create_arrow_array
static void release_array_child_string(struct ArrowArray* a)
{
    if (a->buffers)
    {
        free((void*)(uintptr_t)a->buffers[1]); // offsets
        free((void*)(uintptr_t)a->buffers[2]); // string data
        free((void**)(uintptr_t)a->buffers);
    }
    free(a);
}

// Release function for main array in create_arrow_array
static void release_arrow_array_main(struct ArrowArray * a)
{
    if (a->children)
    {
        for (int64_t i = 0; i < a->n_children; i++)
        {
            if (a->children[i] && a->children[i]->release)
            {
                a->children[i]->release(a->children[i]);
            }
        }
        free(a->children);
    }

    if (a->buffers)
    {
        free((void**)(uintptr_t)a->buffers);
    }
}

// Helper function to create ArrowArray with specified row count
static void create_arrow_array(struct ArrowArray * array, uint64_t row_count)
{
    struct ArrowArray * id_array;
    struct ArrowArray * value_array;
    int64_t * id_data;
    int32_t * offsets;
    size_t total_string_size;
    char * string_data;
    size_t current_pos;
    uint64_t i;

    array->length = row_count;
    array->null_count = 0;
    array->offset = 0;
    array->n_buffers = 1;
    array->n_children = 2;
    array->buffers = (const void**)malloc(1 * sizeof(void*));
    array->buffers[0] = NULL; // validity buffer

    array->children = (struct ArrowArray**)malloc(2 * sizeof(struct ArrowArray*));
    array->dictionary = NULL;

    // Create id column (int64)
    array->children[0] = (struct ArrowArray*)malloc(sizeof(struct ArrowArray));
    id_array = array->children[0];
    id_array->length = row_count;
    id_array->null_count = 0;
    id_array->offset = 0;
    id_array->n_buffers = 2;
    id_array->n_children = 0;
    id_array->children = NULL;
    id_array->dictionary = NULL;

    id_array->buffers = (const void**)malloc(2 * sizeof(void*));
    id_array->buffers[0] = NULL; // validity buffer

    // Allocate and populate id data
    id_data = (int64_t*)malloc(row_count * sizeof(int64_t));
    for (i = 0; i < row_count; i++)
    {
        id_data[i] = (int64_t)i;
    }
    id_array->buffers[1] = id_data;
    id_array->release = release_array_child_id;

    // Create value column (string)
    array->children[1] = (struct ArrowArray*)malloc(sizeof(struct ArrowArray));
    value_array = array->children[1];
    value_array->length = row_count;
    value_array->null_count = 0;
    value_array->offset = 0;
    value_array->n_buffers = 3;
    value_array->n_children = 0;
    value_array->children = NULL;
    value_array->dictionary = NULL;

    value_array->buffers = (const void**)malloc(3 * sizeof(void*));
    value_array->buffers[0] = NULL; // validity buffer

    // Calculate total string data size and create offset array
    offsets = (int32_t*)malloc((row_count + 1) * sizeof(int32_t));
    total_string_size = 0;
    offsets[0] = 0;

    for (i = 0; i < row_count; i++)
    {
        char value_str[64];
        size_t len;
        snprintf(value_str, sizeof(value_str), "value_%llu", (unsigned long long)i);
        len = strlen(value_str);
        total_string_size += len;
        offsets[i + 1] = (int32_t)total_string_size;
    }

    value_array->buffers[1] = offsets;

    // Allocate and populate string data
    string_data = (char *)malloc(total_string_size);
    current_pos = 0;
    for (i = 0; i < row_count; i++) {
        char value_str[64];
        size_t len;
        snprintf(value_str, sizeof(value_str), "value_%llu", (unsigned long long)i);
        len = strlen(value_str);
        memcpy(string_data + current_pos, value_str, len);
        current_pos += len;
    }
    value_array->buffers[2] = string_data;
    value_array->release = release_array_child_string;

    // Set release callback for main array
    array->release = release_arrow_array_main;
}

void test_arrow_array_scan(chdb_connection conn)
{
    struct ArrowSchema schema;
    struct ArrowArray array;
    struct ArrowSchema schema2;
    struct ArrowArray array2;
    struct ArrowSchema schema3;
    struct ArrowArray array3;
    const char * table_name = "test_arrow_array_table";
    const char * non_exist_table_name = "non_exist_array_table";
    const char * table_name2 = "test_arrow_array_table_2";
    const char * table_name3 = "test_arrow_array_table_3";
    chdb_arrow_schema arrow_schema;
    chdb_arrow_array arrow_array;
    chdb_arrow_schema arrow_schema2;
    chdb_arrow_array arrow_array2;
    chdb_arrow_schema arrow_schema3;
    chdb_arrow_array arrow_array3;
    chdb_state result;
    chdb_result * count_result;
    chdb_result * sample_result;
    chdb_result * last_result;
    chdb_result * count2_result;
    chdb_result * count3_result;
    chdb_result * join_result;
    chdb_result * union_result;
    chdb_result * unregister_result;
    const char * error;
    char error_message[512];

    printf("\n=== Testing ArrowArray Scan Functions ===\n");
    printf("Data specification: 1,000,000 rows × 2 columns (id: int64, value: string)\n");

    // Create ArrowSchema (reuse existing function)
    create_schema(&schema);

    // Create ArrowArray with 1,000,000 rows
    memset(&array, 0, sizeof(array));
    create_arrow_array(&array, 1000000);

    printf("✓ ArrowArray initialization completed\n");
    printf("Starting registration with chDB...\n");

    arrow_schema = (chdb_arrow_schema)&schema;
    arrow_array = (chdb_arrow_array)&array;

    // Test 1: Register -> Query -> Unregister for row count
    result = chdb_arrow_array_scan(conn, table_name, arrow_schema, arrow_array);
    test_assert_chdb_state(result, "Register ArrowArray to table: test_arrow_array_table");

    count_result = chdb_query(conn, "SELECT COUNT(*) as total_rows FROM arrowstream(test_arrow_array_table)", "CSV");
    test_assert_row_count(count_result, 1000000, "Count total rows");
    chdb_destroy_query_result(count_result);

    result = chdb_arrow_unregister_table(conn, table_name);
    test_assert_chdb_state(result, "Unregister ArrowArray table after count query");

    // Test 2: Unregister non-existent table should handle gracefully
    result = chdb_arrow_unregister_table(conn, non_exist_table_name);
    test_assert_chdb_state(result, "Unregister non-existent array table: non_exist_array_table");

    // Test 3: Register -> Query -> Unregister for first 5 rows
    result = chdb_arrow_array_scan(conn, table_name, arrow_schema, arrow_array);
    test_assert_chdb_state(result, "Register ArrowArray for sample query");

    sample_result = chdb_query(conn, "SELECT * FROM arrowstream(test_arrow_array_table) LIMIT 5", "CSV");
    test_assert_query_result_contains(sample_result, "0,\"value_0\"", "First 5 rows contain first row");
    test_assert_query_result_contains(sample_result, "4,\"value_4\"", "First 5 rows contain fifth row");
    chdb_destroy_query_result(sample_result);

    result = chdb_arrow_unregister_table(conn, table_name);
    test_assert_chdb_state(result, "Unregister ArrowArray table after sample query");

    // Test 4: Register -> Query -> Unregister for last 5 rows
    result = chdb_arrow_array_scan(conn, table_name, arrow_schema, arrow_array);
    test_assert_chdb_state(result, "Register ArrowArray for last rows query");

    last_result = chdb_query(conn, "SELECT * FROM arrowstream(test_arrow_array_table) ORDER BY id DESC LIMIT 5", "CSV");
    test_assert_query_result_contains(last_result, "999999,\"value_999999\"", "Last 5 rows contain last row");
    test_assert_query_result_contains(last_result, "999995,\"value_999995\"", "Last 5 rows contain fifth row");
    chdb_destroy_query_result(last_result);

    result = chdb_arrow_unregister_table(conn, table_name);
    test_assert_chdb_state(result, "Unregister ArrowArray table after last rows query");

    // Test 5: Independent multiple table tests
    // Create second ArrowArray with different data (500,000 rows)
    create_schema(&schema2);
    memset(&array2, 0, sizeof(array2));
    create_arrow_array(&array2, 500000);

    // Create third ArrowArray with different data (100,000 rows)
    create_schema(&schema3);
    memset(&array3, 0, sizeof(array3));
    create_arrow_array(&array3, 100000);

    arrow_schema2 = (chdb_arrow_schema)&schema2;
    arrow_array2 = (chdb_arrow_array)&array2;
    arrow_schema3 = (chdb_arrow_schema)&schema3;
    arrow_array3 = (chdb_arrow_array)&array3;

    // Test 5a: Register -> Query -> Unregister for second table (500K rows)
    result = chdb_arrow_array_scan(conn, table_name2, arrow_schema2, arrow_array2);
    test_assert_chdb_state(result, "Register second ArrowArray to table: test_arrow_array_table_2");

    count2_result = chdb_query(conn, "SELECT COUNT(*) FROM arrowstream(test_arrow_array_table_2)", "CSV");
    test_assert_row_count(count2_result, 500000, "Second array table row count");
    chdb_destroy_query_result(count2_result);

    result = chdb_arrow_unregister_table(conn, table_name2);
    test_assert_chdb_state(result, "Unregister second ArrowArray table");

    // Test 5b: Register -> Query -> Unregister for third table (100K rows)
    result = chdb_arrow_array_scan(conn, table_name3, arrow_schema3, arrow_array3);
    test_assert_chdb_state(result, "Register third ArrowArray to table: test_arrow_array_table_3");

    count3_result = chdb_query(conn, "SELECT COUNT(*) FROM arrowstream(test_arrow_array_table_3)", "CSV");
    test_assert_row_count(count3_result, 100000, "Third array table row count");
    chdb_destroy_query_result(count3_result);

    result = chdb_arrow_unregister_table(conn, table_name3);
    test_assert_chdb_state(result, "Unregister third ArrowArray table");

    // Test 6: Cross-table JOIN query (Register both -> Query -> Unregister both)
    result = chdb_arrow_array_scan(conn, table_name, arrow_schema, arrow_array);
    test_assert_chdb_state(result, "Register first ArrowArray for JOIN");

    result = chdb_arrow_array_scan(conn, table_name2, arrow_schema2, arrow_array2);
    test_assert_chdb_state(result, "Register second ArrowArray for JOIN");

    join_result = chdb_query(conn,
        "SELECT t1.id, t1.value, t2.value as value2 "
        "FROM arrowstream(test_arrow_array_table) t1 "
        "INNER JOIN arrowstream(test_arrow_array_table_2) t2 ON t1.id = t2.id "
        "WHERE t1.id < 5 ORDER BY t1.id", "CSV");
    test_assert_query_result_contains(join_result, "0,\"value_0\",\"value_0\"", "Array JOIN query contains expected data");
    test_assert_query_result_contains(join_result, "4,\"value_4\",\"value_4\"", "Array JOIN query contains fifth row");
    chdb_destroy_query_result(join_result);

    result = chdb_arrow_unregister_table(conn, table_name);
    test_assert_chdb_state(result, "Unregister first ArrowArray after JOIN");

    result = chdb_arrow_unregister_table(conn, table_name2);
    test_assert_chdb_state(result, "Unregister second ArrowArray after JOIN");

    // Test 7: Cross-table UNION query (Register both -> Query -> Unregister both)
    result = chdb_arrow_array_scan(conn, table_name2, arrow_schema2, arrow_array2);
    test_assert_chdb_state(result, "Register second ArrowArray for UNION");

    result = chdb_arrow_array_scan(conn, table_name3, arrow_schema3, arrow_array3);
    test_assert_chdb_state(result, "Register third ArrowArray for UNION");

    union_result = chdb_query(conn,
        "SELECT COUNT(*) FROM ("
        "SELECT id FROM arrowstream(test_arrow_array_table_2) WHERE id < 10 "
        "UNION ALL "
        "SELECT id FROM arrowstream(test_arrow_array_table_3) WHERE id < 10"
        ")", "CSV");
    test_assert_row_count(union_result, 20, "Array UNION query row count");
    chdb_destroy_query_result(union_result);

    result = chdb_arrow_unregister_table(conn, table_name2);
    test_assert_chdb_state(result, "Unregister second ArrowArray after UNION");

    result = chdb_arrow_unregister_table(conn, table_name3);
    test_assert_chdb_state(result, "Unregister third ArrowArray after UNION");

    // Test 8: Query after unregister should fail
    unregister_result = chdb_query(conn, "SELECT * FROM arrowstream(test_arrow_array_table) ORDER BY id DESC LIMIT 5", "CSV");
    error = chdb_result_error(unregister_result);

    if (error)
    {
        snprintf(error_message, sizeof(error_message), "Got expected error: %s", error);
        test_assert(error != NULL, "Array query after unregister should fail", error_message);
    }
    else
    {
        test_assert(error != NULL, "Array query after unregister should fail", "No error returned when error was expected");
    }
    chdb_destroy_query_result(unregister_result);

    // Cleanup ArrowArrays and schemas
    if (array.release) array.release(&array);
    if (schema.release) schema.release(&schema);
    if (array2.release) array2.release(&array2);
    if (schema2.release) schema2.release(&schema2);
    if (array3.release) array3.release(&array3);
    if (schema3.release) schema3.release(&schema3);
}

int main(void)
{
    char * argv[] = {"clickhouse", "--multiquery"};
    int argc = sizeof(argv) / sizeof(argv[0]);
    chdb_connection * conn_ptr;
    chdb_connection conn;

    printf("=== chDB Arrow Functions Test ===\n");

    /* Create connection */
    conn_ptr = chdb_connect(argc, argv);
    if (!conn_ptr || !*conn_ptr) {
        printf("Failed to create chDB connection\n");
        exit(1);
    }

    conn = *conn_ptr;
    printf("✓ chDB connection established\n");

    /* Run test suites */
    test_arrow_scan(conn);
    test_arrow_array_scan(conn);

    /* Clean up */
    chdb_close_conn(conn_ptr);

    printf("\n=== chDB Arrow Functions Test Completed ===\n");

    return 0;
}
