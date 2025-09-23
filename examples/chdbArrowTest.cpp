#include <iostream>
#include <vector>
#include <cstdint>
#include <string>
#include <cstring>

#include "../programs/local/chdb.h"
#include "../contrib/arrow/cpp/src/arrow/c/abi.h"

// Custom ArrowArrayStream implementation data
struct CustomStreamData
{
    bool schema_sent;
    size_t current_row;
    size_t total_rows;
    size_t batch_size;
    std::string last_error;

    CustomStreamData() : schema_sent(false), current_row(0), total_rows(1000000), batch_size(10000) {}

    // Reset the stream to allow reading from the beginning
    void reset()
    {
        current_row = 0;
        last_error.clear();
    }
};

// Helper function to create schema with 2 columns: id(int64), value(string)
static void create_schema(struct ArrowSchema * schema) {
    schema->format = "+s";  // struct format
    schema->name = nullptr;
    schema->metadata = nullptr;
    schema->flags = 0;
    schema->n_children = 2;
    schema->children = static_cast<struct ArrowSchema **>(malloc(2 * sizeof(struct ArrowSchema *)));
    schema->dictionary = nullptr;
    schema->release = [](struct ArrowSchema * s)
    {
        if (s->children) {
            for (int64_t i = 0; i < s->n_children; i++) {
                if (s->children[i] && s->children[i]->release) {
                    s->children[i]->release(s->children[i]);
                }
                free(s->children[i]);
            }
            free(s->children);
        }
        s->release = nullptr;
    };

    // Field 0: id (int64)
    schema->children[0] = static_cast<struct ArrowSchema*>(malloc(sizeof(struct ArrowSchema)));
    schema->children[0]->format = "l";  // int64
    schema->children[0]->name = "id";
    schema->children[0]->metadata = nullptr;
    schema->children[0]->flags = 0;
    schema->children[0]->n_children = 0;
    schema->children[0]->children = nullptr;
    schema->children[0]->dictionary = nullptr;
    schema->children[0]->release = [](struct ArrowSchema* s) { s->release = nullptr; };

    // Field 1: value (string)
    schema->children[1] = static_cast<struct ArrowSchema*>(malloc(sizeof(struct ArrowSchema)));
    schema->children[1]->format = "u";  // utf8 string
    schema->children[1]->name = "value";
    schema->children[1]->metadata = nullptr;
    schema->children[1]->flags = 0;
    schema->children[1]->n_children = 0;
    schema->children[1]->children = nullptr;
    schema->children[1]->dictionary = nullptr;
    schema->children[1]->release = [](struct ArrowSchema* s) { s->release = nullptr; };
}

// Helper function to create a batch of data
static void create_batch(struct ArrowArray* array, size_t start_row, size_t batch_size)
{
    // Main array structure
    array->length = batch_size;
    array->null_count = 0;
    array->offset = 0;
    array->n_buffers = 1;
    array->n_children = 2;
    array->buffers = static_cast<const void**>(malloc(1 * sizeof(void*)));
    array->buffers[0] = nullptr;  // validity buffer (no nulls)
    array->children = static_cast<struct ArrowArray**>(malloc(2 * sizeof(struct ArrowArray*)));
    array->dictionary = nullptr;

    // Create id column (int64)
    array->children[0] = static_cast<struct ArrowArray*>(malloc(sizeof(struct ArrowArray)));
    struct ArrowArray* id_array = array->children[0];
    id_array->length = batch_size;
    id_array->null_count = 0;
    id_array->offset = 0;
    id_array->n_buffers = 2;
    id_array->n_children = 0;
    id_array->buffers = static_cast<const void**>(malloc(2 * sizeof(void*)));
    id_array->buffers[0] = nullptr;  // validity buffer

    // Allocate and fill id data
    int64_t* id_data = static_cast<int64_t*>(malloc(batch_size * sizeof(int64_t)));
    for (size_t i = 0; i < batch_size; i++)
        id_data[i] = start_row + i;

    id_array->buffers[1] = id_data;  // data buffer
    id_array->children = nullptr;
    id_array->dictionary = nullptr;
    id_array->release = [](struct ArrowArray* arr) 
    {
        if (arr->buffers) {
            free(const_cast<void*>(arr->buffers[1]));  // free data buffer
            free(const_cast<void**>(arr->buffers));
        }
        arr->release = nullptr;
    };

    // Create value column (string)
    array->children[1] = static_cast<struct ArrowArray*>(malloc(sizeof(struct ArrowArray)));
    struct ArrowArray* str_array = array->children[1];
    str_array->length = batch_size;
    str_array->null_count = 0;
    str_array->offset = 0;
    str_array->n_buffers = 3;
    str_array->n_children = 0;
    str_array->buffers = static_cast<const void**>(malloc(3 * sizeof(void*)));
    str_array->buffers[0] = nullptr;  // validity buffer

    // Create offset buffer (int32)
    int32_t* offsets = static_cast<int32_t*>(malloc((batch_size + 1) * sizeof(int32_t)));
    offsets[0] = 0;

    // Calculate total string length and create strings
    size_t total_str_len = 0;
    std::vector<std::string> strings;
    for (size_t i = 0; i < batch_size; i++)
    {
        std::string str = "value_" + std::to_string(start_row + i);
        strings.push_back(str);
        total_str_len += str.length();
        offsets[i + 1] = total_str_len;
    }
    str_array->buffers[1] = offsets;  // offset buffer

    // Create data buffer
    char* str_data = static_cast<char*>(malloc(total_str_len));
    size_t pos = 0;
    for (const auto& str : strings)
    {
        memcpy(str_data + pos, str.c_str(), str.length());
        pos += str.length();
    }
    str_array->buffers[2] = str_data;  // data buffer

    str_array->children = nullptr;
    str_array->dictionary = nullptr;
    str_array->release = [](struct ArrowArray* arr) 
    {
        if (arr->buffers) {
            free(const_cast<void*>(arr->buffers[1]));  // free offset buffer
            free(const_cast<void*>(arr->buffers[2]));  // free data buffer
            free(const_cast<void**>(arr->buffers));
        }
        arr->release = nullptr;
    };

    // Main array release function
    array->release = [](struct ArrowArray* arr) {
        if (arr->children) {
            for (int64_t i = 0; i < arr->n_children; i++) {
                if (arr->children[i] && arr->children[i]->release) {
                    arr->children[i]->release(arr->children[i]);
                }
                free(arr->children[i]);
            }
            free(arr->children);
        }
        if (arr->buffers) {
            free(const_cast<void**>(arr->buffers));
        }
        arr->release = nullptr;
    };
}

// Callback function to get schema
static int custom_get_schema(struct ArrowArrayStream * /* stream */, struct ArrowSchema * out)
{
    create_schema(out);
    return 0;
}

// Callback function to get next array
static int custom_get_next(struct ArrowArrayStream * stream, struct ArrowArray * out)
{
    auto* data = static_cast<CustomStreamData*>(stream->private_data);
    if (!data)
        return EINVAL;

    // Check if we've reached the end of the stream
    if (data->current_row >= data->total_rows)
    {
        // End of stream - set release to nullptr to indicate no more data
        out->release = nullptr;
        return 0;
    }

    // Calculate batch size for this iteration
    size_t remaining_rows = data->total_rows - data->current_row;
    size_t batch_size = std::min(data->batch_size, remaining_rows);

    // Create the batch
    create_batch(out, data->current_row, batch_size);

    data->current_row += batch_size;
    return 0;
}

// Callback function to get last error
static const char* custom_get_last_error(struct ArrowArrayStream* stream) {
    auto* data = static_cast<CustomStreamData*>(stream->private_data);
    if (!data || data->last_error.empty())
        return nullptr;

    return data->last_error.c_str();
}

// Callback function to release stream resources
static void custom_release(struct ArrowArrayStream* stream) {
    if (stream->private_data)
    {
        delete static_cast<CustomStreamData*>(stream->private_data);
        stream->private_data = nullptr;
    }
    stream->release = nullptr;
}

// Helper function to reset the ArrowArrayStream for reuse
static void reset_arrow_stream(struct ArrowArrayStream* stream)
{
    if (stream && stream->private_data)
    {
        auto* data = static_cast<CustomStreamData*>(stream->private_data);
        data->reset();
        std::cout << "✓ ArrowArrayStream has been reset, ready for re-reading\n";
    }
}

//===--------------------------------------------------------------------===//
// Unit Test Utilities
//===--------------------------------------------------------------------===//

static void test_assert(bool condition, const std::string& test_name, const std::string& message = "")
{
    if (condition)
    {
        std::cout << "✓ PASS: " << test_name << std::endl;
    }
    else
    {
        std::cout << "✗ FAIL: " << test_name;
        if (!message.empty())
        {
            std::cout << " - " << message;
        }
        std::cout << std::endl;
        exit(1);
    }
}

static void test_assert_chdb_state(chdb_state state, const std::string& operation_name)
{
    test_assert(state == CHDBSuccess,
                "chDB operation: " + operation_name,
                state == CHDBError ? "Operation failed" : "Unknown state");
}

static void test_assert_not_null(void* ptr, const std::string& test_name)
{
    test_assert(ptr != nullptr, test_name, "Pointer is null");
}

static void test_assert_no_error(chdb_result* result, const std::string& query_name)
{
    test_assert_not_null(result, query_name + " - Result is not null");

    const char * error = chdb_result_error(result);
    test_assert(error == nullptr,
                query_name + " - No query error",
                error ? std::string("Error: ") + error : "");
}

static void test_assert_query_result_contains(chdb_result* result, const std::string& expected_content, const std::string& query_name)
{
    test_assert_no_error(result, query_name);

    char * buffer = chdb_result_buffer(result);
    test_assert_not_null(buffer, query_name + " - Result buffer is not null");

    std::string result_str(buffer);
    test_assert(result_str.find(expected_content) != std::string::npos,
                query_name + " - Result contains expected content",
                "Expected: " + expected_content + ", Actual: " + result_str);
}

static void test_assert_row_count(chdb_result* result, uint64_t expected_rows, const std::string& query_name)
{
    test_assert_no_error(result, query_name);

    char* buffer = chdb_result_buffer(result);
    test_assert_not_null(buffer, query_name + " - Result buffer is not null");

    // Parse the count result (assuming CSV format with just the number)
    std::string result_str(buffer);
    // Remove trailing whitespace/newlines
    result_str.erase(result_str.find_last_not_of(" \t\n\r\f\v") + 1);

    uint64_t actual_rows = std::stoull(result_str);
    test_assert(actual_rows == expected_rows,
                query_name + " - Row count matches",
                "Expected: " + std::to_string(expected_rows) + ", Actual: " + std::to_string(actual_rows));
}

void test_arrow_scan(chdb_connection conn)
{
    std::cout << "\n=== Creating Custom ArrowArrayStream ===\n";
    std::cout << "Data specification: 1,000,000 rows × 2 columns (id: int64, value: string)\n";

    struct ArrowArrayStream stream;
    memset(&stream, 0, sizeof(stream));

    // Create and initialize stream data
    auto * stream_data = new CustomStreamData();

    // Set up the ArrowArrayStream callbacks
    stream.get_schema = custom_get_schema;
    stream.get_next = custom_get_next;
    stream.get_last_error = custom_get_last_error;
    stream.release = custom_release;
    stream.private_data = stream_data;

    std::cout << "✓ ArrowArrayStream initialization completed\n";
    std::cout << "Starting registration with chDB...\n";

    const char * table_name = "test_arrow_table";
    const char * non_exist_table_name = "non_exist_table";

    chdb_arrow_stream arrow_stream = reinterpret_cast<chdb_arrow_stream>(&stream);
    chdb_state result = chdb_arrow_scan(conn, table_name, arrow_stream);

    // Test 1: Verify arrow registration succeeded
    test_assert_chdb_state(result, "Register ArrowArrayStream to table: " + std::string(table_name));

    // Test 2: Unregister non-existent table should handle gracefully
    result = chdb_arrow_unregister_table(conn, non_exist_table_name);
    test_assert_chdb_state(result, "Unregister non-existent table: " + std::string(non_exist_table_name));

    // Test 3: Count rows - should be exactly 1,000,000
    chdb_result * count_result = chdb_query(conn, "SELECT COUNT(*) as total_rows FROM arrowstream(test_arrow_table)", "CSV");
    test_assert_row_count(count_result, 1000000, "Count total rows");
    chdb_destroy_query_result(count_result);

    // Test 4: Sample first 5 rows - should contain id=0,1,2,3,4
    reset_arrow_stream(&stream);
    chdb_result * sample_result = chdb_query(conn, "SELECT * FROM arrowstream(test_arrow_table) LIMIT 5", "CSV");
    test_assert_query_result_contains(sample_result, "0,\"value_0\"", "First 5 rows contain first row");
    test_assert_query_result_contains(sample_result, "4,\"value_4\"", "First 5 rows contain fifth row");
    chdb_destroy_query_result(sample_result);

    // Test 5: Sample last 5 rows - should contain id=999999,999998,999997,999996,999995
    reset_arrow_stream(&stream);
    chdb_result * last_result = chdb_query(conn, "SELECT * FROM arrowstream(test_arrow_table) ORDER BY id DESC LIMIT 5", "CSV");
    test_assert_query_result_contains(last_result, "999999,\"value_999999\"", "Last 5 rows contain last row");
    test_assert_query_result_contains(last_result, "999995,\"value_999995\"", "Last 5 rows contain fifth row");
    chdb_destroy_query_result(last_result);

    // Test 6: Multiple table registration tests
    // Create second ArrowArrayStream with different data (500,000 rows)
    struct ArrowArrayStream stream2;
    memset(&stream2, 0, sizeof(stream2));
    auto * stream_data2 = new CustomStreamData();
    stream_data2->total_rows = 500000;  // Different row count
    stream_data2->current_row = 0;
    stream2.get_schema = custom_get_schema;
    stream2.get_next = custom_get_next;
    stream2.get_last_error = custom_get_last_error;
    stream2.release = custom_release;
    stream2.private_data = stream_data2;

    // Create third ArrowArrayStream with different data (100,000 rows)
    struct ArrowArrayStream stream3;
    memset(&stream3, 0, sizeof(stream3));
    auto * stream_data3 = new CustomStreamData();
    stream_data3->total_rows = 100000;  // Different row count
    stream_data3->current_row = 0;
    stream3.get_schema = custom_get_schema;
    stream3.get_next = custom_get_next;
    stream3.get_last_error = custom_get_last_error;
    stream3.release = custom_release;
    stream3.private_data = stream_data3;

    const char * table_name2 = "test_arrow_table_2";
    const char * table_name3 = "test_arrow_table_3";

    // Register second table
    chdb_arrow_stream arrow_stream2 = reinterpret_cast<chdb_arrow_stream>(&stream2);
    result = chdb_arrow_scan(conn, table_name2, arrow_stream2);
    test_assert_chdb_state(result, "Register second ArrowArrayStream to table: " + std::string(table_name2));

    // Register third table
    chdb_arrow_stream arrow_stream3 = reinterpret_cast<chdb_arrow_stream>(&stream3);
    result = chdb_arrow_scan(conn, table_name3, arrow_stream3);
    test_assert_chdb_state(result, "Register third ArrowArrayStream to table: " + std::string(table_name3));

    // Test 6a: Verify each table has correct row counts
    reset_arrow_stream(&stream);
    chdb_result * count1_result = chdb_query(conn, "SELECT COUNT(*) FROM arrowstream(test_arrow_table)", "CSV");
    test_assert_row_count(count1_result, 1000000, "First table row count");
    chdb_destroy_query_result(count1_result);

    reset_arrow_stream(&stream2);
    chdb_result * count2_result = chdb_query(conn, "SELECT COUNT(*) FROM arrowstream(test_arrow_table_2)", "CSV");
    test_assert_row_count(count2_result, 500000, "Second table row count");
    chdb_destroy_query_result(count2_result);

    reset_arrow_stream(&stream3);
    chdb_result * count3_result = chdb_query(conn, "SELECT COUNT(*) FROM arrowstream(test_arrow_table_3)", "CSV");
    test_assert_row_count(count3_result, 100000, "Third table row count");
    chdb_destroy_query_result(count3_result);

    // Test 6b: Test cross-table JOIN query
    reset_arrow_stream(&stream);
    reset_arrow_stream(&stream2);
    chdb_result * join_result = chdb_query(conn,
        "SELECT t1.id, t1.value, t2.value as value2 "
        "FROM arrowstream(test_arrow_table) t1 "
        "INNER JOIN arrowstream(test_arrow_table_2) t2 ON t1.id = t2.id "
        "WHERE t1.id < 5 ORDER BY t1.id", "CSV");
    test_assert_query_result_contains(join_result, R"(0,"value_0","value_0")", "JOIN query contains expected data");
    test_assert_query_result_contains(join_result, R"(4,"value_4","value_4")", "JOIN query contains fifth row");
    chdb_destroy_query_result(join_result);

    // Test 6c: Test UNION query across multiple tables
    reset_arrow_stream(&stream2);
    reset_arrow_stream(&stream3);
    chdb_result * union_result = chdb_query(conn,
        "SELECT COUNT(*) FROM ("
        "SELECT id FROM arrowstream(test_arrow_table_2) WHERE id < 10 "
        "UNION ALL "
        "SELECT id FROM arrowstream(test_arrow_table_3) WHERE id < 10"
        ")", "CSV");
    test_assert_row_count(union_result, 20, "UNION query row count");
    chdb_destroy_query_result(union_result);

    // Cleanup additional tables
    result = chdb_arrow_unregister_table(conn, table_name2);
    test_assert_chdb_state(result, "Unregister second ArrowArrayStream table");

    result = chdb_arrow_unregister_table(conn, table_name3);
    test_assert_chdb_state(result, "Unregister third ArrowArrayStream table");

    // Test 7: Unregister original table should succeed
    result = chdb_arrow_unregister_table(conn, table_name);
    test_assert_chdb_state(result, "Unregister ArrowArrayStream table: " + std::string(table_name));

    // Test 8: Sample last 5 rows after unregister should fail
    reset_arrow_stream(&stream);
    chdb_result * unregister_result = chdb_query(conn, "SELECT * FROM arrowstream(test_arrow_table) ORDER BY id DESC LIMIT 5", "CSV");
    const char * error = chdb_result_error(unregister_result);
    test_assert(error != nullptr,
                "Query after unregister should fail",
                error ? std::string("Got expected error: ") + error : "No error returned when error was expected");
    chdb_destroy_query_result(unregister_result);
}

// Helper function to create ArrowArray with specified row count
static void create_arrow_array(struct ArrowArray * array, uint64_t row_count)
{
    array->length = row_count;
    array->null_count = 0;
    array->offset = 0;
    array->n_buffers = 1;
    array->n_children = 2;
    array->buffers = static_cast<const void **>(malloc(1 * sizeof(void *)));
    array->buffers[0] = nullptr; // validity buffer

    array->children = static_cast<struct ArrowArray **>(malloc(2 * sizeof(struct ArrowArray *)));
    array->dictionary = nullptr;

    // Create id column (int64)
    array->children[0] = static_cast<struct ArrowArray *>(malloc(sizeof(struct ArrowArray)));
    struct ArrowArray * id_array = array->children[0];
    id_array->length = row_count;
    id_array->null_count = 0;
    id_array->offset = 0;
    id_array->n_buffers = 2;
    id_array->n_children = 0;
    id_array->children = nullptr;
    id_array->dictionary = nullptr;

    id_array->buffers = static_cast<const void **>(malloc(2 * sizeof(void *)));
    id_array->buffers[0] = nullptr; // validity buffer

    // Allocate and populate id data
    int64_t * id_data = static_cast<int64_t *>(malloc(row_count * sizeof(int64_t)));
    for (uint64_t i = 0; i < row_count; i++)
    {
        id_data[i] = static_cast<int64_t>(i);
    }
    id_array->buffers[1] = id_data;

    id_array->release = [](struct ArrowArray * a)
    {
        if (a->buffers)
        {
            free(const_cast<void *>(a->buffers[1])); // id data
            free(const_cast<void **>(a->buffers));
        }
        free(a);
    };

    // Create value column (string)
    array->children[1] = static_cast<struct ArrowArray *>(malloc(sizeof(struct ArrowArray)));
    struct ArrowArray * value_array = array->children[1];
    value_array->length = row_count;
    value_array->null_count = 0;
    value_array->offset = 0;
    value_array->n_buffers = 3;
    value_array->n_children = 0;
    value_array->children = nullptr;
    value_array->dictionary = nullptr;

    value_array->buffers = static_cast<const void **>(malloc(3 * sizeof(void *)));
    value_array->buffers[0] = nullptr; // validity buffer

    // Calculate total string data size and create offset array
    int32_t * offsets = static_cast<int32_t *>(malloc((row_count + 1) * sizeof(int32_t)));
    size_t total_string_size = 0;
    offsets[0] = 0;

    for (uint64_t i = 0; i < row_count; i++)
    {
        std::string value_str = "value_" + std::to_string(i);
        total_string_size += value_str.length();
        offsets[i + 1] = static_cast<int32_t>(total_string_size);
    }

    value_array->buffers[1] = offsets;

    // Allocate and populate string data
    char * string_data = static_cast<char *>(malloc(total_string_size));
    size_t current_pos = 0;
    for (uint64_t i = 0; i < row_count; i++) {
        std::string value_str = "value_" + std::to_string(i);
        memcpy(string_data + current_pos, value_str.c_str(), value_str.length());
        current_pos += value_str.length();
    }
    value_array->buffers[2] = string_data;

    value_array->release = [](struct ArrowArray * a) {
        if (a->buffers) {
            free(const_cast<void *>(a->buffers[1])); // offsets
            free(const_cast<void *>(a->buffers[2])); // string data
            free(const_cast<void **>(a->buffers));
        }
        free(a);
    };

    // Set release callback for main array
    array->release = [](struct ArrowArray * a)
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
        if (a->buffers) {
            free(const_cast<void **>(a->buffers));
        }
    };
}

void test_arrow_array_scan(chdb_connection conn)
{
    std::cout << "\n=== Testing ArrowArray Scan Functions ===\n";
    std::cout << "Data specification: 1,000,000 rows × 2 columns (id: int64, value: string)\n";

    // Create ArrowSchema (reuse existing function)
    struct ArrowSchema schema;
    create_schema(&schema);

    // Create ArrowArray with 1,000,000 rows
    struct ArrowArray array;
    memset(&array, 0, sizeof(array));
    create_arrow_array(&array, 1000000);

    std::cout << "✓ ArrowArray initialization completed\n";
    std::cout << "Starting registration with chDB...\n";

    const char * table_name = "test_arrow_array_table";
    const char * non_exist_table_name = "non_exist_array_table";

    chdb_arrow_schema arrow_schema = reinterpret_cast<chdb_arrow_schema>(&schema);
    chdb_arrow_array arrow_array = reinterpret_cast<chdb_arrow_array>(&array);

    // Test 1: Register -> Query -> Unregister for row count
    chdb_state result = chdb_arrow_array_scan(conn, table_name, arrow_schema, arrow_array);
    test_assert_chdb_state(result, "Register ArrowArray to table: " + std::string(table_name));

    chdb_result * count_result = chdb_query(conn, "SELECT COUNT(*) as total_rows FROM arrowstream(test_arrow_array_table)", "CSV");
    test_assert_row_count(count_result, 1000000, "Count total rows");
    chdb_destroy_query_result(count_result);

    result = chdb_arrow_unregister_table(conn, table_name);
    test_assert_chdb_state(result, "Unregister ArrowArray table after count query");

    // Test 2: Unregister non-existent table should handle gracefully
    result = chdb_arrow_unregister_table(conn, non_exist_table_name);
    test_assert_chdb_state(result, "Unregister non-existent array table: " + std::string(non_exist_table_name));

    // Test 3: Register -> Query -> Unregister for first 5 rows
    result = chdb_arrow_array_scan(conn, table_name, arrow_schema, arrow_array);
    test_assert_chdb_state(result, "Register ArrowArray for sample query");

    chdb_result * sample_result = chdb_query(conn, "SELECT * FROM arrowstream(test_arrow_array_table) LIMIT 5", "CSV");
    test_assert_query_result_contains(sample_result, "0,\"value_0\"", "First 5 rows contain first row");
    test_assert_query_result_contains(sample_result, "4,\"value_4\"", "First 5 rows contain fifth row");
    chdb_destroy_query_result(sample_result);

    result = chdb_arrow_unregister_table(conn, table_name);
    test_assert_chdb_state(result, "Unregister ArrowArray table after sample query");

    // Test 4: Register -> Query -> Unregister for last 5 rows
    result = chdb_arrow_array_scan(conn, table_name, arrow_schema, arrow_array);
    test_assert_chdb_state(result, "Register ArrowArray for last rows query");

    chdb_result * last_result = chdb_query(conn, "SELECT * FROM arrowstream(test_arrow_array_table) ORDER BY id DESC LIMIT 5", "CSV");
    test_assert_query_result_contains(last_result, "999999,\"value_999999\"", "Last 5 rows contain last row");
    test_assert_query_result_contains(last_result, "999995,\"value_999995\"", "Last 5 rows contain fifth row");
    chdb_destroy_query_result(last_result);

    result = chdb_arrow_unregister_table(conn, table_name);
    test_assert_chdb_state(result, "Unregister ArrowArray table after last rows query");

    // Test 5: Independent multiple table tests
    // Create second ArrowArray with different data (500,000 rows)
    struct ArrowSchema schema2;
    create_schema(&schema2);
    struct ArrowArray array2;
    memset(&array2, 0, sizeof(array2));
    create_arrow_array(&array2, 500000);

    // Create third ArrowArray with different data (100,000 rows)
    struct ArrowSchema schema3;
    create_schema(&schema3);
    struct ArrowArray array3;
    memset(&array3, 0, sizeof(array3));
    create_arrow_array(&array3, 100000);

    const char * table_name2 = "test_arrow_array_table_2";
    const char * table_name3 = "test_arrow_array_table_3";

    chdb_arrow_schema arrow_schema2 = reinterpret_cast<chdb_arrow_schema>(&schema2);
    chdb_arrow_array arrow_array2 = reinterpret_cast<chdb_arrow_array>(&array2);
    chdb_arrow_schema arrow_schema3 = reinterpret_cast<chdb_arrow_schema>(&schema3);
    chdb_arrow_array arrow_array3 = reinterpret_cast<chdb_arrow_array>(&array3);

    // Test 5a: Register -> Query -> Unregister for second table (500K rows)
    result = chdb_arrow_array_scan(conn, table_name2, arrow_schema2, arrow_array2);
    test_assert_chdb_state(result, "Register second ArrowArray to table: " + std::string(table_name2));

    chdb_result * count2_result = chdb_query(conn, "SELECT COUNT(*) FROM arrowstream(test_arrow_array_table_2)", "CSV");
    test_assert_row_count(count2_result, 500000, "Second array table row count");
    chdb_destroy_query_result(count2_result);

    result = chdb_arrow_unregister_table(conn, table_name2);
    test_assert_chdb_state(result, "Unregister second ArrowArray table");

    // Test 5b: Register -> Query -> Unregister for third table (100K rows)
    result = chdb_arrow_array_scan(conn, table_name3, arrow_schema3, arrow_array3);
    test_assert_chdb_state(result, "Register third ArrowArray to table: " + std::string(table_name3));

    chdb_result * count3_result = chdb_query(conn, "SELECT COUNT(*) FROM arrowstream(test_arrow_array_table_3)", "CSV");
    test_assert_row_count(count3_result, 100000, "Third array table row count");
    chdb_destroy_query_result(count3_result);

    result = chdb_arrow_unregister_table(conn, table_name3);
    test_assert_chdb_state(result, "Unregister third ArrowArray table");

    // Test 6: Cross-table JOIN query (Register both -> Query -> Unregister both)
    result = chdb_arrow_array_scan(conn, table_name, arrow_schema, arrow_array);
    test_assert_chdb_state(result, "Register first ArrowArray for JOIN");

    result = chdb_arrow_array_scan(conn, table_name2, arrow_schema2, arrow_array2);
    test_assert_chdb_state(result, "Register second ArrowArray for JOIN");

    chdb_result * join_result = chdb_query(conn,
        "SELECT t1.id, t1.value, t2.value as value2 "
        "FROM arrowstream(test_arrow_array_table) t1 "
        "INNER JOIN arrowstream(test_arrow_array_table_2) t2 ON t1.id = t2.id "
        "WHERE t1.id < 5 ORDER BY t1.id", "CSV");
    test_assert_query_result_contains(join_result, R"(0,"value_0","value_0")", "Array JOIN query contains expected data");
    test_assert_query_result_contains(join_result, R"(4,"value_4","value_4")", "Array JOIN query contains fifth row");
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

    chdb_result * union_result = chdb_query(conn,
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
    chdb_result * unregister_result = chdb_query(conn, "SELECT * FROM arrowstream(test_arrow_array_table) ORDER BY id DESC LIMIT 5", "CSV");
    const char * error = chdb_result_error(unregister_result);
    test_assert(error != nullptr,
                "Array query after unregister should fail",
                error ? std::string("Got expected error: ") + error : "No error returned when error was expected");
    chdb_destroy_query_result(unregister_result);

    // Cleanup ArrowArrays and schemas
    if (array.release) array.release(&array);
    if (schema.release) schema.release(&schema);
    if (array2.release) array2.release(&array2);
    if (schema2.release) schema2.release(&schema2);
    if (array3.release) array3.release(&array3);
    if (schema3.release) schema3.release(&schema3);
}

int main()
{
    const char *argv[] = {"clickhouse", "--multiquery"};
    int argc = sizeof(argv) / sizeof(argv[0]);
    chdb_connection * conn_ptr;
    chdb_connection conn;

    std::cout << "=== chDB Arrow Functions Test ===\n";

    // Create connection
    conn_ptr = chdb_connect(argc, const_cast<char**>(argv));
    if (!conn_ptr || !*conn_ptr) {
        std::cout << "Failed to create chDB connection\n";
        return 1;
    }

    conn = *conn_ptr;
    std::cout << "✓ chDB connection established\n";

    // Run test suites
    test_arrow_scan(conn);
    test_arrow_array_scan(conn);

    // Clean up
    chdb_close_conn(conn_ptr);

    std::cout << "\n=== chDB Arrow Functions Test Completed ===\n";

    return 0;
}
