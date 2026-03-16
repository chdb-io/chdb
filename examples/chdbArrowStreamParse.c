/**
 * chdbArrowStreamParse.c
 *
 * Query chDB with "ArrowStream" format, then use nanoarrow (Apache Arrow
 * official lightweight C library) to decode and print the result.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanoarrow/nanoarrow.h"
#include "nanoarrow/nanoarrow_ipc.h"
#include "chdb.h"

/* ----------------------------------------------------------------
 * Helper: wrap a raw (ptr, len) as an ArrowIpcInputStream
 * ---------------------------------------------------------------- */
typedef struct
{
    const uint8_t * data;
    int64_t        len;
    int64_t        pos;
} BufStreamCtx;

static ArrowErrorCode buf_stream_read(struct ArrowIpcInputStream * stream,
                                      uint8_t * buf, int64_t buf_size,
                                      int64_t * size_read_out,
                                      struct ArrowError *error)
{
    (void)error;
    BufStreamCtx * ctx = (BufStreamCtx *)stream->private_data;
    int64_t avail = ctx->len - ctx->pos;
    int64_t n = buf_size < avail ? buf_size : avail;
    if (n > 0) {
        memcpy(buf, ctx->data + ctx->pos, (size_t)n);
        ctx->pos += n;
    }
    *size_read_out = n;
    return NANOARROW_OK;
}

static void buf_stream_release(struct ArrowIpcInputStream * stream)
{
    free(stream->private_data);
    stream->release = NULL;
}

static void init_buf_stream(struct ArrowIpcInputStream * stream,
                            const uint8_t * data, int64_t len)
{
    BufStreamCtx * ctx = (BufStreamCtx *)malloc(sizeof(BufStreamCtx));
    ctx->data = data;
    ctx->len  = len;
    ctx->pos  = 0;
    stream->read         = buf_stream_read;
    stream->release      = buf_stream_release;
    stream->private_data = ctx;
}

/* ----------------------------------------------------------------
 * Print schema info
 * ---------------------------------------------------------------- */
static void print_schema(const struct ArrowSchema * schema)
{
    printf("  Schema (%lld columns):\n", (long long)schema->n_children);
    for (int64_t i = 0; i < schema->n_children; i++) {
        const struct ArrowSchema *child = schema->children[i];
        printf("    [%lld] %-20s format=%s\n",
               (long long)i,
               child->name ? child->name : "(null)",
               child->format ? child->format : "?");
    }
}

/* ----------------------------------------------------------------
 * Print one batch using ArrowArrayView
 * ---------------------------------------------------------------- */
static void print_batch(const struct ArrowSchema *schema,
                        const struct ArrowArray *array)
{
    int64_t ncols = schema->n_children;
    int64_t nrows = array->length;
    int64_t show  = nrows > 20 ? 20 : nrows;

    struct ArrowArrayView **views =
        (struct ArrowArrayView **)calloc((size_t)ncols, sizeof(void *));

    for (int64_t c = 0; c < ncols; c++) {
        views[c] = (struct ArrowArrayView *)calloc(1, sizeof(struct ArrowArrayView));
        ArrowArrayViewInitFromSchema(views[c], schema->children[c], NULL);
        ArrowArrayViewSetArray(views[c], array->children[c], NULL);
    }

    /* header */
    printf("  ");
    for (int64_t c = 0; c < ncols; c++)
        printf("%-20s", schema->children[c]->name ? schema->children[c]->name : "?");
    printf("\n  ");
    for (int64_t c = 0; c < ncols; c++)
        printf("%-20s", "-------------------");
    printf("\n");

    /* rows */
    for (int64_t row = 0; row < show; row++) {
        printf("  ");
        for (int64_t c = 0; c < ncols; c++) {
            if (ArrowArrayViewIsNull(views[c], row)) {
                printf("%-20s", "NULL");
                continue;
            }

            enum ArrowType t = views[c]->storage_type;
            switch (t) {
            case NANOARROW_TYPE_UINT8:
            case NANOARROW_TYPE_UINT16:
            case NANOARROW_TYPE_UINT32:
            case NANOARROW_TYPE_UINT64:
                printf("%-20llu",
                       (unsigned long long)ArrowArrayViewGetUIntUnsafe(views[c], row));
                break;
            case NANOARROW_TYPE_INT8:
            case NANOARROW_TYPE_INT16:
            case NANOARROW_TYPE_INT32:
            case NANOARROW_TYPE_INT64:
                printf("%-20lld",
                       (long long)ArrowArrayViewGetIntUnsafe(views[c], row));
                break;
            case NANOARROW_TYPE_FLOAT:
            case NANOARROW_TYPE_DOUBLE:
                printf("%-20.6f",
                       ArrowArrayViewGetDoubleUnsafe(views[c], row));
                break;
            case NANOARROW_TYPE_STRING:
            case NANOARROW_TYPE_LARGE_STRING: {
                struct ArrowStringView sv =
                    ArrowArrayViewGetStringUnsafe(views[c], row);
                int len = sv.size_bytes > 19 ? 19 : (int)sv.size_bytes;
                printf("%-20.*s", len, sv.data);
                break;
            }
            default:
                printf("%-20s", "(unsupported)");
            }
        }
        printf("\n");
    }

    if (nrows > show)
        printf("  ... (%lld more rows)\n", (long long)(nrows - show));

    for (int64_t c = 0; c < ncols; c++) {
        ArrowArrayViewReset(views[c]);
        free(views[c]);
    }
    free(views);
}

/* ----------------------------------------------------------------
 * Run one query and print results
 * ---------------------------------------------------------------- */
static int run_and_print(chdb_connection conn, const char *sql)
{
    printf("SQL: %s\n", sql);

    chdb_result * result = chdb_query(conn, sql, "ArrowStream");
    const char *error = chdb_result_error(result);
    if (error) {
        fprintf(stderr, "  ERROR: %s\n", error);
        chdb_destroy_query_result(result);
        return -1;
    }
    const uint8_t * buf = (const uint8_t *)chdb_result_buffer(result);
    size_t len = chdb_result_length(result);
    printf("  ArrowStream payload: %zu bytes\n", len);

    /* Wrap raw bytes as an ArrowIpcInputStream */
    struct ArrowIpcInputStream input;
    init_buf_stream(&input, buf, (int64_t)len);

    /* Decode IPC stream → ArrowArrayStream (Arrow C Data Interface) */
    struct ArrowArrayStream stream;
    int rc = ArrowIpcArrayStreamReaderInit(&stream, &input, NULL);
    if (rc != NANOARROW_OK) {
        fprintf(stderr, "  Failed to init IPC reader: %d\n", rc);
        if (input.release) input.release(&input);
        chdb_destroy_query_result(result);
        return -1;
    }

    /* Read schema */
    struct ArrowSchema schema;
    memset(&schema, 0, sizeof(schema));
    rc = stream.get_schema(&stream, &schema);
    if (rc != 0) {
        fprintf(stderr, "  get_schema failed: %s\n",
                stream.get_last_error(&stream));
        stream.release(&stream);
        chdb_destroy_query_result(result);
        return -1;
    }
    print_schema(&schema);

    /* Read record batches */
    int batch_no = 0;
    struct ArrowArray batch;
    memset(&batch, 0, sizeof(batch));

    while (1) {
        rc = stream.get_next(&stream, &batch);
        if (rc != 0) {
            fprintf(stderr, "  get_next failed: %s\n",
                    stream.get_last_error(&stream));
            break;
        }
        if (batch.release == NULL)
            break;

        printf("  Batch #%d (%lld rows):\n", batch_no++, (long long)batch.length);
        print_batch(&schema, &batch);
        batch.release(&batch);
    }

    schema.release(&schema);
    stream.release(&stream);
    chdb_destroy_query_result(result);
    printf("\n");
    return 0;
}

/* ---------------------------------------------------------------- */
int main(void)
{
    char *argv[] = {"clickhouse", "--multiquery"};
    int argc = sizeof(argv) / sizeof(argv[0]);

    printf("=== chDB ArrowStream Parse (nanoarrow) ===\n\n");

    chdb_connection * conn_ptr = chdb_connect(argc, argv);
    if (!conn_ptr || !*conn_ptr) {
        fprintf(stderr, "Failed to connect\n");
        return 1;
    }
    chdb_connection conn = *conn_ptr;

    run_and_print(conn,
        "SELECT number AS id, number * 2 AS doubled, "
        "number * number AS squared FROM numbers(10)");

    run_and_print(conn,
        "SELECT number AS id, "
        "concat('user_', toString(number)) AS name, "
        "number * 100 AS score FROM numbers(8)");

    run_and_print(conn,
        "SELECT number % 3 AS grp, count() AS cnt, "
        "sum(number) AS total FROM numbers(100) "
        "GROUP BY grp ORDER BY grp");

    run_and_print(conn,
        "SELECT number AS n, "
        "toFloat64(number) / 3.0 AS ratio "
        "FROM numbers(6)");

    chdb_close_conn(conn_ptr);
    printf("=== Done ===\n");
    return 0;
}
