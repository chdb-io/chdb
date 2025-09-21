#include "chdb.h"
#include "chdb-internal.h"
#include "ArrowStreamRegistry.h"

#include <shared_mutex>
#include <arrow/c/abi.h>
#include <base/defines.h>

namespace CHDB
{

struct PrivateData {
    ArrowSchema * schema;
    ArrowArray * array;
    bool done = false;
};

void EmptySchemaRelease(ArrowSchema * schema)
{
    schema->release = nullptr;
}

void EmptyArrayRelease(ArrowArray * array)
{
    array->release = nullptr;
}

void EmptyStreamRelease(ArrowArrayStream * stream)
{
    stream->release = nullptr;
}

int GetSchema(struct ArrowArrayStream * stream, struct ArrowSchema * out)
{
	auto * private_data = static_cast<PrivateData *>((stream->private_data));
	if (private_data->schema == nullptr)
		return CHDBError;

	*out = *private_data->schema;
	out->release = EmptySchemaRelease;
	return CHDBSuccess;
}

int GetNext(struct ArrowArrayStream * stream, struct ArrowArray * out)
{
	auto * private_data = static_cast<PrivateData *>((stream->private_data));
	*out = *private_data->array;
	if (private_data->done)
    {
		out->release = nullptr;
    }
	else
    {
		out->release = EmptyArrayRelease;
    }

	private_data->done = true;
	return CHDBSuccess;
}

const char * GetLastError(struct ArrowArrayStream * /*stream*/)
{
	return nullptr;
}

void Release(struct ArrowArrayStream * stream)
{
	if (stream->private_data != nullptr)
		delete reinterpret_cast<PrivateData *>(stream->private_data);

	stream->private_data = nullptr;
	stream->release = nullptr;
}

} // namespace CHDB

chdb_state chdb_arrow_scan(
    chdb_connection conn, const char * table_name,
    chdb_arrow_stream arrow_stream)
{
    ChdbDestructorGuard guard;

    std::shared_lock<std::shared_mutex> global_lock(global_connection_mutex);

    if (!table_name || !arrow_stream)
        return CHDBError;

    auto * connection = reinterpret_cast<chdb_conn *>(conn);
    if (!checkConnectionValidity(connection))
        return CHDBError;

    auto * stream = reinterpret_cast<ArrowArrayStream *>(arrow_stream);

    ArrowSchema schema;
    if (stream->get_schema(stream, &schema) == CHDBError)
        return CHDBError;

    using ReleaseFunction = void (*)(ArrowSchema *);
    std::vector<ReleaseFunction> releases(static_cast<size_t>(schema.n_children));
    for (size_t i = 0; i < static_cast<size_t>(schema.n_children); i++)
    {
        auto * child = schema.children[i];
        releases[i] = child->release;
        child->release = CHDB::EmptySchemaRelease;
    }

    try
    {
        bool success = DB::ArrowStreamRegistry::instance().registerArrowStream(String(table_name), stream);
        return success ? CHDBSuccess : CHDBError;
    }
    catch (...)
    {
        return CHDBError;
    }

    for (size_t i = 0; i < static_cast<size_t>(schema.n_children); ++i)
    {
        schema.children[i]->release = releases[i];
    }

    return CHDBSuccess;
}

chdb_state chdb_arrow_array_scan(
    chdb_connection conn, const char * table_name,
    chdb_arrow_schema arrow_schema, chdb_arrow_array arrow_array,
    chdb_arrow_stream * out_stream)
{
    auto * private_data = new CHDB::PrivateData;
	private_data->schema = reinterpret_cast<ArrowSchema *>(arrow_schema);
	private_data->array = reinterpret_cast<ArrowArray *>(arrow_array);
	private_data->done = false;

	auto * stream = new ArrowArrayStream();
	*out_stream = reinterpret_cast<chdb_arrow_stream>(stream);
	stream->get_schema = CHDB::GetSchema;
	stream->get_next = CHDB::GetNext;
	stream->get_last_error = CHDB::GetLastError;
	stream->release = CHDB::Release;
	stream->private_data = private_data;

	return chdb_arrow_scan(conn, table_name, reinterpret_cast<chdb_arrow_stream>(stream));
}

void chdb_destroy_arrow_stream(chdb_arrow_stream * arrow_stream)
{
    if (!arrow_stream)
        return;

    auto * stream = reinterpret_cast<ArrowArrayStream *>(*arrow_stream);
	if (!stream)
		return;

	if (stream->release)
		stream->release(stream);
	chassert(!stream->release);

	delete stream;
	*arrow_stream = nullptr;
}

chdb_state chdb_arrow_unregister_table(chdb_connection conn, const char * table_name)
{
    ChdbDestructorGuard guard;

    std::shared_lock<std::shared_mutex> global_lock(global_connection_mutex);

    if (!table_name)
        return CHDBError;

    auto * connection = reinterpret_cast<chdb_conn *>(conn);
    if (!checkConnectionValidity(connection))
        return CHDBError;

    try
    {
        DB::ArrowStreamRegistry::instance().unregisterArrowStream(String(table_name));
        return CHDBSuccess;
    }
    catch (...)
    {
        return CHDBError;
    }
}
