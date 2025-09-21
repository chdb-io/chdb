#pragma once

#include "PythonImportCacheItem.h"

namespace CHDB
{

struct PyarrowIpcCacheItem : public PythonImportCacheItem
{
	explicit PyarrowIpcCacheItem(PythonImportCacheItem * parent)
		: PythonImportCacheItem("ipc", parent), message_reader("MessageReader", this)
	{}
	~PyarrowIpcCacheItem() override = default;

	PythonImportCacheItem message_reader;
};

struct PyarrowDatasetCacheItem : public PythonImportCacheItem
{
	static constexpr const char * Name = "pyarrow.dataset";

	PyarrowDatasetCacheItem()
		: PythonImportCacheItem("pyarrow.dataset"), scanner("Scanner", this), dataset("Dataset", this)
	{}
	~PyarrowDatasetCacheItem() override = default;

	PythonImportCacheItem scanner;
	PythonImportCacheItem dataset;
};

struct PyarrowCacheItem : public PythonImportCacheItem
{
	static constexpr const char * Name = "pyarrow";

	PyarrowCacheItem()
	    : PythonImportCacheItem("pyarrow"), dataset(), table("Table", this),
		record_batch_reader("RecordBatchReader", this), ipc(this)
	{}
	~PyarrowCacheItem() override = default;

	PyarrowDatasetCacheItem dataset;
	PythonImportCacheItem table;
	PythonImportCacheItem record_batch_reader;
	PyarrowIpcCacheItem ipc;
};

} // namespace CHDB
