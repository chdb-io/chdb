#pragma once

#include "DatetimeCacheItem.h"
#include "DecimalCacheItem.h"
#include "PandasCacheItem.h"
#include "PyArrowCacheItem.h"
#include "PythonImportCacheItem.h"

#include <vector>

namespace CHDB {

struct PythonImportCache;
using PythonImportCachePtr = std::shared_ptr<PythonImportCache>;

struct PythonImportCache {
public:
	explicit PythonImportCache()  = default;

	~PythonImportCache();

	PandasCacheItem pandas;
	PyarrowCacheItem pyarrow;
	DatetimeCacheItem datetime;
	DecimalCacheItem decimal;

	py::handle AddCache(py::object item);

private:
	std::vector<py::object> owned_objects;
};

} // namespace CHDB
