#pragma once

#include "PythonImportCache.h"
#include "PythonImportCacheItem.h"

#include <stack>

namespace CHDB {

struct PythonImporter {
public:
	static py::handle Import(std::stack<PythonImportCacheItem *> & hierarchy, bool load = true);

	static PythonImportCache & ImportCache();

private:
	static PythonImportCachePtr python_import_cache;
};

} // namespace CHDB
