#pragma once

#include "PythonImportCacheItem.h"

namespace CHDB {

struct PytzCacheItem : public PythonImportCacheItem
{
public:
	static constexpr const char *Name = "pytz";

	PytzCacheItem() : PythonImportCacheItem("pytz"), timezone("timezone", this) {}

	~PytzCacheItem() override = default;

	PythonImportCacheItem timezone;
};

} // namespace CHDB
