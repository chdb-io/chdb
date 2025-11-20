#pragma once

#include "PythonImportCacheItem.h"

namespace CHDB {

struct UUIDCacheItem : public PythonImportCacheItem
{
public:
	static constexpr const char * Name = "uuid";

	UUIDCacheItem() : PythonImportCacheItem("uuid"), UUID("UUID", this)
	{
	}

	~UUIDCacheItem() override = default;

	PythonImportCacheItem UUID;
};

} // namespace CHDB
