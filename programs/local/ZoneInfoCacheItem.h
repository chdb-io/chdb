#pragma once

#include "PythonImportCacheItem.h"

namespace CHDB {

struct ZoneInfoCacheItem : public PythonImportCacheItem
{
public:
	static constexpr const char *Name = "zoneinfo";

	ZoneInfoCacheItem() : PythonImportCacheItem("zoneinfo"), ZoneInfo("ZoneInfo", this) {}

	~ZoneInfoCacheItem() override = default;

	PythonImportCacheItem ZoneInfo;
};

} // namespace CHDB
