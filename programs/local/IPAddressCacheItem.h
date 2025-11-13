#pragma once

#include "PythonImportCacheItem.h"

namespace CHDB {

struct IPAddressCacheItem : public PythonImportCacheItem
{
public:
	static constexpr const char * Name = "ipaddress";

	IPAddressCacheItem()
        : PythonImportCacheItem("ipaddress")
        , ipv4_address("IPv4Address", this)
        , ipv6_address("IPv6Address", this)
	{
	}

	~IPAddressCacheItem() override = default;

	PythonImportCacheItem ipv4_address;
	PythonImportCacheItem ipv6_address;
};

} // namespace CHDB
