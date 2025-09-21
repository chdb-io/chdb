#pragma once

#include <arrow/c/abi.h>

namespace CHDB
{

class ArrowSchemaWrapper
{
public:
	ArrowSchema arrow_schema;

	ArrowSchemaWrapper()
	{
		arrow_schema.release = nullptr;
	}

	~ArrowSchemaWrapper();
};

} // namespace CHDB
