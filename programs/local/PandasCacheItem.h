#pragma once

#include "PythonImportCacheItem.h"

namespace CHDB {

struct PandasCacheItem : public PythonImportCacheItem {

public:
	static constexpr const char *Name = "pandas";

public:
	PandasCacheItem()
	    : PythonImportCacheItem("pandas"), DataFrame("DataFrame", this), Categorical("Categorical", this),
	      CategoricalDtype("CategoricalDtype", this), Series("Series", this), NaT("NaT", this), NA("NA", this),
	      isnull("isnull", this), ArrowDtype("ArrowDtype", this), BooleanDtype("BooleanDtype", this),
	      UInt8Dtype("UInt8Dtype", this), UInt16Dtype("UInt16Dtype", this), UInt32Dtype("UInt32Dtype", this),
	      UInt64Dtype("UInt64Dtype", this), Int8Dtype("Int8Dtype", this), Int16Dtype("Int16Dtype", this),
	      Int32Dtype("Int32Dtype", this), Int64Dtype("Int64Dtype", this), Float32Dtype("Float32Dtype", this),
	      Float64Dtype("Float64Dtype", this) {
	}
	~PandasCacheItem() override = default;

	PythonImportCacheItem DataFrame;
	PythonImportCacheItem Categorical;
	PythonImportCacheItem CategoricalDtype;
	PythonImportCacheItem Series;
	PythonImportCacheItem NaT;
	PythonImportCacheItem NA;
	PythonImportCacheItem isnull;
	PythonImportCacheItem ArrowDtype;
	PythonImportCacheItem BooleanDtype;
	PythonImportCacheItem UInt8Dtype;
	PythonImportCacheItem UInt16Dtype;
	PythonImportCacheItem UInt32Dtype;
	PythonImportCacheItem UInt64Dtype;
	PythonImportCacheItem Int8Dtype;
	PythonImportCacheItem Int16Dtype;
	PythonImportCacheItem Int32Dtype;
	PythonImportCacheItem Int64Dtype;
	PythonImportCacheItem Float32Dtype;
	PythonImportCacheItem Float64Dtype;

protected:
	bool IsRequired() const override final
	{
		return false;
	}
};

} // namespace CHDB
