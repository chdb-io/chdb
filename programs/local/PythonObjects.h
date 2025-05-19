#pragma once

#include "PybindWrapper.h"

#include <base/types.h>

namespace CHDB {

struct PyDictionary {
public:
	PyDictionary(py::object dict);
	py::object keys;
	py::object values;
	size_t len;

public:
	py::handle operator[](const py::object & obj) const
	{
		return PyDict_GetItem(dict.ptr(), obj.ptr());
	}

public:
	String ToString() const
	{
		return String(py::str(dict));
	}

private:
	py::object dict;
};

} // namespace CHDB
