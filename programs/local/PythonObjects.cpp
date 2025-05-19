#include "PythonObjects.h"

namespace CHDB {

PyDictionary::PyDictionary(py::object dict) {
	keys = py::list(dict.attr("keys")());
	values = py::list(dict.attr("values")());
	len = py::len(keys);
	this->dict = std::move(dict);
}

} // namespace CHDB
