#!/bin/bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJ_DIR=$(dirname ${DIR})

cd ${PROJ_DIR}

export PATH=$(brew --prefix llvm@15)/bin:$PATH
export CC=$(brew --prefix llvm@15)/bin/clang
export CXX=$(brew --prefix llvm@15)/bin/clang++

python3 -m pip install -U pybind11 wheel build tox

rm -rf ${PROJ_DIR}/buildlib
bash ${PROJ_DIR}/chdb/build_libchdb.sh