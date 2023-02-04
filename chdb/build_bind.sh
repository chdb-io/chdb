#!/bin/bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

. ${DIR}/vars.sh

cd ${CHDB_DIR}

CHDB_PY_MODULE="_chdb$(python3-config --extension-suffix)"

# compile the pybind module, MUST use "./libchdb.so" instead of ${LIBCHDB} or "libchdb.so"
clang++ -O3 -Wall -shared -std=c++17 -fPIC -I../ -I../base -I../src -I../programs/local/ \
    $(python3 -m pybind11 --includes) chdb.cpp \
    -Wl,--exclude-libs,ALL -stdlib=libstdc++ -static-libstdc++ -static-libgcc \
    ./libchdb.so -o ${CHDB_PY_MODULE} 
