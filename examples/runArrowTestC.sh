#!/bin/bash

set -e

# CFLAGS="-g -O0 -DDEBUG"
CFLAGS="-std=c99"

# check current os type, and make ldd command
if [ "$(uname)" == "Darwin" ]; then
    LDD="otool -L"
    LIB_PATH="DYLD_LIBRARY_PATH"
elif [ "$(uname)" == "Linux" ]; then
    LDD="ldd"
    LIB_PATH="LD_LIBRARY_PATH"
else
    echo "OS not supported"
    exit 1
fi

# cd to the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

echo "Compile and link chdbArrowTest (C version)"
clang $CFLAGS chdbArrowTest.c -o chdbArrowTestC \
    -I../programs/local/ \
    -I../contrib/arrow/cpp/src \
    -I../contrib/arrow-cmake/cpp/src \
    -I../src \
    -L../ -lchdb

export ${LIB_PATH}=..
${LDD} chdbArrowTestC

echo "Run Arrow API tests (C version):"
./chdbArrowTestC
