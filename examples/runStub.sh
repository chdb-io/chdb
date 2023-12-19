#!/bin/bash

set -e
# cd to the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

echo "Compile and link"
clang chdbStub.c -o chdbStub -I../programs/local/ -L../chdb/ -lchdb

ldd chdbStub

echo "Run it:"
LD_LIBRARY_PATH=../chdb ./chdbStub
