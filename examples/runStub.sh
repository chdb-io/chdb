#!/bin/bash

echo "Compile and link"
clang chdbStub.c -o chdbStub -lc -I../programs/local/ -L../chdb/ -lchdb

ldd chdbStub

echo "Run it:"
LD_LIBRARY_PATH=../chdb ./chdbStub
