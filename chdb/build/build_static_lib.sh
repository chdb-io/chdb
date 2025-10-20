#!/bin/bash

set -e

build_type=${1:-Release}

MY_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

. ${MY_DIR}/../vars.sh

BUILD_DIR=${PROJ_DIR}/build-static-lib

HDFS="-DENABLE_HDFS=1 -DENABLE_GSASL_LIBRARY=1 -DENABLE_KRB5=1"
MYSQL="-DENABLE_MYSQL=1"
RUST_FEATURES="-DENABLE_RUST=0"
JEMALLOC="-DENABLE_JEMALLOC=0"
LLVM="-DENABLE_EMBEDDED_COMPILER=0 -DENABLE_DWARF_PARSER=0"
if [ "$(uname)" == "Darwin" ]; then
    export CXX=$(brew --prefix llvm@19)/bin/clang++
    export CC=$(brew --prefix llvm@19)/bin/clang
    export PATH=$(brew --prefix llvm@19)/bin:$PATH
    GLIBC_COMPATIBILITY="-DGLIBC_COMPATIBILITY=0"
    UNWIND="-DUSE_UNWIND=0"
    HDFS="-DENABLE_HDFS=0 -DENABLE_GSASL_LIBRARY=0 -DENABLE_KRB5=0"
    MYSQL="-DENABLE_MYSQL=0"
    ICU="-DENABLE_ICU=0"
    CPU_FEATURES="-DENABLE_AVX=0 -DENABLE_AVX2=0"
elif [ "$(uname)" == "Linux" ]; then
    GLIBC_COMPATIBILITY="-DGLIBC_COMPATIBILITY=1"
    UNWIND="-DUSE_UNWIND=1"
    ICU="-DENABLE_ICU=1"
    if [ "$(uname -m)" == "x86_64" ]; then
        CPU_FEATURES="-DENABLE_AVX=1 -DENABLE_AVX2=0"
    else
        CPU_FEATURES="-DENABLE_AVX=0 -DENABLE_AVX2=0 -DNO_ARMV81_OR_HIGHER=1"
    fi
else
    echo "OS not supported"
    exit 1
fi

if [ ! -d $BUILD_DIR ]; then
    mkdir $BUILD_DIR
fi

cd ${BUILD_DIR}
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=${build_type} -DENABLE_THINLTO=0 -DENABLE_TESTS=0 -DENABLE_CLICKHOUSE_SERVER=0 -DENABLE_CLICKHOUSE_CLIENT=0 \
    -DENABLE_CLICKHOUSE_KEEPER=0 -DENABLE_CLICKHOUSE_KEEPER_CONVERTER=0 -DENABLE_CLICKHOUSE_LOCAL=1 -DENABLE_CLICKHOUSE_SU=0 -DENABLE_CLICKHOUSE_BENCHMARK=0 \
    -DENABLE_AZURE_BLOB_STORAGE=1 -DENABLE_CLICKHOUSE_COPIER=0 -DENABLE_CLICKHOUSE_DISKS=0 -DENABLE_CLICKHOUSE_FORMAT=0 -DENABLE_CLICKHOUSE_GIT_IMPORT=0 \
    -DENABLE_AWS_S3=1 -DENABLE_HIVE=0 -DENABLE_AVRO=1 \
    -DENABLE_CLICKHOUSE_OBFUSCATOR=0 -DENABLE_CLICKHOUSE_ODBC_BRIDGE=0 -DENABLE_CLICKHOUSE_STATIC_FILES_DISK_UPLOADER=0 \
    -DENABLE_KAFKA=1 -DENABLE_LIBPQXX=1 -DENABLE_NATS=0 -DENABLE_AMQPCPP=0 -DENABLE_NURAFT=0 \
    -DENABLE_CASSANDRA=0 -DENABLE_ODBC=0 -DENABLE_NLP=0 \
    -DENABLE_LDAP=0 \
    ${MYSQL} \
    ${HDFS} \
    -DENABLE_LIBRARIES=0 ${RUST_FEATURES} \
    ${GLIBC_COMPATIBILITY} \
    -DENABLE_UTILS=0 ${LLVM} ${UNWIND} \
    ${ICU} -DENABLE_UTF8PROC=1 ${JEMALLOC} \
    -DENABLE_PARQUET=1 -DENABLE_ROCKSDB=1 -DENABLE_SQLITE=1 -DENABLE_VECTORSCAN=1 \
    -DENABLE_PROTOBUF=1 -DENABLE_THRIFT=1 -DENABLE_MSGPACK=1 \
    -DENABLE_BROTLI=1 -DENABLE_H3=1 -DENABLE_CURL=1 \
    -DENABLE_CLICKHOUSE_ALL=0 -DUSE_STATIC_LIBRARIES=1 -DSPLIT_SHARED_LIBRARIES=0 \
    -DENABLE_SIMDJSON=1 -DENABLE_RAPIDJSON=1 \
    ${CPU_FEATURES} \
    -DENABLE_AVX512=0 -DENABLE_AVX512_VBMI=0 \
    -DENABLE_LIBFIU=1 \
    -DCHDB_VERSION=${CHDB_VERSION} \
    "

cmake ${CMAKE_ARGS} -DENABLE_PYTHON=0 -DCHDB_STATIC_LIBRARY_BUILD=1 ..
ninja -d keeprsp -v 2>&1 | tee build.log

ccache -s || true

cd ${MY_DIR}

# Create static library
echo "Creating static library libchdb.a..."
python3 create_static_libchdb.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to create static library"
    exit 1
fi

# Prepare cpp-example directory and copy header file
echo "Preparing cpp-example directory..."
cd ${MY_DIR}/cpp-example
cp ${PROJ_DIR}/programs/local/chdb.h .
cp ${MY_DIR}/libchdb.a .
echo "Copied chdb.h and libchdb.a to cpp-example directory"

# Compile example program
echo "Compiling chdb_example.cpp..."
if [ "$(uname)" == "Darwin" ]; then
    CLANG_CMD="$(brew --prefix llvm@19)/bin/clang"
    ${CLANG_CMD} chdb_example.cpp -o chdb_example -mmacosx-version-min=10.15 -L. -lchdb -liconv -framework CoreFoundation -Wl,-map,chdb_example.map
else
    CLANG_CMD="clang"
    ${CLANG_CMD} chdb_example.cpp -o chdb_example -L. -lchdb -lpthread -ldl -lc -lm -lrt -Wl,-Map=chdb_example.map -Wl,--allow-multiple-definition
fi
if [ $? -ne 0 ]; then
    echo "Error: Failed to compile chdb_example.cpp"
    exit 1
fi

# Copy map file to parent directory for analysis
echo "Copying chdb_example.map to parent directory..."
cp chdb_example.map ${MY_DIR}/
cd ${MY_DIR}

# Analyze map file to extract chdb objects
echo "Analyzing map file to extract chdb objects..."
python3 extract_chdb_objects.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to analyze map file"
    exit 1
fi

# Create minimal libchdb.a based on extracted objects
echo "Creating minimal libchdb.a..."
python3 create_minimal_libchdb.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to create minimal libchdb.a"
    exit 1
fi

# Strip the libchdb_minimal.a
if [ ${build_type} == "Debug" ]; then
    echo -e "\nDebug build, skip strip"
else
    echo -e "\nStrip the libchdb_minimal.a:"
    ${STRIP} --strip-debug --remove-section=.comment --remove-section=.note libchdb_minimal.a
fi


# Test with Go example
echo "Preparing go-example directory..."
cd ${MY_DIR}/go-example
cp ${MY_DIR}/libchdb_minimal.a ./libchdb.a
cp ${PROJ_DIR}/programs/local/chdb.h .
echo "Copied libchdb_minimal.a as libchdb.a and chdb.h to go-example directory"

# Run Go test
echo "Running Go test..."
# export CGO_CFLAGS_ALLOW=".*"
# export CGO_LDFLAGS_ALLOW=".*"
go run .
if [ $? -ne 0 ]; then
    echo "Error: Go test failed"
    exit 1
fi

echo "Go test completed successfully!"

# Copy final library to project root
echo "Copying libchdb_minimal.a to project root as libchdb.a..."
cp ${MY_DIR}/libchdb_minimal.a ${PROJ_DIR}/libchdb.a
echo "Final libchdb.a created at ${PROJ_DIR}/libchdb.a"

# Print final library size
echo "Final libchdb.a size:"
ls -lh ${PROJ_DIR}/libchdb.a
