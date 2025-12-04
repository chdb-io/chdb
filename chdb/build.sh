#!/bin/bash

set -e

# default to build Release
build_type=${1:-RelWithDebInfo}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

. ${DIR}/vars.sh

BUILD_DIR=${PROJ_DIR}/buildlib

HDFS="-DENABLE_HDFS=1 -DENABLE_GSASL_LIBRARY=1 -DENABLE_KRB5=1"
MYSQL="-DENABLE_MYSQL=1"
RUST_FEATURES="-DENABLE_RUST=0"
export CXX=$(brew --prefix llvm@19)/bin/clang++
export CC=$(brew --prefix llvm@19)/bin/clang
export PATH=$(brew --prefix llvm@19)/bin:$PATH
GLIBC_COMPATIBILITY="-DGLIBC_COMPATIBILITY=0"
UNWIND="-DUSE_UNWIND=0"
JEMALLOC="-DENABLE_JEMALLOC=0"
PYINIT_ENTRY="-Wl,-exported_symbol,_PyInit_${CHDB_PY_MOD}"
HDFS="-DENABLE_HDFS=0 -DENABLE_GSASL_LIBRARY=0 -DENABLE_KRB5=0"
MYSQL="-DENABLE_MYSQL=0"
ICU="-DENABLE_ICU=0"
SED_INPLACE="sed -i ''"
# if Darwin ARM64 (M1, M2), disable AVX
if [ "$(uname -m)" == "arm64" ]; then
    CPU_FEATURES="-DENABLE_AVX=0 -DENABLE_AVX2=0"
    LLVM="-DENABLE_EMBEDDED_COMPILER=0 -DENABLE_DWARF_PARSER=0"
else
    LLVM="-DENABLE_EMBEDDED_COMPILER=0 -DENABLE_DWARF_PARSER=0"
    # disable AVX on Darwin for macos11
    if [ "$(sw_vers -productVersion | cut -d. -f1)" -le 11 ]; then
        CPU_FEATURES="-DENABLE_AVX=0 -DENABLE_AVX2=0"
    else
        CPU_FEATURES="-DENABLE_AVX=0 -DENABLE_AVX2=0"
    fi
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

BINARY=${BUILD_DIR}/programs/clickhouse

# build chdb python module
py_version="3.8"
# check current os type and architecture for py_version
if [ "$(uname)" == "Darwin" ] && [ "$(uname -m)" == "x86_64" ]; then
    py_version="3.9"
fi
current_py_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$current_py_version" != "$py_version" ]; then
    echo "Error: Current Python version is $current_py_version, but required version is $py_version"
    echo "Please switch to Python $py_version using: pyenv shell $py_version"
    exit 1
fi
echo "Using Python version: $current_py_version"

cmake ${CMAKE_ARGS} -DENABLE_PYTHON=1 -DPYBIND11_NONLIMITEDAPI_PYTHON_HEADERS_VERSION=${py_version} ..
ninja -d keeprsp || true

# del the binary and run ninja -v again to capture the command, then modify it to generate CHDB_PY_MODULE
/bin/rm -f ${BINARY}
cd ${BUILD_DIR}
ninja -d keeprsp -v > build.log || true

USING_RESPONSE_FILE=$(grep -m 1 'clang++.*-o programs/clickhouse .*' build.log | grep '@CMakeFiles/clickhouse.rsp' || true)

if [ ! "${USING_RESPONSE_FILE}" == "" ]; then
    if [ -f CMakeFiles/clickhouse.rsp ]; then
        cp -a CMakeFiles/clickhouse.rsp CMakeFiles/pychdb.rsp
    else
        echo "CMakeFiles/clickhouse.rsp not found"
        exit 1
    fi
fi

# extract the command to generate CHDB_PY_MODULE
PYCHDB_CMD=$(grep -m 1 'clang++.*-o programs/clickhouse .*' build.log \
    | sed "s/-o programs\/clickhouse/-fPIC -Wl,-undefined,dynamic_lookup -shared ${PYINIT_ENTRY} -o ${CHDB_PY_MODULE}/" \
    | sed 's/^[^&]*&& //' | sed 's/&&.*//' \
    | sed 's/ -Wl,-undefined,error/ -Wl,-undefined,dynamic_lookup/g' \
    | sed 's/ -Xlinker --no-undefined//g' \
    | sed 's/@CMakeFiles\/clickhouse.rsp/@CMakeFiles\/pychdb.rsp/g' \
     )

PYCHDB_CMD=$(echo ${PYCHDB_CMD} | sed 's|-Wl,-rpath,/[^[:space:]]*/pybind11-cmake|-Wl,-rpath,@loader_path|g')

# save the command to a file for debug
echo ${PYCHDB_CMD} > pychdb_cmd.sh

# Clean up to free disk space before linking
echo "Cleaning up to free disk space..."
rm -f ${BUILD_DIR}/programs/clickhouse

${PYCHDB_CMD}

ls -lh ${CHDB_PY_MODULE}

PYCHDB=${BUILD_DIR}/${CHDB_PY_MODULE}

echo -e "\nPYCHDB: ${PYCHDB}"
ls -lh ${PYCHDB}

rm -f ${CHDB_DIR}/*.so
mv ${PYCHDB} ${CHDB_DIR}/${CHDB_PY_MODULE}

echo -e "\nAfter copy:"
cd ${PROJ_DIR} && pwd

ccache -s || true

CMAKE_ARGS="${CMAKE_ARGS}" bash ${DIR}/build_pybind11.sh --version=3.8
CMAKE_ARGS="${CMAKE_ARGS}" bash ${DIR}/build_pybind11.sh --version=3.9

# rm -rf ${PROJ_DIR}/contrib
# df -h
