#!/bin/bash

set -e

export USE_MUSL=1

build_type=${1:-Release}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

. ${DIR}/vars.sh

BUILD_DIR=${PROJ_DIR}/buildlib

HDFS="-DENABLE_HDFS=1 -DENABLE_GSASL_LIBRARY=1 -DENABLE_KRB5=1"
MYSQL="-DENABLE_MYSQL=1"
RUST_FEATURES="-DENABLE_RUST=0"
if [ "$(uname)" == "Linux" ]; then
    GLIBC_COMPATIBILITY="-DGLIBC_COMPATIBILITY=0"
    UNWIND="-DUSE_UNWIND=1"
    JEMALLOC="-DENABLE_JEMALLOC=0"
    PYINIT_ENTRY="-Wl,-ePyInit_${CHDB_PY_MOD}"
    ICU="-DENABLE_ICU=1"
    SED_INPLACE="sed -i"
    # only x86_64, enable AVX, enable embedded compiler
    if [ "$(uname -m)" == "x86_64" ]; then
        CPU_FEATURES="-DENABLE_AVX=1 -DENABLE_AVX2=0"
        LLVM="-DENABLE_EMBEDDED_COMPILER=1 -DENABLE_DWARF_PARSER=1"
        RUST_FEATURES="-DENABLE_RUST=1 -DENABLE_DELTA_KERNEL_RS=1"
        CORROSION_CMAKE_FILE="${PROJ_DIR}/contrib/corrosion-cmake/CMakeLists.txt"
        if [ -f "${CORROSION_CMAKE_FILE}" ]; then
            if ! grep -q 'OPENSSL_NO_DEPRECATED_3_0' "${CORROSION_CMAKE_FILE}"; then
                echo "Modifying corrosion CMakeLists.txt for Linux x86_64..."
                ${SED_INPLACE} 's/corrosion_set_env_vars(${target_name} "RUSTFLAGS=${RUSTFLAGS}")/corrosion_set_env_vars(${target_name} "RUSTFLAGS=${RUSTFLAGS} --cfg osslconf=\\\"OPENSSL_NO_DEPRECATED_3_0\\\"")/g' "${CORROSION_CMAKE_FILE}"
            else
                echo "corrosion CMakeLists.txt already modified, skipping..."
            fi
        else
            echo "Warning: corrosion CMakeLists.txt not found at ${CORROSION_CMAKE_FILE}"
        fi
    else
        CPU_FEATURES="-DENABLE_AVX=0 -DENABLE_AVX2=0 -DNO_ARMV81_OR_HIGHER=1"
        LLVM="-DENABLE_EMBEDDED_COMPILER=0 -DENABLE_DWARF_PARSER=0"
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
    -DUSE_MUSL=1 \
    -DRust_RUSTUP_INSTALL_MISSING_TARGET=ON \
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
    ${COMPILER_CACHE} \
    -DCHDB_VERSION=${CHDB_VERSION} \
    "

BINARY=${BUILD_DIR}/programs/clickhouse

# build chdb python module
py_version="3.8"
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

PYCHDB_CMD=$(echo ${PYCHDB_CMD} | sed 's/ src\/CMakeFiles\/clickhouse_malloc.dir\/Common\/stubFree.c.o//g')
if [ ! "${USING_RESPONSE_FILE}" == "" ]; then
    ${SED_INPLACE} 's/ src\/CMakeFiles\/clickhouse_malloc.dir\/Common\/stubFree.c.o//g' CMakeFiles/pychdb.rsp
fi

PYCHDB_CMD=$(echo ${PYCHDB_CMD} | sed 's|-Wl,-rpath,/[^[:space:]]*/pybind11-cmake|-Wl,-rpath,\$ORIGIN|g')

echo ${PYCHDB_CMD} > pychdb_cmd.sh

${PYCHDB_CMD}

ls -lh ${CHDB_PY_MODULE}

PYCHDB=${BUILD_DIR}/${CHDB_PY_MODULE}

if [ ${build_type} == "Debug" ]; then
    echo -e "\nDebug build, skip strip"
else
    echo -e "\nStrip the binary:"
    ${STRIP} --remove-section=.comment --remove-section=.note ${PYCHDB}
fi
echo -e "\nStripe the binary:"

echo -e "\nPYCHDB: ${PYCHDB}"
ls -lh ${PYCHDB}
echo -e "\nldd ${PYCHDB}"
${LDD} ${PYCHDB} || echo "Binary is statically linked (not a dynamic executable)"
echo -e "\nfile info of ${PYCHDB}"
file ${PYCHDB}

rm -f ${CHDB_DIR}/*.so
cp -a ${PYCHDB} ${CHDB_DIR}/${CHDB_PY_MODULE}

echo -e "\nSymbols:"
echo -e "\nPyInit in PYCHDB: ${PYCHDB}"
${NM} ${PYCHDB} | grep PyInit || true
echo -e "\nquery_stable in PYCHDB: ${PYCHDB}"
${NM} ${PYCHDB} | grep query_stable || true

echo -e "\nAfter copy:"
cd ${PROJ_DIR} && pwd

ccache -s || true

CMAKE_ARGS="${CMAKE_ARGS}" bash ${DIR}/build_pybind11.sh --all
