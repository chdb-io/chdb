#!/bin/bash

set -e

# Cross-compile chdb for macOS (x86_64 or arm64) on Linux
# Usage: ./build_mac_on_linux_universal.sh [x86_64|arm64] [Release|Debug]

# Parse arguments
TARGET_ARCH=${1:-x86_64}
build_type=${2:-Release}

# Validate architecture
if [[ "$TARGET_ARCH" != "x86_64" && "$TARGET_ARCH" != "arm64" ]]; then
    echo "Error: Invalid architecture. Use 'x86_64' or 'arm64'"
    echo "Usage: $0 [x86_64|arm64] [Release|Debug]"
    exit 1
fi

echo "Cross-compiling chdb for macOS ${TARGET_ARCH} on Linux..."

# Verify we're running on Linux
if [ "$(uname)" != "Linux" ]; then
    echo "Error: This script must be run on Linux"
    exit 1
fi

# Set architecture-specific variables first
if [ "$TARGET_ARCH" == "x86_64" ]; then
    DARWIN_TRIPLE="x86_64-apple-darwin"
    CMAKE_ARCH="x86_64"
    TOOLCHAIN_FILE="cmake/darwin/toolchain-x86_64.cmake"
    BUILD_DIR_SUFFIX="darwin-x86_64"
    CPU_FEATURES="-DENABLE_AVX=0 -DENABLE_AVX2=0"
else
    # arm64
    DARWIN_TRIPLE="aarch64-apple-darwin"
    CMAKE_ARCH="aarch64"
    TOOLCHAIN_FILE="cmake/darwin/toolchain-aarch64.cmake"
    BUILD_DIR_SUFFIX="darwin-arm64"
    CPU_FEATURES="-DENABLE_AVX=0 -DENABLE_AVX2=0 -DNO_ARMV81_OR_HIGHER=0"
fi

# Install cctools if not already installed
CCTOOLS_INSTALL_DIR="${HOME}/cctools"
CCTOOLS_BIN="${CCTOOLS_INSTALL_DIR}/bin"

if [ -z "${CCTOOLS:-}" ]; then
    echo "CCTOOLS environment variable not set, checking for installation..."

    # Check if cctools is already installed
    if [ -f "${CCTOOLS_BIN}/${DARWIN_TRIPLE}-ld" ]; then
        echo "Found existing cctools installation at ${CCTOOLS_INSTALL_DIR}"
        export CCTOOLS="${CCTOOLS_BIN}"
    else
        echo "cctools not found, installing..."

        mkdir ~/cctools
        export CCTOOLS=$(cd ~/cctools && pwd)
        cd ${CCTOOLS}

        git clone https://github.com/tpoechtrager/apple-libtapi.git
        cd apple-libtapi
        git checkout 15dfc2a8c9a2a89d06ff227560a69f5265b692f9
        INSTALLPREFIX=${CCTOOLS} ./build.sh
        ./install.sh
        cd ..

        git clone https://github.com/chdb-io/cctools-port.git
        cd cctools-port/cctools

        # Set cctools target based on architecture
        if [ "$TARGET_ARCH" == "x86_64" ]; then
            CCTOOLS_TARGET="x86_64-apple-darwin"
        else
            CCTOOLS_TARGET="aarch64-apple-darwin"
        fi

        ./configure --prefix=$(readlink -f ${CCTOOLS}) --with-libtapi=$(readlink -f ${CCTOOLS}) --target=${CCTOOLS_TARGET}
        make install
    fi
else
    echo "Using CCTOOLS from environment variable: ${CCTOOLS}"
fi

# Verify cctools installation
if [ ! -f "${CCTOOLS}/${DARWIN_TRIPLE}-ld" ]; then
    echo "Error: cctools linker not found at ${CCTOOLS}/${DARWIN_TRIPLE}-ld"
    echo "Please verify cctools installation or set CCTOOLS environment variable correctly"
    exit 1
fi

echo "cctools verified: ${CCTOOLS}/${DARWIN_TRIPLE}-ld"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

. ${DIR}/vars.sh

BUILD_DIR=${PROJ_DIR}/build-${BUILD_DIR_SUFFIX}

# Set up cross-compilation tools
export CC=clang-19
export CXX=clang++-19

# macOS-specific settings
GLIBC_COMPATIBILITY="-DGLIBC_COMPATIBILITY=0"
UNWIND="-DUSE_UNWIND=0"
JEMALLOC="-DENABLE_JEMALLOC=0"
PYINIT_ENTRY="-Wl,-exported_symbol,_PyInit_${CHDB_PY_MOD}"
HDFS="-DENABLE_HDFS=0 -DENABLE_GSASL_LIBRARY=0 -DENABLE_KRB5=0"
MYSQL="-DENABLE_MYSQL=0"
ICU="-DENABLE_ICU=0"
SED_INPLACE="sed -i"
RUST_FEATURES="-DENABLE_RUST=0"

# Disable embedded compiler for cross-compilation
LLVM="-DENABLE_EMBEDDED_COMPILER=0 -DENABLE_DWARF_PARSER=0"

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
    -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
    "

LIBCHDB_SO="libchdb.so"

# Build libchdb.so
echo "Running cmake configuration..."
cmake ${CMAKE_ARGS} -DENABLE_PYTHON=0 ..

echo "Building with ninja..."
ninja -d keeprsp

BINARY=${BUILD_DIR}/programs/clickhouse
echo -e "\nBINARY: ${BINARY}"
ls -lh ${BINARY}
echo -e "\nfile info of ${BINARY}"
file ${BINARY}
rm -f ${BINARY}

cd ${BUILD_DIR}
ninja -d keeprsp -v > build.log || true
USING_RESPONSE_FILE=$(grep -m 1 'clang++.*-o programs/clickhouse .*' build.log | grep '@CMakeFiles/clickhouse.rsp' || true)

if [ ! "${USING_RESPONSE_FILE}" == "" ]; then
    if [ -f CMakeFiles/clickhouse.rsp ]; then
        cp -a CMakeFiles/clickhouse.rsp CMakeFiles/libchdb.rsp
    else
        echo "CMakeFiles/clickhouse.rsp not found"
        exit 1
    fi
fi

LIBCHDB_CMD=$(grep -m 1 'clang++.*-o programs/clickhouse .*' build.log \
    | sed "s/-o programs\/clickhouse/-fPIC -shared -o ${LIBCHDB_SO}/" \
    | sed 's/^[^&]*&& //' | sed 's/&&.*//' \
    | sed 's/ -Wl,-undefined,error/ -Wl,-undefined,dynamic_lookup/g' \
    | sed 's/ -Xlinker --no-undefined//g' \
    | sed 's/@CMakeFiles\/clickhouse.rsp/@CMakeFiles\/libchdb.rsp/g' \
     )

# Generate the command to generate libchdb.so
LIBCHDB_CMD=$(echo ${LIBCHDB_CMD} | sed 's/ '${CHDB_PY_MODULE}'/ '${LIBCHDB_SO}'/g')

if [ ! "${USING_RESPONSE_FILE}" == "" ]; then
    ${SED_INPLACE} 's/ '${CHDB_PY_MODULE}'/ '${LIBCHDB_SO}'/g' CMakeFiles/libchdb.rsp
fi

# For macOS, replace PyInit entry point with exported symbols for libchdb
LIBCHDB_CMD=$(echo ${LIBCHDB_CMD} | sed 's/ '${PYINIT_ENTRY}'/ -Wl,-exported_symbol,_query_stable -Wl,-exported_symbol,_free_result -Wl,-exported_symbol,_query_stable_v2 -Wl,-exported_symbol,_free_result_v2/g')

LIBCHDB_CMD=$(echo ${LIBCHDB_CMD} | sed 's/@CMakeFiles\/clickhouse.rsp/@CMakeFiles\/libchdb.rsp/g')

# Save the command to a file for debug
echo ${LIBCHDB_CMD} > libchdb_cmd.sh

# Build libchdb.so
echo "Building libchdb.so..."
${LIBCHDB_CMD}

LIBCHDB_DIR=${BUILD_DIR}/
LIBCHDB=${LIBCHDB_DIR}/${LIBCHDB_SO}
ls -lh ${LIBCHDB}

# Build chdb python module
py_version="3.9"
current_py_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$current_py_version" != "$py_version" ]; then
    echo "Error: Current Python version is $current_py_version, but required version is $py_version"
    echo "Please switch to Python $py_version using: pyenv shell $py_version"
    exit 1
fi
cmake ${CMAKE_ARGS} -DENABLE_PYTHON=1 -DPYBIND11_NONLIMITEDAPI_PYTHON_HEADERS_VERSION=${py_version} ..
ninja -d keeprsp || true

# Delete the binary and run ninja -v again to capture the command
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

# Extract the command to generate CHDB_PY_MODULE
PYCHDB_CMD=$(grep -m 1 'clang++.*-o programs/clickhouse .*' build.log \
    | sed "s/-o programs\/clickhouse/-fPIC -Wl,-undefined,dynamic_lookup -shared ${PYINIT_ENTRY} -o ${CHDB_PY_MODULE}/" \
    | sed 's/^[^&]*&& //' | sed 's/&&.*//' \
    | sed 's/ -Wl,-undefined,error/ -Wl,-undefined,dynamic_lookup/g' \
    | sed 's/ -Xlinker --no-undefined//g' \
    | sed 's/@CMakeFiles\/clickhouse.rsp/@CMakeFiles\/pychdb.rsp/g' \
     )

# For macOS, set rpath
PYCHDB_CMD=$(echo ${PYCHDB_CMD} | sed 's|-Wl,-rpath,/[^[:space:]]*/pybind11-cmake|-Wl,-rpath,@loader_path|g')

# Save the command to a file for debug
echo ${PYCHDB_CMD} > pychdb_cmd.sh

echo "Building Python module..."
${PYCHDB_CMD}

ls -lh ${CHDB_PY_MODULE}

## Check all the so files
LIBCHDB_DIR=${BUILD_DIR}/

PYCHDB=${LIBCHDB_DIR}/${CHDB_PY_MODULE}
LIBCHDB=${LIBCHDB_DIR}/${LIBCHDB_SO}

if [ ${build_type} == "Debug" ]; then
    echo -e "\nDebug build, skip strip"
else
    echo -e "\nStrip the binary:"
    ${STRIP} --remove-section=.comment --remove-section=.note ${PYCHDB}
    ${STRIP} --remove-section=.comment --remove-section=.note ${LIBCHDB}
fi

echo -e "\nPYCHDB: ${PYCHDB}"
ls -lh ${PYCHDB}
echo -e "\nLIBCHDB: ${LIBCHDB}"
ls -lh ${LIBCHDB}
echo -e "\nfile info of ${PYCHDB}"
file ${PYCHDB}
echo -e "\nfile info of ${LIBCHDB}"
file ${LIBCHDB}

rm -f ${CHDB_DIR}/*.so
cp -a ${PYCHDB} ${CHDB_DIR}/${CHDB_PY_MODULE}
cp -a ${LIBCHDB} ${PROJ_DIR}/${LIBCHDB_SO}

echo -e "\nSymbols:"
echo -e "\nPyInit in PYCHDB: ${PYCHDB}"
${NM} ${PYCHDB} | grep PyInit || true
echo -e "\nPyInit in LIBCHDB: ${LIBCHDB}"
${NM} ${LIBCHDB} | grep PyInit || echo "PyInit not found in ${LIBCHDB}, it's OK"
echo -e "\nquery_stable in PYCHDB: ${PYCHDB}"
${NM} ${PYCHDB} | grep query_stable || true
echo -e "\nquery_stable in LIBCHDB: ${LIBCHDB}"
${NM} ${LIBCHDB} | grep query_stable || true

echo -e "\nAfter copy:"
cd ${PROJ_DIR} && pwd

ccache -s || true

# Skip pybind11 libraries build for cross-compilation
echo "Skipping pybind11 libraries build for cross-compilation"
echo "These should be built separately on the target macOS system using:"
echo "  CMAKE_ARGS=\"\${CMAKE_ARGS}\" bash \${DIR}/build_pybind11.sh --all"

echo -e "\nCross-compilation for macOS ${TARGET_ARCH} completed successfully!"
echo -e "Generated files:"
echo -e "  - ${PROJ_DIR}/${LIBCHDB_SO}"
echo -e "  - ${CHDB_DIR}/${CHDB_PY_MODULE}"
echo -e "\nBuild directory: ${BUILD_DIR}"
