#!/bin/bash

set -e

TARGET_ARCH=${1:-x86_64}
build_type=${2:-Release}
MY_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${MY_DIR}/../vars.sh cross-compile

# Validate architecture
if [[ "$TARGET_ARCH" != "x86_64" && "$TARGET_ARCH" != "arm64" ]]; then
    echo "Error: Invalid architecture. Use 'x86_64' or 'arm64'"
    echo "Usage: $0 [x86_64|arm64] [Release|Debug]"
    exit 1
fi

echo "Cross-compiling chdb static library for macOS ${TARGET_ARCH} on Linux..."

# Verify we're running on Linux
if [ "$(uname)" != "Linux" ]; then
    echo "Error: This script must be run on Linux"
    exit 1
fi

# Set architecture-specific variables
if [ "$TARGET_ARCH" == "x86_64" ]; then
    DARWIN_TRIPLE="x86_64-apple-darwin"
    TOOLCHAIN_FILE="cmake/darwin/toolchain-x86_64.cmake"
    BUILD_DIR_SUFFIX="darwin-x86_64"
    MACOS_MIN_VERSION="10.15"
    CPU_FEATURES="-DENABLE_AVX=0 -DENABLE_AVX2=0"
else
    # arm64
    DARWIN_TRIPLE="aarch64-apple-darwin"
    TOOLCHAIN_FILE="cmake/darwin/toolchain-aarch64.cmake"
    BUILD_DIR_SUFFIX="darwin-arm64"
    MACOS_MIN_VERSION="11.0"
    CPU_FEATURES="-DENABLE_AVX=0 -DENABLE_AVX2=0"
fi

# Download macOS SDK
SDK_PATH="${PROJ_DIR}/cmake/toolchain/${SDK_DIR}"
echo "Downloading macOS SDK to ${SDK_PATH}..."
mkdir -p "${SDK_PATH}"
cd "${SDK_PATH}"
if ! curl -L 'https://github.com/phracker/MacOSX-SDKs/releases/download/11.3/MacOSX11.0.sdk.tar.xz' | tar xJ --strip-components=1; then
    echo "Error: Failed to download macOS SDK"
    exit 1
fi
echo "macOS SDK downloaded successfully"

# Download Python headers
echo "Downloading Python headers..."
if ! bash "${DIR}/build/download_python_headers.sh"; then
    echo "Error: Failed to download Python headers"
    exit 1
fi

# Install cctools
if ! bash "${DIR}/build/install_cctools.sh" "${TARGET_ARCH}"; then
    echo "Error: Failed to install cctools"
    exit 1
fi
# Set CCTOOLS path after installation
CCTOOLS_INSTALL_DIR="${HOME}/cctools"
CCTOOLS_BIN="${CCTOOLS_INSTALL_DIR}/bin"

# Override tools with cross-compilation versions from cctools
export STRIP="llvm-strip-19"
export AR="${CCTOOLS_BIN}/${DARWIN_TRIPLE}-ar"
export NM="${CCTOOLS_BIN}/${DARWIN_TRIPLE}-nm"
export LDD="${CCTOOLS_BIN}/${DARWIN_TRIPLE}-otool -L"

echo "Using cross-compilation tools:"
echo "  STRIP: ${STRIP}"
echo "  AR: ${AR}"
echo "  NM: ${NM}"
echo "  LDD: ${LDD}"

BUILD_DIR=${PROJ_DIR}/build-${BUILD_DIR_SUFFIX}

# Set up cross-compilation tools
export CC=clang-19
export CXX=clang++-19

# macOS-specific settings
GLIBC_COMPATIBILITY="-DGLIBC_COMPATIBILITY=0"
UNWIND="-DUSE_UNWIND=0"
HDFS="-DENABLE_HDFS=0 -DENABLE_GSASL_LIBRARY=0 -DENABLE_KRB5=0"
MYSQL="-DENABLE_MYSQL=0"
ICU="-DENABLE_ICU=0"
RUST_FEATURES="-DENABLE_RUST=0"
JEMALLOC="-DENABLE_JEMALLOC=0"
LLVM="-DENABLE_EMBEDDED_COMPILER=0 -DENABLE_DWARF_PARSER=0"
CMAKE_AR_FILEPATH="${CCTOOLS_BIN}/${DARWIN_TRIPLE}-ar"
CMAKE_INSTALL_NAME_TOOL="${CCTOOLS_BIN}/${DARWIN_TRIPLE}-install_name_tool"
CMAKE_RANLIB_FILEPATH="${CCTOOLS_BIN}/${DARWIN_TRIPLE}-ranlib"
CMAKE_LINKER_NAME="${CCTOOLS_BIN}/${DARWIN_TRIPLE}-ld"

if [ ! -d $BUILD_DIR ]; then
    mkdir $BUILD_DIR
fi

cd ${BUILD_DIR}

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=${build_type} \
    -DCMAKE_AR:FILEPATH=${CMAKE_AR_FILEPATH} \
    -DCMAKE_INSTALL_NAME_TOOL=${CMAKE_INSTALL_NAME_TOOL} \
    -DCMAKE_RANLIB:FILEPATH=${CMAKE_RANLIB_FILEPATH} \
    -DLINKER_NAME=${CMAKE_LINKER_NAME} \
    -DENABLE_THINLTO=0 -DENABLE_TESTS=0 -DENABLE_CLICKHOUSE_SERVER=0 -DENABLE_CLICKHOUSE_CLIENT=0 \
    -DENABLE_CLICKHOUSE_KEEPER=0 -DENABLE_CLICKHOUSE_KEEPER_CONVERTER=0 -DENABLE_CLICKHOUSE_LOCAL=1 -DENABLE_CLICKHOUSE_SU=0 -DENABLE_CLICKHOUSE_BENCHMARK=0 \
    -DENABLE_AZURE_BLOB_STORAGE=1 -DENABLE_CLICKHOUSE_COPIER=0 -DENABLE_CLICKHOUSE_DISKS=0 -DENABLE_CLICKHOUSE_FORMAT=0 -DENABLE_CLICKHOUSE_GIT_IMPORT=0 \
    -DENABLE_AWS_S3=1 -DENABLE_HIVE=0 -DENABLE_AVRO=1 \
    -DENABLE_CLICKHOUSE_OBFUSCATOR=0 -DENABLE_CLICKHOUSE_ODBC_BRIDGE=0 -DENABLE_CLICKHOUSE_STATIC_FILES_DISK_UPLOADER=0 \
    -DENABLE_KAFKA=1 -DENABLE_LIBPQXX=1 -DENABLE_NATS=0 -DENABLE_AMQPCPP=0 -DENABLE_NURAFT=0 \
    -DENABLE_CASSANDRA=0 -DENABLE_ODBC=0 -DENABLE_NLP=0 \
    -DENABLE_LDAP=0 \
    -DENABLE_CLIENT_AI=1 \
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

cmake ${CMAKE_ARGS} -DENABLE_PYTHON=0 -DCHDB_STATIC_LIBRARY_BUILD=1 ..
ninja -d keeprsp

BINARY=${BUILD_DIR}/programs/clickhouse
rm -f ${BINARY}

cd ${BUILD_DIR}
ninja -d keeprsp -v > build.log || true

ccache -s || true

cd ${MY_DIR}

# Create static library
echo "Creating static library libchdb.a for macOS..."
python3 create_static_libchdb.py --cross-compile --build-dir=build-${BUILD_DIR_SUFFIX} --ar-cmd=${AR}
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
if [ "$TARGET_ARCH" == "x86_64" ]; then
    SYSROOT="${PROJ_DIR}/cmake/toolchain/darwin-x86_64"
else
    SYSROOT="${PROJ_DIR}/cmake/toolchain/darwin-aarch64"
fi
clang-19 chdb_example.cpp -o chdb_example \
    --target=${DARWIN_TRIPLE} \
    -isysroot ${SYSROOT} \
    -mmacosx-version-min=${MACOS_MIN_VERSION} \
    -nostdinc++ \
    -I${PROJ_DIR}/contrib/llvm-project/libcxx/include \
    -I${PROJ_DIR}/contrib/llvm-project/libcxxabi/include \
    --ld-path=${CCTOOLS_BIN}/${DARWIN_TRIPLE}-ld \
    -L. -lchdb -liconv \
    -framework CoreFoundation \
    -framework Security \
    -Wl,-map,chdb_example.map
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
python3 create_minimal_libchdb.py --ar-cmd=${AR}
if [ $? -ne 0 ]; then
    echo "Error: Failed to create minimal libchdb.a"
    exit 1
fi

# Strip the libchdb_minimal.a
if [ ${build_type} == "Debug" ]; then
    echo -e "\nDebug build, skip strip"
else
    echo -e "\nStrip the libchdb_minimal.a:"
    ${STRIP} -S libchdb_minimal.a
fi

# Copy final library to project root
echo "Copying libchdb_minimal.a to project root as libchdb.a..."
cp ${MY_DIR}/libchdb_minimal.a ${PROJ_DIR}/libchdb.a
echo "Final libchdb.a created at ${PROJ_DIR}/libchdb.a"

# Print final library size
echo "Final libchdb.a size:"
ls -lh ${PROJ_DIR}/libchdb.a
