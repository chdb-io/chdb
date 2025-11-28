#!/bin/bash

set -e

# Cross-compile chdb static library for macOS (x86_64 or arm64) on Linux
# Usage: ./build_static_lib_mac_on_linux.sh [x86_64|arm64] [Release|Debug]

# Parse arguments
TARGET_ARCH=${1:-x86_64}
build_type=${2:-Release}

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

# Verify required environment variables
if [ -z "${CCTOOLS:-}" ]; then
    echo "Error: CCTOOLS environment variable not set. Please set it to the cctools bin directory."
    echo "Example: export CCTOOLS=/path/to/cctools"
    exit 1
fi

# Set architecture-specific variables
if [ "$TARGET_ARCH" == "x86_64" ]; then
    DARWIN_TRIPLE="x86_64-apple-darwin"
    CMAKE_ARCH="x86_64"
    TOOLCHAIN_FILE="cmake/darwin/toolchain-x86_64.cmake"
    BUILD_DIR_SUFFIX="static-lib-darwin-x86_64"
    OUTPUT_SUFFIX="darwin-x86_64"
    EXAMPLE_DIR_SUFFIX="darwin-x86_64"
    MACOS_MIN_VERSION="10.15"
    # x86_64 specific: disable AVX for compatibility
    CPU_FEATURES="-DENABLE_AVX=0 -DENABLE_AVX2=0"
else
    # arm64
    DARWIN_TRIPLE="aarch64-apple-darwin"
    CMAKE_ARCH="aarch64"
    TOOLCHAIN_FILE="cmake/darwin/toolchain-aarch64.cmake"
    BUILD_DIR_SUFFIX="static-lib-darwin-arm64"
    OUTPUT_SUFFIX="darwin-arm64"
    EXAMPLE_DIR_SUFFIX="darwin-arm64"
    MACOS_MIN_VERSION="11.0"
    # ARM64 specific: disable x86 features
    CPU_FEATURES="-DENABLE_AVX=0 -DENABLE_AVX2=0 -DNO_ARMV81_OR_HIGHER=0"
fi

# Check if cctools exist for this architecture
if [ ! -f "${CCTOOLS}/bin/${DARWIN_TRIPLE}-ar" ]; then
    echo "Error: cctools not found at ${CCTOOLS}/bin/${DARWIN_TRIPLE}-ar"
    echo "Tip: You may need to rebuild cctools with support for ${TARGET_ARCH}"
    exit 1
fi

MY_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

. ${MY_DIR}/../vars.sh

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
    -DCMAKE_AR:FILEPATH=${CCTOOLS}/bin/${DARWIN_TRIPLE}-ar \
    -DCMAKE_INSTALL_NAME_TOOL=${CCTOOLS}/bin/${DARWIN_TRIPLE}-install_name_tool \
    -DCMAKE_RANLIB:FILEPATH=${CCTOOLS}/bin/${DARWIN_TRIPLE}-ranlib \
    -DCMAKE_LINKER:FILEPATH=${CCTOOLS}/bin/${DARWIN_TRIPLE}-ld \
    -DLINKER_NAME=${CCTOOLS}/bin/${DARWIN_TRIPLE}-ld \
    -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
    "

echo "Running cmake configuration..."
cmake ${CMAKE_ARGS} -DENABLE_PYTHON=0 -DCHDB_STATIC_LIBRARY_BUILD=1 ..

echo "Building with ninja..."
ninja -d keeprsp

BINARY=${BUILD_DIR}/programs/clickhouse
rm -f ${BINARY}

cd ${BUILD_DIR}
ninja -d keeprsp -v > build.log || true

ccache -s || true

cd ${MY_DIR}

# Create static library
echo "Creating static library libchdb.a for macOS..."
python3 create_static_libchdb.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to create static library"
    exit 1
fi

# Prepare cpp-example directory and copy header file
echo "Preparing cpp-example-${EXAMPLE_DIR_SUFFIX} directory..."
if [ ! -d ${MY_DIR}/cpp-example-${EXAMPLE_DIR_SUFFIX} ]; then
    cp -r ${MY_DIR}/cpp-example ${MY_DIR}/cpp-example-${EXAMPLE_DIR_SUFFIX}
fi

cd ${MY_DIR}/cpp-example-${EXAMPLE_DIR_SUFFIX}
cp ${PROJ_DIR}/programs/local/chdb.h .
cp ${MY_DIR}/libchdb.a .
echo "Copied chdb.h and libchdb.a to cpp-example-${EXAMPLE_DIR_SUFFIX} directory"

echo "Note: Skipping C++ example compilation for cross-compilation."
echo "The example can be compiled on the target macOS ${TARGET_ARCH} system with:"
echo "  clang chdb_example.cpp -o chdb_example -mmacosx-version-min=${MACOS_MIN_VERSION} -L. -lchdb -liconv -framework CoreFoundation"

# For cross-compilation, we'll create a minimal analysis without running the compiled binary
echo "Creating analysis files for cross-compilation..."

# Copy map file analysis tools but don't run them (since we can't execute macOS binaries on Linux)
echo "Note: Skipping map file analysis for cross-compilation."
echo "Run the following on macOS ${TARGET_ARCH} to create minimal library:"
echo "  cd ${MY_DIR}/cpp-example-${EXAMPLE_DIR_SUFFIX}"
echo "  clang chdb_example.cpp -o chdb_example -mmacosx-version-min=${MACOS_MIN_VERSION} -L. -lchdb -liconv -framework CoreFoundation -Wl,-map,chdb_example.map"
echo "  cd ${MY_DIR}"
echo "  python3 extract_chdb_objects.py --map-file=cpp-example-${EXAMPLE_DIR_SUFFIX}/chdb_example.map"
echo "  python3 create_minimal_libchdb.py"

# For now, we'll use the full libchdb.a as the final output
echo "Using full libchdb.a for cross-compilation (minimal version requires macOS execution)"

# Strip the libchdb.a if not debug build
if [ ${build_type} == "Debug" ]; then
    echo -e "\nDebug build, skip strip"
else
    echo -e "\nStrip the libchdb.a:"
    # Use macOS-compatible strip command via cctools
    if [ -f "${CCTOOLS}/bin/${DARWIN_TRIPLE}-strip" ]; then
        ${CCTOOLS}/bin/${DARWIN_TRIPLE}-strip -S -x libchdb.a
    else
        echo "Warning: macOS strip not found, skipping strip step"
    fi
fi

echo "Note: Skipping Go test for cross-compilation."

# Copy final library to project root
OUTPUT_NAME="libchdb-${OUTPUT_SUFFIX}.a"
echo "Copying libchdb.a to project root as ${OUTPUT_NAME}..."
cp ${MY_DIR}/libchdb.a ${PROJ_DIR}/${OUTPUT_NAME}
echo "Final ${OUTPUT_NAME} created at ${PROJ_DIR}/${OUTPUT_NAME}"

# Print final library size
echo "Final ${OUTPUT_NAME} size:"
ls -lh ${PROJ_DIR}/${OUTPUT_NAME}

echo "Cross-compilation for macOS ${TARGET_ARCH} completed successfully!"
echo "Generated files:"
echo "  - ${PROJ_DIR}/${OUTPUT_NAME}"
echo "  - ${MY_DIR}/cpp-example-${EXAMPLE_DIR_SUFFIX}/ (for testing on macOS ${TARGET_ARCH})"