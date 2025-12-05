#!/bin/bash

set -e

# default to build Release
build_type=${1:-Release}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

. ${DIR}/vars.sh

BUILD_DIR=${PROJ_DIR}/buildlib

HDFS="-DENABLE_HDFS=1 -DENABLE_GSASL_LIBRARY=1 -DENABLE_KRB5=1"
MYSQL="-DENABLE_MYSQL=1"
RUST_FEATURES="-DENABLE_RUST=0"
# check current os type
if [ "$(uname)" == "Darwin" ]; then
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
            # for M1, M2 using x86_64 emulation, we need to disable AVX and AVX2
            CPU_FEATURES="-DENABLE_AVX=0 -DENABLE_AVX2=0"
            # # If target macos version is 12, we need to test if support AVX2, 
            # # because some Mac Pro Late 2013 (MacPro6,1) support AVX but not AVX2
            # # just test it on the github action, hope you don't using Mac Pro Late 2013.
            # # https://everymac.com/mac-answers/macos-12-monterey-faq/macos-monterey-macos-12-compatbility-list-system-requirements.html
            # if [ "$(sysctl -n machdep.cpu.leaf7_features | grep AVX2)" != "" ]; then
            #     CPU_FEATURES="-DENABLE_AVX=1 -DENABLE_AVX2=1"
            # else
            #     CPU_FEATURES="-DENABLE_AVX=1 -DENABLE_AVX2=0"
            # fi
        fi
    fi
elif [ "$(uname)" == "Linux" ]; then
    GLIBC_COMPATIBILITY="-DGLIBC_COMPATIBILITY=1"
    UNWIND="-DUSE_UNWIND=1"
    JEMALLOC="-DENABLE_JEMALLOC=1"
    PYINIT_ENTRY="-Wl,-ePyInit_${CHDB_PY_MOD}"
    ICU="-DENABLE_ICU=1"
    SED_INPLACE="sed -i"
    # only x86_64, enable AVX, enable embedded compiler
    if [ "$(uname -m)" == "x86_64" ]; then
        CPU_FEATURES="-DENABLE_AVX=1 -DENABLE_AVX2=0"
        LLVM="-DENABLE_EMBEDDED_COMPILER=1 -DENABLE_DWARF_PARSER=1"
    else
        CPU_FEATURES="-DENABLE_AVX=0 -DENABLE_AVX2=0"
        LLVM="-DENABLE_EMBEDDED_COMPILER=0 -DENABLE_DWARF_PARSER=0"
    fi
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

LIBCHDB_SO="libchdb.so"
# Build libchdb.so
cmake ${CMAKE_ARGS} -DENABLE_PYTHON=0 ..
ninja -d keeprsp


BINARY=${BUILD_DIR}/programs/clickhouse
echo -e "\nBINARY: ${BINARY}"
ls -lh ${BINARY}
echo -e "\nldd ${BINARY}"
${LDD} ${BINARY}
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

#   generate the command to generate libchdb.so
LIBCHDB_CMD=$(echo ${LIBCHDB_CMD} | sed 's/ '${CHDB_PY_MODULE}'/ '${LIBCHDB_SO}'/g')

if [ ! "${USING_RESPONSE_FILE}" == "" ]; then
    ${SED_INPLACE} 's/ '${CHDB_PY_MODULE}'/ '${LIBCHDB_SO}'/g' CMakeFiles/libchdb.rsp
fi

if [ "$(uname)" == "Linux" ]; then
    LIBCHDB_CMD=$(echo ${LIBCHDB_CMD} | sed 's/ '${PYINIT_ENTRY}'/ /g')
    if [ ! "${USING_RESPONSE_FILE}" == "" ]; then
        ${SED_INPLACE} 's/ '${PYINIT_ENTRY}'/ /g' CMakeFiles/libchdb.rsp
    fi
fi

if [ "$(uname)" == "Darwin" ]; then
    LIBCHDB_CMD=$(echo ${LIBCHDB_CMD} | sed 's/ '${PYINIT_ENTRY}'/ -Wl,-exported_symbol,_query_stable -Wl,-exported_symbol,_free_result -Wl,-exported_symbol,_query_stable_v2 -Wl,-exported_symbol,_free_result_v2/g')
    # ${SED_INPLACE} 's/ '${PYINIT_ENTRY}'/ -Wl,-exported_symbol,_query_stable -Wl,-exported_symbol,_free_result -Wl,-exported_symbol,_query_stable_v2 -Wl,-exported_symbol,_free_result_v2/g' CMakeFiles/libchdb.rsp
fi

LIBCHDB_CMD=$(echo ${LIBCHDB_CMD} | sed 's/@CMakeFiles\/clickhouse.rsp/@CMakeFiles\/libchdb.rsp/g')

# Step 4:
#   save the command to a file for debug
echo ${LIBCHDB_CMD} > libchdb_cmd.sh

# Step 5:
${LIBCHDB_CMD}

LIBCHDB_DIR=${BUILD_DIR}/
LIBCHDB=${LIBCHDB_DIR}/${LIBCHDB_SO}
ls -lh ${LIBCHDB}

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


# # inplace modify the CMakeFiles/pychdb.rsp
# ${SED_INPLACE} 's/-o programs\/clickhouse/-fPIC -Wl,-undefined,dynamic_lookup -shared ${PYINIT_ENTRY} -o ${CHDB_PY_MODULE}/' CMakeFiles/pychdb.rsp
# ${SED_INPLACE} 's/ -Wl,-undefined,error/ -Wl,-undefined,dynamic_lookup/g' CMakeFiles/pychdb.rsp
# ${SED_INPLACE} 's/ -Xlinker --no-undefined//g' CMakeFiles/pychdb.rsp


if [ "$(uname)" == "Linux" ]; then
    # remove src/CMakeFiles/clickhouse_malloc.dir/Common/stubFree.c.o
    PYCHDB_CMD=$(echo ${PYCHDB_CMD} | sed 's/ src\/CMakeFiles\/clickhouse_malloc.dir\/Common\/stubFree.c.o//g')
    # put -Wl,-wrap,malloc ... after -DUSE_JEMALLOC=1
    PYCHDB_CMD=$(echo ${PYCHDB_CMD} | sed 's/ -DUSE_JEMALLOC=1/ -DUSE_JEMALLOC=1 -Wl,-wrap,malloc -Wl,-wrap,valloc -Wl,-wrap,pvalloc -Wl,-wrap,calloc -Wl,-wrap,realloc -Wl,-wrap,memalign -Wl,-wrap,aligned_alloc -Wl,-wrap,posix_memalign -Wl,-wrap,free/g')
    if [ ! "${USING_RESPONSE_FILE}" == "" ]; then
        ${SED_INPLACE} 's/ src\/CMakeFiles\/clickhouse_malloc.dir\/Common\/stubFree.c.o//g' CMakeFiles/pychdb.rsp
        ${SED_INPLACE} 's/ -DUSE_JEMALLOC=1/ -DUSE_JEMALLOC=1 -Wl,-wrap,malloc -Wl,-wrap,valloc -Wl,-wrap,pvalloc -Wl,-wrap,calloc -Wl,-wrap,realloc -Wl,-wrap,memalign -Wl,-wrap,aligned_alloc -Wl,-wrap,posix_memalign -Wl,-wrap,free/g' CMakeFiles/pychdb.rsp
    fi
fi

if [ "$(uname)" == "Darwin" ]; then
    PYCHDB_CMD=$(echo ${PYCHDB_CMD} | sed 's|-Wl,-rpath,/[^[:space:]]*/pybind11-cmake|-Wl,-rpath,@loader_path|g')
else
    PYCHDB_CMD=$(echo ${PYCHDB_CMD} | sed 's|-Wl,-rpath,/[^[:space:]]*/pybind11-cmake|-Wl,-rpath,\$ORIGIN|g')
fi

# save the command to a file for debug
echo ${PYCHDB_CMD} > pychdb_cmd.sh

${PYCHDB_CMD}

ls -lh ${CHDB_PY_MODULE}

## check all the so files
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
echo -e "\nStripe the binary:"

echo -e "\nPYCHDB: ${PYCHDB}"
ls -lh ${PYCHDB}
echo -e "\nLIBCHDB: ${LIBCHDB}"
ls -lh ${LIBCHDB}
echo -e "\nldd ${PYCHDB}"
${LDD} ${PYCHDB}
echo -e "\nfile info of ${PYCHDB}"
file ${PYCHDB}
echo -e "\nldd ${LIBCHDB}"
${LDD} ${LIBCHDB}
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
# ls -lh ${PROJ_DIR}

# strip the binary (no debug info at all)
# strip ${CHDB_DIR}/${CHDB_PY_MODULE} || true

# echo -e "\nAfter strip:"
# echo -e "\nLIBCHDB: ${PYCHDB}"
# ls -lh ${CHDB_DIR}
# echo -e "\nfile info of ${PYCHDB}"
# file ${CHDB_DIR}/${CHDB_PY_MODULE}

ccache -s || true

# bash ${DIR}/build_bind.sh
# bash ${DIR}/test_smoke.sh

CMAKE_ARGS="${CMAKE_ARGS}" bash ${DIR}/build_pybind11.sh --all
