#!/bin/bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

. ${DIR}/vars_libchdb.sh

BUILD_DIR=${PROJ_DIR}/buildlib

# check current os type
if [ "$(uname)" == "Darwin" ]; then
    GLIBC_COMPATIBILITY="-DGLIBC_COMPATIBILITY=0"
    UNWIND="-DUSE_UNWIND=0"
    JEMALLOC="-DENABLE_JEMALLOC=0"
    PYINIT_ENTRY="-Wl,-exported_symbol,_PyInit_${CHDB_PY_MOD}"
    # if Darwin ARM64 (M1, M2), disable AVX
    if [ "$(uname -m)" == "arm64" ]; then
        CMAKE_TOOLCHAIN_FILE="-DCMAKE_TOOLCHAIN_FILE=cmake/darwin/toolchain-aarch64.cmake"
        CPU_FEATURES="-DENABLE_AVX=0 -DENABLE_AVX2=0"
        EMBEDDED_COMPILER="-DENABLE_EMBEDDED_COMPILER=0"
#        export PATH=$(brew --prefix llvm@15)/bin:$(brew --prefix)/opt/grep/libexec/gnubin:$(brew --prefix)/opt/binutils/bin:$PATH:$(brew --prefix)/opt/findutils/libexec/gnubin
#        export PATH=$(brew --prefix llvm@15)/bin:$(brew --prefix)/opt/grep/libexec/gnubin:$(brew --prefix)/opt/binutils/bin:$(brew --prefix)/opt/protobuf/bin:$(brew --prefix)/opt/findutils/libexec/gnubin:$PATH
        export CC=$(brew --prefix llvm@15)/bin/clang
        export CXX=$(brew --prefix llvm@15)/bin/clang++
    else
        EMBEDDED_COMPILER="-DENABLE_EMBEDDED_COMPILER=1"
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
    # only x86_64, enable AVX and AVX2, enable embedded compiler
    if [ "$(uname -m)" == "x86_64" ]; then
        CPU_FEATURES="-DENABLE_AVX=1 -DENABLE_AVX2=1"
        EMBEDDED_COMPILER="-DENABLE_EMBEDDED_COMPILER=1"
    else
        CPU_FEATURES="-DENABLE_AVX=0 -DENABLE_AVX2=0 -DNO_ARMV81_OR_HIGHER=1"
        EMBEDDED_COMPILER="-DENABLE_EMBEDDED_COMPILER=0"
    fi
else
    echo "OS not supported"
    exit 1
fi

if [ ! -d $BUILD_DIR ]; then
    mkdir $BUILD_DIR
fi

cd ${BUILD_DIR}
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_THINLTO=0 -DENABLE_TESTS=0 -DENABLE_CLICKHOUSE_SERVER=0 -DENABLE_CLICKHOUSE_CLIENT=0 \
    -DENABLE_CLICKHOUSE_KEEPER=0 -DENABLE_CLICKHOUSE_KEEPER_CONVERTER=0 -DENABLE_CLICKHOUSE_LOCAL=1 -DENABLE_CLICKHOUSE_SU=0 -DENABLE_CLICKHOUSE_BENCHMARK=0 \
    -DENABLE_AZURE_BLOB_STORAGE=0 -DENABLE_CLICKHOUSE_COPIER=0 -DENABLE_CLICKHOUSE_DISKS=0 -DENABLE_CLICKHOUSE_FORMAT=0 -DENABLE_CLICKHOUSE_GIT_IMPORT=0 \
    -DENABLE_AWS_S3=1 -DENABLE_HDFS=0 -DENABLE_HIVE=0 \
    -DENABLE_CLICKHOUSE_OBFUSCATOR=0 -DENABLE_CLICKHOUSE_ODBC_BRIDGE=0 -DENABLE_ODBC=0 -DENABLE_CLICKHOUSE_STATIC_FILES_DISK_UPLOADER=0 \
    -DENABLE_KAFKA=0 -DENABLE_MYSQL=0 -DENABLE_NATS=0 -DENABLE_AMQPCPP=0 -DENABLE_NURAFT=0 \
    -DENABLE_CASSANDRA=0 -DENABLE_ODBC=0 -DENABLE_NLP=0 \
    -DENABLE_KRB5=0 -DENABLE_LDAP=0 \
    -DENABLE_LIBRARIES=0 -DENABLE_RUST=0 \
    ${GLIBC_COMPATIBILITY} \
    -DENABLE_UTILS=0 ${EMBEDDED_COMPILER} ${UNWIND} \
    -DENABLE_ICU=0 ${JEMALLOC} \
    -DENABLE_PARQUET=1 -DENABLE_ROCKSDB=1 -DENABLE_SQLITE=1 -DENABLE_VECTORSCAN=1 \
    -DENABLE_PROTOBUF=1 -DENABLE_THRIFT=1 \
    -DENABLE_BROTLI=1 \
    -DENABLE_LIBPQXX=1 \
    -DENABLE_CLICKHOUSE_ALL=0 -DUSE_STATIC_LIBRARIES=1 -DSPLIT_SHARED_LIBRARIES=0 \
    ${CPU_FEATURES} \
    ${CMAKE_TOOLCHAIN_FILE} \
    -DENABLE_AVX512=0 -DENABLE_AVX512_VBMI=0 \
    ..
ninja

BINARY=${BUILD_DIR}/programs/clickhouse
echo -e "\nBINARY: ${BINARY}"
ls -lh ${BINARY}
echo -e "\nldd ${BINARY}"
${LDD} ${BINARY}
rm -f ${BINARY}

# del the binary and run ninja -v again to capture the command, then modify it to generate CHDB_PY_MODULE
/bin/rm -f ${BINARY} 
cd ${BUILD_DIR} 
ninja -v > build.log

# extract the command to generate CHDB_PY_MODULE

LIBCHDB_CMD=$(grep 'clang++.*-o programs/clickhouse .*' build.log \
    | sed "s/-o programs\/clickhouse/-fPIC -shared  -o ${CHDB_PY_MODULE}/" \
    | sed 's/^[^&]*&& //' | sed 's/&&.*//' \
    | sed 's/ -Wl,-undefined,error/ -Wl,-undefined,dynamic_lookup/g' \
    | sed 's/ -Xlinker --no-undefined//g' \
     )

if [ "$(uname)" == "Linux" ]; then
    # remove src/CMakeFiles/clickhouse_malloc.dir/Common/stubFree.c.o
    LIBCHDB_CMD=$(echo ${LIBCHDB_CMD} | sed 's/ src\/CMakeFiles\/clickhouse_malloc.dir\/Common\/stubFree.c.o//g')
    # put -Wl,-wrap,malloc ... after -DUSE_JEMALLOC=1
    LIBCHDB_CMD=$(echo ${LIBCHDB_CMD} | sed 's/ -DUSE_JEMALLOC=1/ -DUSE_JEMALLOC=1 -Wl,-wrap,malloc -Wl,-wrap,valloc -Wl,-wrap,pvalloc -Wl,-wrap,calloc -Wl,-wrap,realloc -Wl,-wrap,memalign -Wl,-wrap,aligned_alloc -Wl,-wrap,posix_memalign -Wl,-wrap,free/g')
fi

# save the command to a file for debug
echo ${LIBCHDB_CMD} > libchdb_cmd.sh

${LIBCHDB_CMD}

LIBCHDB_DIR=${BUILD_DIR}/
LIBCHDB=${LIBCHDB_DIR}/${CHDB_PY_MODULE}
echo -e "\nLIBCHDB: ${LIBCHDB}"
ls -lh ${LIBCHDB}
echo -e "\nldd ${LIBCHDB}"
${LDD} ${LIBCHDB}
echo -e "\nfile info of ${LIBCHDB}"
file ${LIBCHDB}

rm -f ${CHDB_DIR}/*.so
mv ${LIBCHDB} ${CHDB_DIR}/${CHDB_PY_MODULE}
strip ${CHDB_DIR}/${CHDB_PY_MODULE} || true

# strip the binary (no debug info at all)
strip ${CHDB_DIR}/${CHDB_PY_MODULE} || true
# echo -e "\nAfter strip:"
# echo -e "\nLIBCHDB: ${LIBCHDB}"
# ls -lh ${CHDB_DIR}
# echo -e "\nfile info of ${LIBCHDB}"
# file ${CHDB_DIR}/${CHDB_PY_MODULE}

ccache -s || true

# bash ${DIR}/build_bind.sh
# bash ${DIR}/test_smoke.sh
