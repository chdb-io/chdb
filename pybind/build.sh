#!/bin/bash

set -e

# get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJ_DIR="${DIR}/.." # project root directory
BUILD_DIR="$PROJ_DIR/buildlib" # build directory
BIND_DIR="$PROJ_DIR/pybind" # bind directory
CHDB_DIR="$PROJ_DIR/chdb" # chdb directory

if [ ! -d $BUILD_DIR ]; then
    mkdir $BUILD_DIR
fi

cd ${BUILD_DIR}
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_THINLTO=1 -DENABLE_TESTS=0 -DENABLE_CLICKHOUSE_SERVER=0 -DENABLE_CLICKHOUSE_CLIENT=0 \
    -DENABLE_CLICKHOUSE_KEEPER=0 -DENABLE_CLICKHOUSE_KEEPER_CONVERTER=0 -DENABLE_CLICKHOUSE_LOCAL=1 -DENABLE_CLICKHOUSE_SU=0 -DENABLE_CLICKHOUSE_BENCHMARK=0 \
    -DENABLE_AZURE_BLOB_STORAGE=0 -DENABLE_CLICKHOUSE_COPIER=0 -DENABLE_CLICKHOUSE_DISKS=0 -DENABLE_CLICKHOUSE_FORMAT=0 -DENABLE_CLICKHOUSE_GIT_IMPORT=0 \
    -DENABLE_S3=0 -DENABLE_HDFS=0 -DENABLE_HIVE=0 \
    -DENABLE_CLICKHOUSE_OBFUSCATOR=0 -DENABLE_CLICKHOUSE_ODBC_BRIDGE=0 -DENABLE_ODBC=0 -DENABLE_CLICKHOUSE_STATIC_FILES_DISK_UPLOADER=0 \
    -DENABLE_KAFKA=0 -DENABLE_MYSQL=0 -DENABLE_NATS=0 -DENABLE_AMQPCPP=0 -DENABLE_NURAFT=0 \
    -DENABLE_CASSANDRA=0 -DENABLE_ODBC=0 -DENABLE_NLP=0 \
    -DENABLE_KRB5=0 -DENABLE_LDAP=0 \
    -DENABLE_LIBRARIES=0 \
    -DGLIBC_COMPATIBILITY=1 \
    -DCLICKHOUSE_ONE_SHARED=0 \
    -DENABLE_UTILS=0 -DENABLE_EMBEDDED_COMPILER=1 -DUSE_UNWIND=1 \
    -DENABLE_ICU=0 -DENABLE_JEMALLOC=0 \
    -DENABLE_PARQUET=1 -DENABLE_ROCKSDB=1 -DENABLE_SQLITE=1 -DENABLE_VECTORSCAN=1 \
    -DENABLE_PROTOBUF=1 -DENABLE_THRIFT=1 \
    -DENABLE_CLICKHOUSE_ALL=0 -DUSE_STATIC_LIBRARIES=1 -DSPLIT_SHARED_LIBRARIES=0 \
    -DENABLE_AVX=1 -DENABLE_AVX2=1 \
    -DENABLE_AVX512=0 -DENABLE_AVX512_VBMI=0 \
    ..
ninja

BINARY=${BUILD_DIR}/programs/clickhouse
echo -e "\nBINARY: ${BINARY}"
ls -lh ${BINARY}
echo -e "\nldd ${BINARY}"
ldd ${BINARY}

# del the binary and run ninja -v again to capture the command, then modify it to generate libchdb.so
/bin/rm -f ${BINARY} 
cd ${BUILD_DIR} 
ninja -v > build.log

# extract the command to generate libchdb.so

LIBCHDB_CMD=$(grep '/usr/bin/clang\+\+.*-o programs/clickhouse .*' build.log | sed 's/-o programs\/clickhouse/-fPIC -shared -o programs\/libchdb.so/' | sed 's/^[^&]*&& //' |sed 's/&&.*//')
${LIBCHDB_CMD}

LIBCHDB_DIR=${BUILD_DIR}/programs
LIBCHDB=${LIBCHDB_DIR}/libchdb.so
echo -e "\nLIBCHDB: ${LIBCHDB}"
ls -lh ${LIBCHDB}
echo -e "\nldd ${LIBCHDB}"
ldd ${LIBCHDB}
echo -e "\nfile info of ${LIBCHDB}"
file ${LIBCHDB}

# # strip the binary (no debug info at all)
# strip ${LIBCHDB}
# echo -e "\nAfter strip:"
# echo -e "\nLIBCHDB: ${LIBCHDB}"
# ls -lh ${LIBCHDB}
# echo -e "\nfile info of ${LIBCHDB}"
# file ${LIBCHDB}

cd ${BIND_DIR}
/bin/cp -a ${LIBCHDB} ${BIND_DIR}
/bin/cp -a ${LIBCHDB} ${CHDB_DIR}

CHDB_PY_MODULE="_chdb$(python3-config --extension-suffix)"

# compile the pybind module, MUST use "./libchdb.so" instead of ${LIBCHDB} or "libchdb.so"
clang++ -O3 -Wall -shared -std=c++17 -fPIC -I../ -I../base -I../src -I../programs/local/ \
    $(python3 -m pybind11 --includes) chdb.cpp \
    -Wl,--exclude-libs,ALL -stdlib=libstdc++ -static-libstdc++ -static-libgcc \
    ./libchdb.so -o ${CHDB_PY_MODULE} 

/bin/cp -a ${CHDB_PY_MODULE} ${CHDB_DIR}

# test the pybind module
python3 -c \
    "import _chdb; res = _chdb.query('select 1112222222,555', 'JSON'); print(res.get_memview().tobytes())"

python3 -c \
    "import _chdb; res = _chdb.query('select 1112222222,555', 'Arrow'); print(res.get_memview().tobytes())"

# test the python wrapped module
cd ${PROJ_DIR}
python3 -c \
    "import chdb; res = chdb._chdb.query('select version()', 'CSV'); print(str(res.get_memview().tobytes()))"
