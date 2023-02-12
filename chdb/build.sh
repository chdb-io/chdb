#!/bin/bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

. ${DIR}/vars.sh

BUILD_DIR=${PROJ_DIR}/buildlib

# check current os type
if [ "$(uname)" == "Darwin" ]; then
    GLIBC_COMPATIBILITY="-DGLIBC_COMPATIBILITY=0"
    UNWIND="-DUSE_UNWIND=0"
    PYINIT_ENTRY="-Wl,-exported_symbol,_PyInit_${CHDB_PY_MOD}"
elif [ "$(uname)" == "Linux" ]; then
    GLIBC_COMPATIBILITY="-DGLIBC_COMPATIBILITY=1"
    UNWIND="-DUSE_UNWIND=1"
    PYINIT_ENTRY="-Wl,-ePyInit_${CHDB_PY_MOD}"
else
    echo "OS not supported"
    exit 1
fi

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
    ${GLIBC_COMPATIBILITY} \
    -DCLICKHOUSE_ONE_SHARED=0 \
    -DENABLE_UTILS=0 -DENABLE_EMBEDDED_COMPILER=1 ${UNWIND} \
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
${LDD} ${BINARY}

# del the binary and run ninja -v again to capture the command, then modify it to generate CHDB_PY_MODULE
/bin/rm -f ${BINARY} 
cd ${BUILD_DIR} 
ninja -v > build.log

# extract the command to generate CHDB_PY_MODULE

LIBCHDB_CMD=$(grep 'clang++.*-o programs/clickhouse .*' build.log \
    | sed "s/-o programs\/clickhouse/-fPIC -shared ${PYINIT_ENTRY} -o ${CHDB_PY_MODULE}/" \
    | sed 's/^[^&]*&& //' | sed 's/&&.*//' \
    | sed 's/ -Wl,-undefined,error/ -Wl,-undefined,dynamic_lookup/g' \
    | sed 's/ -Xlinker --no-undefined//g' \
     )

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

/bin/cp -a ${LIBCHDB} ${CHDB_DIR}/${CHDB_PY_MODULE}

# # strip the binary (no debug info at all)
# strip ${LIBCHDB}
# echo -e "\nAfter strip:"
# echo -e "\nLIBCHDB: ${LIBCHDB}"
# ls -lh ${LIBCHDB}
# echo -e "\nfile info of ${LIBCHDB}"
# file ${LIBCHDB}

# bash ${DIR}/build_bind.sh
# bash ${DIR}/test_smoke.sh