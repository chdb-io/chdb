#!/bin/bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

. ${DIR}/vars.sh

# test the pybind module
cd ${CHDB_DIR}

python3 -c \
    "import _chdb; res = _chdb.query('select 1112222222,555', 'JSON'); print(res.get_memview().tobytes())"

python3 -c \
    "import _chdb; res = _chdb.query('select 1112222222,555', 'Arrow'); print(res.get_memview().tobytes())"

# test the python wrapped module
cd ${PROJ_DIR}

python3 -c \
    "import chdb; res = chdb._chdb.query('select version()', 'CSV'); print(str(res.get_memview().tobytes()))"
