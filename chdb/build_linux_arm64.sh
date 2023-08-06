#!/bin/bash -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJ_DIR=$(dirname ${DIR})

[ -z "${PYTHON_VERSIONS}" ] && { echo "please provide PYTHON_VERSIONS env, e.g: PYTHON_VERSIONS='3.8 3.9'"; exit 1; }

# echo ${PROJ_DIR}

for PY_VER in ${PYTHON_VERSIONS}; do
    cd ${PROJ_DIR}
    pyenv local "${PY_VER}"
    python3 --version
    python3 -m pip install pybind11
    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    # Install universal2 pkg
    python3 -VV
    python3-config --includes
    # if python3 -VV does not contain ${PY_VER}, then exit
    if ! python3 -VV 2>&1 | grep -q "${PY_VER}"; then
        echo "Error: Required version of Python (${PY_VER}) not found. Aborting."
        exit 1
    fi

    python3 -m pip install -U pybind11 wheel build tox
    rm -rf ./buildlib

    ./chdb/build.sh
    cd chdb && python3 -c "import _chdb; res = _chdb.query('select 1112222222,555', 'JSON'); print(res)" && cd -

    ./gen_manifest.sh
    cat ./MANIFEST.in

    # try delete 
    whl_file=$(find dist | grep 'whl$' | grep cp${PY_VER//./}-cp${PY_VER//./} || echo "notfound.whl")
    rm -f ${whl_file} || :

    python3 -m build --wheel

    python3 -m wheel tags --platform-tag=manylinux_2_17_aarch64

    find dist

    python3 -m pip install pandas pyarrow psutil
    find dist
    whl_file=$(find dist | grep 'whl$' | grep cp${PY_VER//./}-cp${PY_VER//./})
    python3 -m pip install --force-reinstall ${whl_file}

    python3 -c "import chdb; res = chdb.query('select version()', 'CSV'); print(res)"

    python3 -m chdb "SELECT 1, 'ab'" arrowtable

    python3 -m chdb "SELECT 1, 'ab'" dataframe

    bash -x ./chdb/test_smoke.sh

    make test
done
