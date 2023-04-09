#!/usr/bin/arch -arm64 /bin/bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJ_DIR=$(dirname ${DIR})

cd ${PROJ_DIR}

# Download py39 py310 py311 universal2 pkg from python.org
if [ ! -d ${PROJ_DIR}/python_pkg ]; then
    mkdir ${PROJ_DIR}/python_pkg
fi

# prefer /usr/local/opt/llvm@15/bin/clang++ then /usr/local/opt/llvm/bin/clang++
if [ -f /usr/local/opt/llvm@15/bin/clang++ ]; then
    export CXX=/usr/local/opt/llvm@15/bin/clang++
elif [ -f /usr/local/opt/llvm/bin/clang++ ]; then
    export CXX=/usr/local/opt/llvm/bin/clang++
fi
if [ -f /usr/local/opt/llvm@15/bin/clang ]; then
    export CC=/usr/local/opt/llvm@15/bin/clang
elif [ -f /usr/local/opt/llvm/bin/clang ]; then
    export CC=/usr/local/opt/llvm/bin/clang
fi

for PY_VER in 3.9.13 3.10.11 3.11.3; do
    if [ ! -f ${PROJ_DIR}/python_pkg/python-${PY_VER}-macos11.pkg ]; then
        wget https://www.python.org/ftp/python/${PY_VER}/python-${PY_VER}-macos11.pkg -O ${PROJ_DIR}/python_pkg/python-${PY_VER}-macos11.pkg
    fi
    
    PY_SHORT_VER=$(echo ${PY_VER} | cut -d. -f1,2)
    # Install universal2 pkg
    sudo installer -pkg ${PROJ_DIR}/python_pkg/python-${PY_VER}-macos11.pkg -target /
    export PATH=/Library/Frameworks/Python.framework/Versions/${PY_SHORT_VER}/bin:/Library/Frameworks/Python.framework/Versions/${PY_SHORT_VER}/Resources/Python.app/Contents/MacOS/:$PATH
    python3 -VV
    python3-config --includes
    # if python3 -VV does not contain ${PY_VER}, then exit
    if ! python3 -VV 2>&1 | grep -q "${PY_VER}"; then
        echo "Error: Required version of Python (${PY_VER}) not found. Aborting."
        exit 1
    fi
    # if python3-config --includes does not contain ${PY_SHORT_VER}, then exit
    if ! python3-config --includes 2>&1 | grep -q "${PY_SHORT_VER}"; then
        echo "Error: Required version of Python (${PY_VER}) not found. Aborting."
        exit 1
    fi

    python3 -m pip install -U pybind11 wheel build tox
    rm -rf ${PROJ_DIR}/buildlib

    ${PROJ_DIR}/chdb/build.sh
    cd ${PROJ_DIR}/chdb && python3 -c "import _chdb; res = _chdb.query('select 1112222222,555', 'JSON'); print(res.get_memview().tobytes())"

    cd ${PROJ_DIR}
    ${PROJ_DIR}/gen_manifest.sh
    cat ${PROJ_DIR}/MANIFEST.in

    python3 -m build --wheel

    python3 -m wheel tags --platform-tag=macosx_11_0_arm64 --remove dist/chdb-*-cp${PY_SHORT_VER//./}-cp${PY_SHORT_VER//./}-macosx_*_universal2.whl

    python3 -m pip install --force-reinstall dist/chdb-*-cp${PY_SHORT_VER//./}-cp${PY_SHORT_VER//./}-macosx_11_0_arm64.whl

    python3 -c "import chdb; res = chdb.query('select version()', 'CSV'); print(str(res.get_memview().tobytes()))"
done

cd ${PROJ_DIR}
make pub

