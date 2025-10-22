#!/bin/bash

set -e

build_all=false
py_version=""

for arg in "$@"; do
    case $arg in
        --all)
            build_all=true
            shift
            ;;
        --version=*)
            py_version="${arg#*=}"
            shift
            ;;
    esac
done

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/vars.sh

# Check if CMAKE_ARGS is passed from build.sh
if [ -z "$CMAKE_ARGS" ]; then
    echo "Error: CMAKE_ARGS not provided. This script should be called from build.sh."
    exit 1
fi

build_pybind11_nonlimitedapi() {
    cd "${BUILD_DIR}"

    local py_version=$1
    echo "Building pybind11 nonlimitedapi library for Python ${py_version}..."

    local py_cmake_args="${CMAKE_ARGS} -DPYBIND11_NONLIMITEDAPI_PYTHON_HEADERS_VERSION=${py_version}"

    cmake ${py_cmake_args} -DENABLE_PYTHON=1 ..

    # Build only the pybind11 targets
    ninja pybind11nonlimitedapi_chdb_${py_version} || {
        echo "Failed to build pybind11nonlimitedapi library for Python ${py_version}"
        return 1
    }

    # Copy the built library to output directory
    local lib_name="pybind11nonlimitedapi_chdb_${py_version}"
    if [ "$(uname)" == "Darwin" ]; then
        local lib_file="lib${lib_name}.dylib"
    else
        local lib_file="lib${lib_name}.so"
    fi

    if [ -f "${BUILD_DIR}/contrib/pybind11-cmake/${lib_file}" ]; then
        cp "${BUILD_DIR}/contrib/pybind11-cmake/${lib_file}" "${CHDB_DIR}/${lib_file}"
        echo "Copied ${lib_file} to ${CHDB_DIR}/"
        echo "Library location: $(realpath ${CHDB_DIR}/${lib_file})"
    else
        echo "Warning: ${lib_file} not found in ${BUILD_DIR}/contrib/pybind11-cmake/"
        echo "Available files in contrib/pybind11-cmake/:"
        ls -la "${BUILD_DIR}/contrib/pybind11-cmake/" || true
    fi
}

build_all_pybind11_nonlimitedapi() {
    local python_versions=("3.8" "3.9" "3.10" "3.11" "3.12" "3.13")
    
    # Skip Python 3.8 for macOS x86_64
    if [ "$(uname)" == "Darwin" ] && [ "$(uname -m)" == "x86_64" ]; then
        python_versions=("3.9" "3.10" "3.11" "3.12" "3.13")
    fi

    echo "Building pybind11 nonlimitedapi libraries for all Python versions..."

    # Check if pyenv is available
    if [ -z "$(command -v pyenv)" ]; then
        echo "Error: pyenv not found. Please install pyenv first."
        exit 1
    fi

    for version in "${python_versions[@]}"; do
        # Use pyenv to find specific version
        local pyenv_version=$(pyenv versions --bare | grep "^${version}\." | head -1)
        if [ -z "$pyenv_version" ]; then
            echo "Error: Python ${version} not found in pyenv. Please install it with: pyenv install ${version}.x"
            exit 1
        fi

        echo "Found pyenv Python ${pyenv_version}"
        export PYENV_VERSION=$pyenv_version

        local python_include=$(python -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null)
        local active_version=$(python --version 2>&1)
        echo "  Active Python: $active_version"

        if [ -f "$python_include/Python.h" ]; then
            echo "  Headers found at: $python_include"
            build_pybind11_nonlimitedapi "${version}"
        else
            echo "Error: Python.h not found for Python ${version} at $python_include"
            unset PYENV_VERSION
            exit 1
        fi

        unset PYENV_VERSION
    done

    echo "Finished building pybind11 nonlimitedapi libraries"
}

copy_stubs() {
    if [ "$(uname)" == "Darwin" ]; then
        local lib_file="libpybind11nonlimitedapi_stubs.dylib"
    else
        local lib_file="libpybind11nonlimitedapi_stubs.so"
    fi
    if [ -f ${BUILD_DIR}/contrib/pybind11-cmake/${lib_file} ]; then
        cp -a ${BUILD_DIR}/contrib/pybind11-cmake/${lib_file} ${CHDB_DIR}/
    fi
}

if [ "$build_all" = true ]; then
    copy_stubs
    build_all_pybind11_nonlimitedapi
elif [ -n "$py_version" ]; then
    copy_stubs
    build_pybind11_nonlimitedapi "$py_version"
else
    echo "No action specified."
    exit 1
fi
