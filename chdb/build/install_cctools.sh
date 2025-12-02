#!/bin/bash

set -e

# Parse arguments
TARGET_ARCH="${1:-x86_64}"

# Set Darwin triple based on architecture
if [ "$TARGET_ARCH" == "x86_64" ]; then
    DARWIN_TRIPLE="x86_64-apple-darwin"
else
    DARWIN_TRIPLE="aarch64-apple-darwin"
fi

# Install cctools if not already installed
CCTOOLS_INSTALL_DIR="${HOME}/cctools"
CCTOOLS_BIN="${CCTOOLS_INSTALL_DIR}/bin"

if [ -z "${CCTOOLS:-}" ]; then
    echo "CCTOOLS environment variable not set, checking for installation..." >&2

    # Check if cctools is already installed
    if [ -f "${CCTOOLS_BIN}/${DARWIN_TRIPLE}-ld" ]; then
        echo "Found existing cctools installation at ${CCTOOLS_INSTALL_DIR}" >&2
        export CCTOOLS="${CCTOOLS_BIN}"
    else
        echo "cctools not found, installing..." >&2

        mkdir -p ~/cctools
        export CCTOOLS=$(cd ~/cctools && pwd)
        cd ${CCTOOLS}

        git clone https://github.com/tpoechtrager/apple-libtapi.git
        cd apple-libtapi
        git checkout 15dfc2a8c9a2a89d06ff227560a69f5265b692f9
        INSTALLPREFIX=${CCTOOLS} ./build.sh
        ./install.sh
        cd ..

        git clone https://github.com/chdb-io/cctools-port.git
        cd cctools-port/cctools

        # Set cctools target based on architecture
        if [ "$TARGET_ARCH" == "x86_64" ]; then
            CCTOOLS_TARGET="x86_64-apple-darwin"
        else
            CCTOOLS_TARGET="aarch64-apple-darwin"
        fi

        ./configure --prefix=$(readlink -f ${CCTOOLS}) --with-libtapi=$(readlink -f ${CCTOOLS}) --target=${CCTOOLS_TARGET}
        make install
    fi
else
    echo "Using CCTOOLS from environment variable: ${CCTOOLS}" >&2
fi

# Verify cctools installation
if [ ! -f "${CCTOOLS_BIN}/${DARWIN_TRIPLE}-ld" ]; then
    echo "Error: cctools linker not found at ${CCTOOLS}/${DARWIN_TRIPLE}-ld" >&2
    echo "Please verify cctools installation or set CCTOOLS environment variable correctly" >&2
    exit 1
fi

echo "cctools verified: ${CCTOOLS_BIN}/${DARWIN_TRIPLE}-ld" >&2
