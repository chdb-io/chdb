#!/bin/bash
set -e

# Get script directory
MY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${MY_DIR}/../.." && pwd)"

# Allow custom library path
LIBCHDB_PATH="${1:-${MY_DIR}/libchdb_minimal.a}"

echo "Testing with Go example..."
echo "Using library: ${LIBCHDB_PATH}"

# Prepare go-example directory
echo "Preparing go-example directory..."
cd ${MY_DIR}/go-example

# Copy library and header
if [ -f "${LIBCHDB_PATH}" ]; then
    cp "${LIBCHDB_PATH}" ./libchdb.a
else
    echo "Error: Library not found: ${LIBCHDB_PATH}"
    exit 1
fi

cp ${PROJ_DIR}/programs/local/chdb.h .
echo "Copied library as libchdb.a and chdb.h to go-example directory"

# Run Go test
echo "Running Go test..."
go run .
if [ $? -ne 0 ]; then
    echo "Error: Go test failed"
    exit 1
fi

echo "Go test completed successfully!"
