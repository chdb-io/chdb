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

# Build Go binary (instead of go run) so we can analyze crashes
echo "Building Go test binary..."
go build -o chdb_go_test .

# Run Go test with crash detection
echo "Running Go test..."

# Enable Go crash dump and core dump
export GOTRACEBACK=crash
ulimit -c unlimited 2>/dev/null || true

# Set up core dump directory
CORE_DIR="${MY_DIR}/go-cores"
mkdir -p "${CORE_DIR}"

# On macOS, configure core dump location
if [ "$(uname)" == "Darwin" ]; then
    sudo sysctl kern.corefile="${CORE_DIR}/core.%P" 2>/dev/null || true
    sudo sysctl kern.coredump=1 2>/dev/null || true
fi

EXIT_CODE=0
./chdb_go_test || EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "Go test CRASHED with exit code: $EXIT_CODE"
    echo "========================================"

    # Try to find and analyze core dump
    CORE_FILE=""
    if [ "$(uname)" == "Darwin" ]; then
        # Look for core files in configured dir and default locations
        CORE_FILE=$(ls -t "${CORE_DIR}"/core.* /cores/core.* 2>/dev/null | head -1)
    else
        CORE_FILE=$(ls -t core.* "${CORE_DIR}"/core.* 2>/dev/null | head -1)
    fi

    if [ -n "$CORE_FILE" ] && [ -f "$CORE_FILE" ]; then
        echo "Found core dump: $CORE_FILE"
        echo ""

        if command -v lldb &>/dev/null; then
            echo "=== Stack trace (lldb) ==="
            lldb -c "$CORE_FILE" ./chdb_go_test -b \
                -o "bt all" \
                -o "quit" 2>&1 || true
            echo ""
        fi
    else
        echo "No core dump found. Trying to reproduce with lldb..."
        echo ""

        if command -v lldb &>/dev/null; then
            echo "=== Re-running under lldb to capture crash ==="
            LLDB_OUTPUT=$(mktemp)
            lldb -b \
                -o "process handle SIGILL -n true -p true -s true" \
                -o "process handle SIGABRT -n true -p true -s true" \
                -o "process handle SIGSEGV -n true -p true -s true" \
                -o "process handle SIGBUS -n true -p true -s true" \
                -o "env GOTRACEBACK=crash" \
                -o "run" \
                -o "bt all" \
                -o "thread list" \
                -o "register read" \
                -o "quit" \
                -- ./chdb_go_test 2>&1 | tee "$LLDB_OUTPUT" || true
            rm -f "$LLDB_OUTPUT"
            echo ""
        fi
    fi

    echo "Go test failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "Go test completed successfully!"
