#!/bin/bash
# Stop ClickHouse test server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CH_DIR="${SCRIPT_DIR}/.clickhouse"
CH_PID="${CH_DIR}/clickhouse.pid"
CH_PORT="${TEST_CLICKHOUSE_PORT:-19000}"

# Stop by PID file
if [ -f "${CH_PID}" ]; then
    PID=$(cat "${CH_PID}")
    if kill -0 "${PID}" 2>/dev/null; then
        echo "Stopping ClickHouse server (PID: ${PID})..."
        kill "${PID}"
        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 "${PID}" 2>/dev/null; then
                echo "Server stopped gracefully."
                break
            fi
            sleep 1
        done
        # Force kill if still running
        if kill -0 "${PID}" 2>/dev/null; then
            echo "Force killing server..."
            kill -9 "${PID}" 2>/dev/null || true
        fi
    else
        echo "Server not running (stale PID file)"
    fi
    rm -f "${CH_PID}"
else
    echo "No PID file found"
fi

# Also check for any process on the port (cleanup orphans)
if command -v lsof >/dev/null 2>&1; then
    ORPHAN_PID=$(lsof -ti:${CH_PORT} 2>/dev/null || true)
    if [ -n "$ORPHAN_PID" ]; then
        echo "Cleaning up orphaned process on port ${CH_PORT} (PID: ${ORPHAN_PID})..."
        kill -9 $ORPHAN_PID 2>/dev/null || true
    fi
fi

echo "Cleanup complete."
