#!/bin/bash
# Setup ClickHouse server for integration testing
# Supports: macOS (x86_64, arm64), Linux (x86_64, aarch64)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CH_DIR="${SCRIPT_DIR}/.clickhouse"
CH_BINARY="${CH_DIR}/clickhouse"
CH_DATA="${CH_DIR}/data"
CH_LOG="${CH_DIR}/log"
CH_PID="${CH_DIR}/clickhouse.pid"
CH_PORT="${TEST_CLICKHOUSE_PORT:-19000}"

# Detect OS and architecture
detect_platform() {
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    local arch=$(uname -m)
    
    # Normalize architecture names
    case "$arch" in
        x86_64|amd64)
            arch="amd64"
            ;;
        arm64|aarch64)
            arch="aarch64"
            ;;
        *)
            echo "Unsupported architecture: $arch" >&2
            exit 1
            ;;
    esac
    
    # Normalize OS names
    case "$os" in
        darwin)
            os="macos"
            ;;
        linux)
            os="linux"
            ;;
        *)
            echo "Unsupported OS: $os" >&2
            exit 1
            ;;
    esac
    
    echo "${os}-${arch}"
}

# Get download URL for ClickHouse binary
# Using official ClickHouse 25.12 release builds
get_download_url() {
    local platform=$1
    local version="25.12.4.35"
    local base_url="https://github.com/ClickHouse/ClickHouse/releases/download/v${version}-stable"
    
    case "$platform" in
        macos-amd64)
            echo "${base_url}/clickhouse-macos"
            ;;
        macos-aarch64)
            echo "${base_url}/clickhouse-macos-aarch64"
            ;;
        linux-amd64)
            echo "${base_url}/clickhouse-linux-amd64"
            ;;
        linux-aarch64)
            echo "${base_url}/clickhouse-linux-aarch64"
            ;;
        *)
            echo "Unknown platform: $platform" >&2
            exit 1
            ;;
    esac
}

# Create directories
mkdir -p "${CH_DIR}" "${CH_DATA}" "${CH_LOG}"

# Detect platform
PLATFORM=$(detect_platform)
echo "Detected platform: ${PLATFORM}"

# Download ClickHouse if not present
if [ ! -f "${CH_BINARY}" ]; then
    echo "Downloading ClickHouse binary for ${PLATFORM}..."
    DOWNLOAD_URL=$(get_download_url "$PLATFORM")
    echo "URL: ${DOWNLOAD_URL}"
    
    # Download with curl, follow redirects
    if ! curl -L -f -o "${CH_BINARY}" "${DOWNLOAD_URL}"; then
        echo "Failed to download ClickHouse from ${DOWNLOAD_URL}" >&2
        exit 1
    fi
    chmod +x "${CH_BINARY}"
    echo "Download complete!"
fi

# Verify binary works
echo "Verifying ClickHouse binary..."
if ! "${CH_BINARY}" --version; then
    echo "ClickHouse binary verification failed. Redownloading..." >&2
    rm -f "${CH_BINARY}"
    DOWNLOAD_URL=$(get_download_url "$PLATFORM")
    if ! curl -L -f -o "${CH_BINARY}" "${DOWNLOAD_URL}"; then
        echo "Failed to download ClickHouse" >&2
        exit 1
    fi
    chmod +x "${CH_BINARY}"
    if ! "${CH_BINARY}" --version; then
        echo "ClickHouse binary still not working after redownload" >&2
        exit 1
    fi
fi

# Check if server is already running
if [ -f "${CH_PID}" ]; then
    PID=$(cat "${CH_PID}")
    if kill -0 "${PID}" 2>/dev/null; then
        echo "ClickHouse server already running (PID: ${PID})"
        exit 0
    fi
    # Clean up stale PID file
    rm -f "${CH_PID}"
fi

# Defensive cleanup: kill any orphaned clickhouse processes on our port
echo "Checking for orphaned processes on port ${CH_PORT}..."
if command -v lsof >/dev/null 2>&1; then
    ORPHAN_PID=$(lsof -ti:${CH_PORT} 2>/dev/null || true)
    if [ -n "$ORPHAN_PID" ]; then
        echo "Found orphaned process on port ${CH_PORT} (PID: ${ORPHAN_PID}), killing..."
        kill -9 $ORPHAN_PID 2>/dev/null || true
        sleep 1
    fi
fi

# Clean up old data directory to ensure fresh state
echo "Cleaning up old data directory..."
rm -rf "${CH_DATA}"
mkdir -p "${CH_DATA}" "${CH_DATA}/tmp" "${CH_DATA}/user_files" "${CH_DATA}/format_schemas"

echo "Starting ClickHouse server on port ${CH_PORT}..."

# Create minimal config file
CONFIG_FILE="${CH_DIR}/config.xml"
cat > "${CONFIG_FILE}" << EOF
<?xml version="1.0"?>
<clickhouse>
    <tcp_port>${CH_PORT}</tcp_port>
    <path>${CH_DATA}/</path>
    <tmp_path>${CH_DATA}/tmp/</tmp_path>
    <user_files_path>${CH_DATA}/user_files/</user_files_path>
    <format_schema_path>${CH_DATA}/format_schemas/</format_schema_path>
    <logger>
        <level>warning</level>
        <log>${CH_LOG}/clickhouse-server.log</log>
        <errorlog>${CH_LOG}/clickhouse-server.err.log</errorlog>
    </logger>
    <listen_host>127.0.0.1</listen_host>
    <users_config>users.xml</users_config>
</clickhouse>
EOF

# Create users.xml with default user and profile
USERS_FILE="${CH_DIR}/users.xml"
cat > "${USERS_FILE}" << EOF
<?xml version="1.0"?>
<clickhouse>
    <profiles>
        <default>
            <max_memory_usage>10000000000</max_memory_usage>
            <use_uncompressed_cache>0</use_uncompressed_cache>
            <load_balancing>random</load_balancing>
        </default>
    </profiles>
    <users>
        <default>
            <password></password>
            <networks>
                <ip>::/0</ip>
            </networks>
            <profile>default</profile>
            <quota>default</quota>
            <access_management>1</access_management>
        </default>
    </users>
    <quotas>
        <default>
            <interval>
                <duration>3600</duration>
                <queries>0</queries>
                <errors>0</errors>
                <result_rows>0</result_rows>
                <read_rows>0</read_rows>
                <execution_time>0</execution_time>
            </interval>
        </default>
    </quotas>
</clickhouse>
EOF

# Start server in daemon mode
"${CH_BINARY}" server --config-file="${CONFIG_FILE}" --daemon --pid-file="${CH_PID}"

# Wait for server to be ready
echo "Waiting for server to start..."
MAX_WAIT=60
for i in $(seq 1 $MAX_WAIT); do
    if "${CH_BINARY}" client --port="${CH_PORT}" --query="SELECT 1" >/dev/null 2>&1; then
        echo "ClickHouse server is ready! (took ${i}s)"
        
        # Create test database and tables
        "${CH_BINARY}" client --port="${CH_PORT}" --multiquery <<EOF
CREATE DATABASE IF NOT EXISTS test_db;

DROP TABLE IF EXISTS test_db.users;
CREATE TABLE test_db.users (
    id UInt64,
    name String,
    email String,
    age UInt8,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree() ORDER BY id;

INSERT INTO test_db.users (id, name, email, age) VALUES
    (1, 'Alice', 'alice@example.com', 25),
    (2, 'Bob', 'bob@example.com', 30),
    (3, 'Charlie', 'charlie@example.com', 35);

DROP TABLE IF EXISTS test_db.orders;
CREATE TABLE test_db.orders (
    id UInt64,
    user_id UInt64,
    amount Decimal(10, 2),
    status String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree() ORDER BY id;

INSERT INTO test_db.orders (id, user_id, amount, status) VALUES
    (1, 1, 100.50, 'completed'),
    (2, 1, 200.00, 'completed'),
    (3, 2, 150.75, 'pending'),
    (4, 3, 300.00, 'completed');
EOF
        
        echo "Test data created successfully!"
        exit 0
    fi
    sleep 1
done

# Server failed to start - show error log
echo "ERROR: Failed to start ClickHouse server within ${MAX_WAIT} seconds" >&2
echo "Error log contents:" >&2
cat "${CH_LOG}/clickhouse-server.err.log" 2>/dev/null || echo "(no error log found)" >&2
exit 1
