"""
Pytest configuration for datastore tests.

Provides fixtures for managing ClickHouse test server lifecycle.
The server is automatically started before integration tests and stopped after.
Tests will FAIL (not skip) if the server cannot be started.
"""

import os
import subprocess
import time
import pytest

# Test server configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SETUP_SCRIPT = os.path.join(SCRIPT_DIR, "setup_clickhouse_server.sh")
STOP_SCRIPT = os.path.join(SCRIPT_DIR, "stop_clickhouse_server.sh")
CH_DIR = os.path.join(SCRIPT_DIR, ".clickhouse")
CH_PID_FILE = os.path.join(CH_DIR, "clickhouse.pid")

DEFAULT_PORT = 19000


def is_clickhouse_running():
    """Check if ClickHouse test server is already running."""
    if not os.path.exists(CH_PID_FILE):
        return False

    try:
        with open(CH_PID_FILE, "r") as f:
            pid = int(f.read().strip())
        # Check if process is running
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def wait_for_clickhouse(host="localhost", port=DEFAULT_PORT, timeout=60):
    """Wait for ClickHouse server to be ready."""
    import chdb

    start_time = time.time()
    last_error = None
    while time.time() - start_time < timeout:
        try:
            sql = (
                f"SELECT 1 FROM remote('{host}:{port}', 'system', 'one', 'default', '')"
            )
            result = chdb.query(sql, output_format="DataFrame")
            if len(result) > 0:
                return True
        except Exception as e:
            last_error = e
        time.sleep(1)

    if last_error:
        print(f"Last connection error: {last_error}")
    return False


def start_clickhouse_server():
    """
    Start ClickHouse test server.

    Returns:
        (success: bool, error_message: str or None)
    """
    if is_clickhouse_running():
        print("ClickHouse server already running")
        return True, None

    if not os.path.exists(SETUP_SCRIPT):
        return False, f"Setup script not found: {SETUP_SCRIPT}"

    print("Starting ClickHouse test server...")
    try:
        result = subprocess.run(
            ["bash", SETUP_SCRIPT],
            capture_output=True,
            text=True,
            timeout=180,  # 3 minutes timeout for download + start
        )
        print(result.stdout)
        if result.returncode != 0:
            return (
                False,
                f"Setup script failed:\nstdout: {result.stdout}\nstderr: {result.stderr}",
            )
        return True, None
    except subprocess.TimeoutExpired:
        return False, "Timeout (180s) starting ClickHouse server"
    except Exception as e:
        return False, f"Error starting ClickHouse: {e}"


def stop_clickhouse_server():
    """Stop ClickHouse test server."""
    if not os.path.exists(STOP_SCRIPT):
        return

    print("Stopping ClickHouse test server...")
    try:
        result = subprocess.run(
            ["bash", STOP_SCRIPT], capture_output=True, text=True, timeout=30
        )
        print(result.stdout)
        if result.stderr:
            print(f"stderr: {result.stderr}")
    except Exception as e:
        print(f"Warning: Error stopping ClickHouse: {e}")

    # Verify server is actually stopped
    time.sleep(1)
    if is_clickhouse_running():
        print("Warning: Server may still be running after stop attempt")


@pytest.fixture(scope="session")
def clickhouse_server(request):
    """
    Session-scoped fixture that ensures ClickHouse test server is running.

    The server is started before the first test that needs it and
    stopped after all tests complete (unless KEEP_CLICKHOUSE=1).

    This fixture will FAIL (not skip) if the server cannot be started.

    Usage:
        def test_something(clickhouse_server):
            host, port = clickhouse_server
            # ... use the server ...
    """
    port = int(os.environ.get("TEST_CLICKHOUSE_PORT", DEFAULT_PORT))
    host = "localhost"

    # Check if external server is configured
    external_host = os.environ.get("TEST_CLICKHOUSE_HOST")
    if external_host:
        # Use external server, don't manage lifecycle
        if ":" in external_host:
            host, port_str = external_host.rsplit(":", 1)
            port = int(port_str)
        else:
            host = external_host

        # Verify external server is reachable
        if not wait_for_clickhouse(host, port, timeout=10):
            pytest.fail(f"External ClickHouse server at {host}:{port} is not reachable")

        yield (host, port)
        return

    # Start local test server
    success, error = start_clickhouse_server()
    if not success:
        pytest.fail(f"Failed to start ClickHouse test server: {error}")

    # Wait for server to be ready
    if not wait_for_clickhouse(host, port, timeout=60):
        # Try to get error log for debugging
        error_log = os.path.join(CH_DIR, "log", "clickhouse-server.err.log")
        error_content = ""
        if os.path.exists(error_log):
            try:
                with open(error_log, "r") as f:
                    error_content = f.read()[-2000:]  # Last 2000 chars
            except Exception:
                pass
        pytest.fail(
            f"ClickHouse server did not become ready within 60 seconds.\n"
            f"Error log:\n{error_content}"
        )

    yield (host, port)

    # Stop server unless KEEP_CLICKHOUSE is set
    if not os.environ.get("KEEP_CLICKHOUSE"):
        stop_clickhouse_server()


@pytest.fixture
def clickhouse_connection(clickhouse_server):
    """
    Fixture that provides a DataStore connected to the test ClickHouse server.

    Usage:
        def test_something(clickhouse_connection):
            ds = clickhouse_connection
            ds.databases()  # works
    """
    from datastore import DataStore

    host, port = clickhouse_server
    return DataStore.from_clickhouse(host=f"{host}:{port}", user="default", password="")
