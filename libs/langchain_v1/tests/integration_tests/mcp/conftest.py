import multiprocessing
import socket
import time
from collections.abc import Generator

import pytest
import pytest_socket

from tests.integration_tests.mcp.utils import run_server


@pytest.fixture
def websocket_server_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    raise ValueError("Free port not found!")


@pytest.fixture
def websocket_server(websocket_server_port: int) -> Generator[None, None, None]:
    proc = multiprocessing.Process(
        target=run_server,
        kwargs={"server_port": websocket_server_port},
        daemon=True,
    )
    proc.start()

    # Wait for server to be running
    max_attempts = 20
    attempt = 0

    while attempt < max_attempts:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("127.0.0.1", websocket_server_port))
                break
        except ConnectionRefusedError:
            time.sleep(0.1)
            attempt += 1
    else:
        raise RuntimeError(f"Server failed to start after {max_attempts} attempts")

    yield

    # Signal the server to stop
    proc.kill()
    proc.join(timeout=2)
    if proc.is_alive():
        raise RuntimeError("Server process is still alive after attempting to terminate it")


@pytest.fixture
def socket_enabled():
    """Temporarily enable socket connections for websocket tests."""
    try:
        pytest_socket.enable_socket()
        previous_state = pytest_socket.socket_allow_hosts()
        # Only allow connections to localhost
        pytest_socket.socket_allow_hosts(["127.0.0.1", "localhost"], allow_unix_socket=True)
        yield
    finally:
        # Restore previous state
        pytest_socket.socket_allow_hosts(previous_state)
