import pytest
import pytest_socket


@pytest.fixture
def socket_enabled():
    """Temporarily enable socket connections to the local test servers."""
    try:
        pytest_socket.enable_socket()
        previous_state = pytest_socket.socket_allow_hosts()
        # Only allow connections to localhost
        pytest_socket.socket_allow_hosts(["127.0.0.1", "localhost"], allow_unix_socket=True)
        yield
    finally:
        # Restore previous state
        pytest_socket.socket_allow_hosts(previous_state)
