import pytest
import pytest_socket
import requests


def test_socket_disabled() -> None:
    """This test should fail."""
    with pytest.raises(pytest_socket.SocketBlockedError):
        requests.get("https://www.example.com")
