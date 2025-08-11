import pytest
import pytest_socket
import requests


def test_socket_disabled() -> None:
    """This test should fail."""
    with pytest.raises(pytest_socket.SocketBlockedError):
        # Ignore S113 since we don't need a timeout here as the request
        # should fail immediately
        requests.get("https://www.example.com")  # noqa: S113
