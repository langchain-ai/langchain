from libtmux.server import Server
import pytest

from langchain_community.utilities.tmux import DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT, TmuxPane

@pytest.fixture(scope="function")
def pane(server: Server) -> TmuxPane:
    return TmuxPane(server=server)

def test_pane_window_default_dims(pane):
    assert pane.pane.window.width == str(DEFAULT_WINDOW_WIDTH)
    assert pane.pane.window.height == str(DEFAULT_WINDOW_HEIGHT)

def test_pane_send_keys_and_read_contents(pane):
    pane.send_keys("echo hello world")
    contents = pane.capture()
    assert "hello world" in "\n".join(contents)