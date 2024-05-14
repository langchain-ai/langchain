from typing import Any

from pytest import importorskip

from langchain_community.utilities.tmux import TmuxPane

libtmux = importorskip("libtmux")


def test_pane_using_server(server: Any) -> None:
    pane = TmuxPane(server=server)
    assert pane.server == server


def test_pane_window_default_dims(server: Any) -> None:
    pane = TmuxPane(server=server)
    assert pane.pane.window.width == "120"
    assert pane.pane.window.height == "40"


def test_pane_send_keys_and_read_contents(server: Any) -> None:
    pane = TmuxPane(server=server)
    pane.send_keys("echo hello world")
    assert "hello world" in "\n".join(pane.capture())


def test_create_same_pane_twice(server: Any) -> None:
    TmuxPane(server=server)
    TmuxPane(server=server)


def test_pane_send_keys_and_other_pane_read_contents(server: Any) -> None:
    p1 = TmuxPane(server=server)
    p1.send_keys("echo hello world")
    p2 = TmuxPane(server=server)
    assert "hello world" in "\n".join(p2.capture())


def test_pane_send_keys_and_different_session_pane_doesnt_see_contents(
    server: Any,
) -> None:
    p1 = TmuxPane(server=server)
    p1.send_keys("echo hello world")
    assert "hello world" in "\n".join(p1.capture())
    p2 = TmuxPane(server=server, session_name="s2")
    assert "hello world" not in "\n".join(p2.capture())


def test_pane_null_session_default(server: Any) -> None:
    pane = TmuxPane(server=server, session_name=None)
    assert pane.pane.session.name == "langchain-utility-tmux"
