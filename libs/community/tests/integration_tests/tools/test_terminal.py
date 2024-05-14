import time
from typing import Any

from pytest import importorskip

from langchain_community.tools import (
    TerminalBottomCaptureTool,
    TerminalLiteralInputTool,
    TerminalSpecialInputTool,
    TerminalTopCaptureTool,
)
from langchain_community.utilities import TmuxPane

libtmux = importorskip("libtmux")


def test_terminal_bottom_input_capture(server: Any) -> None:
    pane = TmuxPane(server=server)
    TerminalLiteralInputTool(pane=pane).run("echo hello world")
    assert "hello world" in TerminalBottomCaptureTool(pane=pane).run("2")


def test_terminal_special_input_capture(server: Any) -> None:
    pane = TmuxPane(server=server)
    TerminalSpecialInputTool(pane=pane).run("C-c")
    assert "^C" in TerminalBottomCaptureTool(pane=pane).run("2")


def test_terminal_top_capture(server: Any) -> None:
    pane = TmuxPane(server=server, window_height=17)
    TerminalLiteralInputTool(pane=pane).run("for ((i=1;i<17;i++)); do echo $i; done")
    TerminalSpecialInputTool(pane=pane).run("Enter")
    time.sleep(0.5)
    assert TerminalTopCaptureTool(pane=pane).run("2") == "1\n2"


def test_terminal_interrupt(server: Any) -> None:
    pane = TmuxPane(server=server)
    TerminalLiteralInputTool(pane=pane).run("sleep 2 && echo hi")
    TerminalSpecialInputTool(pane=pane).run("Enter")
    time.sleep(0.5)
    TerminalSpecialInputTool(pane=pane).run("C-c")
    time.sleep(2)
    output = TerminalBottomCaptureTool(pane=pane).run("2")
    assert "hi" not in output
    assert "^C" in output
