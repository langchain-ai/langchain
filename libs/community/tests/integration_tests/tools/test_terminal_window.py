import time

import libtmux
import pytest

from langchain_community.tools.terminal_window import (
    TerminalBottomCaptureTool,
    TerminalLiteralInputTool,
    TerminalSpecialInputTool,
    TerminalTopCaptureTool,
)

def test_terminal_bottom_input_capture(monkeypatch, server):
    monkeypatch.setattr(libtmux, "Server", lambda: server)
    TerminalLiteralInputTool().run("echo hello world")
    assert "hello world" in TerminalBottomCaptureTool().run("2")

def test_terminal_special_input_capture(monkeypatch, server):
    monkeypatch.setattr(libtmux, "Server", lambda: server)
    TerminalSpecialInputTool().run("C-c")
    assert "^C" in TerminalBottomCaptureTool().run("2")

def test_terminal_interrupt(monkeypatch, server):
    monkeypatch.setattr(libtmux, "Server", lambda: server)
    TerminalLiteralInputTool().run("sleep 2 && echo hi")
    time.sleep(0.5)
    TerminalSpecialInputTool().run("C-c")
    time.sleep(2)
    output = TerminalBottomCaptureTool().run("2")
    assert "hi" not in output
    assert "^C" in output
