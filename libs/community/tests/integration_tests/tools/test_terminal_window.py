import libtmux
import pytest

from langchain_community.tools.terminal_window import TerminalCaptureTool, TerminalLiteralInputTool

def test_terminal_input_capture(monkeypatch, server):
    monkeypatch.setattr(libtmux, "Server", lambda: server)
    input_tool = TerminalLiteralInputTool()
    input_tool.run(cmd="echo hello world")
    capture_tool = TerminalCaptureTool()
    assert "hello world" in capture_tool.run()