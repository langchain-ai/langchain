"""Tests for `FileCallbackHandler`."""

import pathlib
import re

from langchain_core.agents import AgentAction
from langchain_core.callbacks import FileCallbackHandler

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _colored_segments(text: str) -> list[str]:
    """Return the visible text of each segment that carries an ANSI color."""
    return [
        _ANSI.sub("", segment)
        for segment in re.findall(r"\x1b\[[\d;]*m.*?\x1b\[0m", text, flags=re.DOTALL)
    ]


def test_on_tool_end_uses_the_color(tmp_path: pathlib.Path) -> None:
    """Tool output is colored, both from `self.color` and from an override."""
    log = tmp_path / "output.log"
    with FileCallbackHandler(str(log), color="green") as handler:
        handler.on_tool_end("DEFAULT")
        handler.on_tool_end("OVERRIDE", color="red")

    written = log.read_text()
    assert _colored_segments(written) == ["DEFAULT", "OVERRIDE"]
    # the override is a different color from the handler default
    assert "\x1b[32;1m" in written
    assert "\x1b[31;1m" in written


def test_on_tool_end_matches_its_siblings(tmp_path: pathlib.Path) -> None:
    """The other handlers already color their output; `on_tool_end` agrees."""
    log = tmp_path / "output.log"
    action = AgentAction(tool="tool", tool_input="input", log="ACTION")
    with FileCallbackHandler(str(log), color="green") as handler:
        handler.on_text("TEXT")
        handler.on_agent_action(action)
        handler.on_tool_end("TOOL")

    assert _colored_segments(log.read_text()) == ["TEXT", "ACTION", "TOOL"]


def test_on_tool_end_keeps_its_prefixes(tmp_path: pathlib.Path) -> None:
    """The prefixes are unchanged and stay outside the colored segment."""
    log = tmp_path / "output.log"
    with FileCallbackHandler(str(log), color="green") as handler:
        handler.on_tool_end(
            "TOOL", observation_prefix="Observation:", llm_prefix="Thought:"
        )

    written = log.read_text()
    assert "Observation:" in _ANSI.sub("", written)
    assert "Thought:" in _ANSI.sub("", written)
    assert _colored_segments(written) == ["TOOL"]
