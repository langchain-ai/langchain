from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_community.utilities.tmux import TmuxPane

class TerminalLiteralInputTool(BaseTool):
    """Tool to send literal keys to a terminal window, followed by Enter"""
    name: str = "terminal_literal_input"
    description: str = (
        "Use this tool to send literal input to a terminal window. "
        "The string you provide will be sent to the terminal exactly, followed by the Enter key. "
        "For example, 'ls' will cause the 'l' key to be pressed, followed by 's', followed by Enter."
    )
    session_name: str = None
    """Optional session name allows for different agents to have different terminal sessions if desired."""

    def _run(self, cmd: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> None:
        TmuxPane(session_name=self.session_name).send_keys(cmd, literal=True, enter=True)

class TerminalCaptureTool(BaseTool):
    """Tool to capture the output of a terminal window"""
    name: str = "terminal_capture"
    description: str = "Use this tool to see the entire contents of the terminal window."
    session_name: str = None
    """Optional session name allows for different agents to have different terminal sessions if desired."""

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> list[str]:
        return TmuxPane(session_name=self.session_name).capture()