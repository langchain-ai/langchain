from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.tmux import TmuxPane


class TerminalLiteralInputTool(BaseTool):
    """Tool to send literal keys to a terminal window"""

    name: str = "terminal_literal_input"
    description: str = (
        "Use this tool to send literal input to a terminal window. "
        "The string you provide will be sent to the terminal exactly"
        "For example, 'ls' will cause the 'l' key to be pressed, followed by 's'. "
        "Note that Enter will NOT be pressed by this tool. "
        "Use the terminal_special_input tool to press the Enter key if you need to. "
    )
    pane: TmuxPane = Field(default_factory=TmuxPane)

    def _run(
        self, literal_keys: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> None:
        self.pane.send_keys(literal_keys, literal=True, enter=False)


class TerminalSpecialInputTool(BaseTool):
    """Tool to send a special key or key combination to a terminal window"""

    name: str = "terminal_special_input"
    description: str = (
        "Use this tool to send a special key or key combination to a terminal window. "
        "tmux syntax is accepted. "
        "For example, 'Enter' will cause literal Enter to be pressed. "
        "C-c or ^C will cause Ctrl+C to be pressed. "
    )
    pane: TmuxPane = Field(default_factory=TmuxPane)

    def _run(
        self, special_key: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> None:
        self.pane.send_keys(special_key, literal=False, enter=False)


class TerminalBottomCaptureTool(BaseTool):
    """Tool to capture the output of the bottom N rows of a terminal window"""

    name: str = "terminal_bottom_capture"
    description: str = "Use this tool to see the bottom N rows of the terminal window."
    pane: TmuxPane = Field(default_factory=TmuxPane)

    def _run(
        self, n_bottom_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        # Ensure we can't exceed the window height
        n_bottom = min(int(n_bottom_str), self.pane.window_height)
        # Grab the last n_bottom rows of the pane contents
        n_bottom_rows = self.pane.capture()[-n_bottom:]
        # Pad with leading empty strings if necessary
        return "\n".join([""] * (n_bottom - len(n_bottom_rows)) + n_bottom_rows)


class TerminalTopCaptureTool(BaseTool):
    """Tool to capture the output of the top N rows of a terminal window"""

    name: str = "terminal_top_capture"
    description: str = (
        "Use this tool to see the top N rows of the terminal window."
        "Note that terminal output always starts from the bottom, so you'll only see "
        "anything in the top row when the entire window is full with output."
    )
    pane: TmuxPane = Field(default_factory=TmuxPane)

    def _run(
        self, n_top_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        # Ensure we can't exceed the window height
        n_top = min(int(n_top_str), self.pane.window_height)
        # Grab the pane contents
        pane_contents = self.pane.capture()
        # Pad with leading empty strings if necessary
        pane_contents = [""] * (
            self.pane.window_height - len(pane_contents)
        ) + pane_contents
        # Return the first n_top
        return "\n".join(pane_contents[0:n_top])
