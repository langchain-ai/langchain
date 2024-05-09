from typing import Any, Dict

from langchain_core.pydantic_v1 import BaseModel, root_validator

DEFAULT_WINDOW_WIDTH = 120
DEFAULT_WINDOW_HEIGHT = 40

class TmuxPane(BaseModel):
    window_width: int = DEFAULT_WINDOW_WIDTH
    window_height: int = DEFAULT_WINDOW_HEIGHT
    server: Any
    session: Any
    pane: Any

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            import libtmux
        except ImportError:
            raise ImportError(
                "Could not import libtmux python package. "
                "Please install it with `pip install libtmux`."
            )
        if not values["server"]:
            values["server"] = libtmux.Server()
        session = values["server"].new_session(session_name="langchain-tool-tmux")
        window = session.active_window
        if window.width != values["window_width"] or window.height != values["window_height"]:
            window.resize(width=values["window_width"], height=values["window_height"])
        values["pane"] = window.panes[0]
        return values

    def send_keys(self, *args, **kwargs) -> None:
        self.pane.send_keys(*args, **kwargs)
    
    def capture(self) -> list[str]:
        return self.pane.cmd('capture-pane', '-p').stdout