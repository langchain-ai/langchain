from typing import Any, Dict, Optional

from langchain_core.pydantic_v1 import BaseModel, root_validator


class TmuxPane(BaseModel):
    window_width: int = 140
    window_height: int = 24
    server: Any
    session_name: Optional[str]
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
        if "server" not in values or not values["server"]:
            values["server"] = libtmux.Server()
        if "session_name" not in values or not values["session_name"]:
            values["session_name"] = "langchain-utility-tmux"
        if values["server"].has_session(values["session_name"]):
            session = values["server"].sessions.get(session_name=values["session_name"])
        else:
            session = values["server"].new_session(session_name=values["session_name"])
        window = session.active_window
        if (
            window.width != values["window_width"]
            or window.height != values["window_height"]
        ):
            window.resize(width=values["window_width"], height=values["window_height"])
        values["pane"] = window.panes[0]
        return values

    def send_keys(self, *args: Any, **kwargs: Any) -> None:
        self.pane.send_keys(*args, **kwargs)

    def capture(self, *args: Any, **kwargs: Any) -> list[str]:
        return self.pane.capture_pane(*args, **kwargs)
