from typing import Any, Callable, Dict, Optional
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler


def _default_approve(_input: str) -> bool:
    msg = (
        "Do you approve of the following input? "
        "Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no."
    )
    msg += "\n\n" + _input + "\n"
    resp = input(msg)
    return resp.lower() in ("yes", "y")


def _default_true(_: Dict[str, Any]) -> bool:
    return True


class HumanRejectedException(Exception):
    """Exception to raise when a person manually review and rejects a value."""


class HumanApprovalCallbackHandler(BaseCallbackHandler):
    """Callback for manually validating values."""

    raise_error: bool = True

    def __init__(
        self,
        approve: Callable[[Any], bool] = _default_approve,
        should_check: Callable[[Dict[str, Any]], bool] = _default_true,
    ):
        self._approve = approve
        self._should_check = should_check

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if self._should_check(serialized) and not self._approve(input_str):
            raise HumanRejectedException(
                f"Inputs {input_str} to tool {serialized} were rejected."
            )
