from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel

from langchain.callbacks.base import BaseCallbackHandler


def _default_approval_fn(_input: Any) -> bool:
    msg = (
        "Do you approve of the following input? "
        "Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no."
    )
    msg += "\n\n" + str(_input) + "\n"
    resp = input(msg)
    return resp.lower() in ("yes", "y")


class InputRejectedException(Exception):
    """"""


class HumanApprovalCallbackHandler(BaseCallbackHandler, BaseModel):
    raise_error: bool = True
    get_approval: Callable[[], bool] = _default_approval_fn
    should_check: Callable[[Dict[str, Any]], bool] = lambda _: True

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if self.should_check(serialized) and not self.get_approval(prompts):
            prompts_str = "\n".join(prompts)
            raise InputRejectedException(
                f"Prompts {prompts_str} to model {serialized} were rejected."
            )

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if self.should_check(serialized) and not self.get_approval(inputs):
            inputs_str = "\n".join(f"{k}: {v}" for k, v in inputs.items())
            raise InputRejectedException(
                f"Inputs {inputs_str} to chain {serialized} were rejected."
            )

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if self.should_check(serialized) and not self.get_approval(input_str):
            raise InputRejectedException(
                f"Inputs {input_str} to tool {serialized} were rejected."
            )
