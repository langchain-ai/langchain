"""Callback handler for promptlayer."""
from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    LLMResult,
)

if TYPE_CHECKING:
    import promptlayer


def _lazy_import_promptlayer() -> promptlayer:
    """Lazy import promptlayer to avoid circular imports."""
    try:
        import promptlayer
    except ImportError:
        raise ImportError(
            "The PromptLayerCallbackHandler requires the promptlayer package. "
            " Please install it with `pip install promptlayer`."
        )
    return promptlayer


class PromptLayerCallbackHandler(BaseCallbackHandler):
    """Callback handler for promptlayer."""

    def __init__(
        self,
        pl_id_callback: Optional[Callable[..., Any]] = None,
        pl_tags: Optional[List[str]] = None,
    ) -> None:
        """Initialize the PromptLayerCallbackHandler."""
        _lazy_import_promptlayer()
        self.pl_id_callback = pl_id_callback
        self.pl_tags = pl_tags or []
        self.runs: Dict[UUID, Dict[str, Any]] = {}

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        self.runs[run_id] = {
            "messages": [self._create_message_dicts(m)[0] for m in messages],
            "invocation_params": kwargs.get("invocation_params", {}),
            "name": ".".join(serialized["id"]),
            "request_start_time": datetime.datetime.now().timestamp(),
            "tags": tags,
        }

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        self.runs[run_id] = {
            "prompts": prompts,
            "invocation_params": kwargs.get("invocation_params", {}),
            "name": ".".join(serialized["id"]),
            "request_start_time": datetime.datetime.now().timestamp(),
            "tags": tags,
        }

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        from promptlayer.utils import get_api_key, promptlayer_api_request

        run_info = self.runs.get(run_id, {})
        if not run_info:
            return
        run_info["request_end_time"] = datetime.datetime.now().timestamp()
        for i in range(len(response.generations)):
            generation = response.generations[i][0]

            resp = {
                "text": generation.text,
                "llm_output": response.llm_output,
            }
            model_params = run_info.get("invocation_params", {})
            is_chat_model = run_info.get("messages", None) is not None
            model_input = (
                run_info.get("messages", [])[i]
                if is_chat_model
                else [run_info.get("prompts", [])[i]]
            )
            model_response = (
                [self._convert_message_to_dict(generation.message)]
                if is_chat_model and isinstance(generation, ChatGeneration)
                else resp
            )

            pl_request_id = promptlayer_api_request(
                run_info.get("name"),
                "langchain",
                model_input,
                model_params,
                self.pl_tags,
                model_response,
                run_info.get("request_start_time"),
                run_info.get("request_end_time"),
                get_api_key(),
                return_pl_id=bool(self.pl_id_callback is not None),
                metadata={
                    "_langchain_run_id": str(run_id),
                    "_langchain_parent_run_id": str(parent_run_id),
                    "_langchain_tags": str(run_info.get("tags", [])),
                },
            )

            if self.pl_id_callback:
                self.pl_id_callback(pl_request_id)

    def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        if isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}
        elif isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        else:
            raise ValueError(f"Got unknown type {message}")
        if "name" in message.additional_kwargs:
            message_dict["name"] = message.additional_kwargs["name"]
        return message_dict

    def _create_message_dicts(
        self, messages: List[BaseMessage]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params: Dict[str, Any] = {}
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        return message_dicts, params
