from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, Document, LLMResult
from langchain_core.utils import guard_import

if TYPE_CHECKING:
    import panel
    from panel.chat.feed import ChatFeed
    from panel.chat.interface import ChatInterface
    from panel.widgets import ChatMessage


def import_panel() -> panel:
    """Import panel"""
    return guard_import("panel")


class PanelCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        instance: ChatFeed | ChatInterface,
        user: str = "LangChain",
        avatar: str | None = None,
        step_width: int = 500,
    ) -> None:
        if BaseCallbackHandler is object:
            raise ImportError(
                "LangChainCallbackHandler requires `langchain` to be installed."
            )

        self._pn = import_panel()
        self._default_avatars = self._pn.chat.message.DEFAULT_AVATARS
        avatar = avatar or self._default_avatars["langchain"]

        self.instance = instance
        self._step: Any = None  # Can be None or panel.widgets.ChatMessage
        self._message: Optional[ChatMessage] = None
        self._active_user = user
        self._active_avatar = avatar
        self._disabled_state = self.instance.disabled
        self._is_streaming: Optional[bool] = None
        self._step_width = step_width

        self._input_user = user
        self._input_avatar = avatar

    def _update_active(self, avatar: str, label: str) -> None:
        if label == "None":
            return

        self._active_avatar = avatar
        if f"- {label}" not in self._active_user:
            self._active_user = f"{self._active_user} - {label}"

    def _reset_active(self) -> None:
        self._active_user = self._input_user
        self._active_avatar = self._input_avatar
        self._message = None

    def _on_start(self, serialized: dict[str, Any], kwargs: dict[str, Any]) -> None:
        model = kwargs.get("invocation_params", {}).get("model_name", "")
        self._is_streaming = serialized.get("kwargs", {}).get("streaming")
        messages = self.instance.objects
        if not messages or messages[-1].user != self._active_user:
            self._message = None
        if self._active_user and model not in self._active_user:
            self._active_user = f"{self._active_user} ({model})"

    def _stream(self, message: str) -> Optional[ChatMessage]:
        if message:
            return self.instance.stream(
                message,
                user=self._active_user,
                avatar=self._active_avatar,
                message=self._message,
            )
        return self._message

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self._on_start(serialized, kwargs)
        return super().on_llm_start(serialized, *args, **kwargs)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self._message = self._stream(token)
        return super().on_llm_new_token(token, **kwargs)

    def on_llm_end(self, response: LLMResult, *args: Any, **kwargs: Any) -> Any:
        if not self._is_streaming:
            self._stream(response.generations[0][0].text)

        self._reset_active()
        return super().on_llm_end(response, *args, **kwargs)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        return super().on_llm_error(
            error, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )

    def on_agent_action(self, action: AgentAction, *args: Any, **kwargs: Any) -> Any:
        return super().on_agent_action(action, *args, **kwargs)

    def on_agent_finish(self, finish: AgentFinish, *args: Any, **kwargs: Any) -> Any:
        return super().on_agent_finish(finish, *args, **kwargs)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self._update_active(self._default_avatars["tool"], serialized["name"])
        self._step = self.instance.add_step(
            title=f"Tool input: {input_str}",
            status="running",
            user=self._active_user,
            avatar=self._active_avatar,
            width=self._step_width,
            default_layout="column",
        )
        return super().on_tool_start(serialized, input_str, *args, **kwargs)

    def on_tool_end(self, output: str, *args: Any, **kwargs: Any) -> Any:
        if self._step is not None:
            self._step.stream(output)
            self._step.status = "success"
        self._reset_active()
        self._step = None
        return super().on_tool_end(output, *args, **kwargs)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        return super().on_tool_error(
            error, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self._disabled_state = self.instance.disabled
        self.instance.disabled = True
        return super().on_chain_start(serialized, inputs, *args, **kwargs)

    def on_chain_end(self, outputs: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        self.instance.disabled = self._disabled_state
        return super().on_chain_end(outputs, *args, **kwargs)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        return super().on_retriever_error(
            error, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        objects = [
            (f"Document {index}", document.page_content)
            for index, document in enumerate(documents)
        ]
        message = self._pn.Accordion(
            *objects, sizing_mode="stretch_width", margin=(10, 13, 10, 5)
        )
        self.instance.send(
            message,
            user="LangChain (retriever)",
            avatar=self._default_avatars["retriever"],
            respond=False,
        )
        return super().on_retriever_end(documents=documents, **kwargs)

    def on_text(self, text: str, **kwargs: Any) -> None:
        super().on_text(text, **kwargs)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[Any],
        **kwargs: Any,
    ) -> None:
        self._on_start(serialized, kwargs)
