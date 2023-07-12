"""A fake callback handler for testing purposes."""
from itertools import chain
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel

from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain.schema.messages import BaseMessage


class BaseFakeCallbackHandler(BaseModel):
    """Base fake callback handler for testing."""

    starts: int = 0
    ends: int = 0
    errors: int = 0
    text: int = 0
    ignore_llm_: bool = False
    ignore_chain_: bool = False
    ignore_agent_: bool = False
    ignore_retriever_: bool = False
    ignore_chat_model_: bool = False

    # to allow for similar callback handlers that are not technicall equal
    fake_id: Union[str, None] = None

    # add finer-grained counters for easier debugging of failing tests
    chain_starts: int = 0
    chain_ends: int = 0
    llm_starts: int = 0
    llm_ends: int = 0
    llm_streams: int = 0
    tool_starts: int = 0
    tool_ends: int = 0
    agent_actions: int = 0
    agent_ends: int = 0
    chat_model_starts: int = 0
    retriever_starts: int = 0
    retriever_ends: int = 0
    retriever_errors: int = 0


class BaseFakeCallbackHandlerMixin(BaseFakeCallbackHandler):
    """Base fake callback handler mixin for testing."""

    def on_llm_start_common(self) -> None:
        self.llm_starts += 1
        self.starts += 1

    def on_llm_end_common(self) -> None:
        self.llm_ends += 1
        self.ends += 1

    def on_llm_error_common(self) -> None:
        self.errors += 1

    def on_llm_new_token_common(self) -> None:
        self.llm_streams += 1

    def on_chain_start_common(self) -> None:
        ("CHAIN START")
        self.chain_starts += 1
        self.starts += 1

    def on_chain_end_common(self) -> None:
        self.chain_ends += 1
        self.ends += 1

    def on_chain_error_common(self) -> None:
        self.errors += 1

    def on_tool_start_common(self) -> None:
        self.tool_starts += 1
        self.starts += 1

    def on_tool_end_common(self) -> None:
        self.tool_ends += 1
        self.ends += 1

    def on_tool_error_common(self) -> None:
        self.errors += 1

    def on_agent_action_common(self) -> None:
        print("AGENT ACTION")
        self.agent_actions += 1
        self.starts += 1

    def on_agent_finish_common(self) -> None:
        self.agent_ends += 1
        self.ends += 1

    def on_chat_model_start_common(self) -> None:
        print("STARTING CHAT MODEL")
        self.chat_model_starts += 1
        self.starts += 1

    def on_text_common(self) -> None:
        self.text += 1

    def on_retriever_start_common(self) -> None:
        self.starts += 1
        self.retriever_starts += 1

    def on_retriever_end_common(self) -> None:
        self.ends += 1
        self.retriever_ends += 1

    def on_retriever_error_common(self) -> None:
        self.errors += 1
        self.retriever_errors += 1


class FakeCallbackHandler(BaseCallbackHandler, BaseFakeCallbackHandlerMixin):
    """Fake callback handler for testing."""

    @property
    def ignore_llm(self) -> bool:
        """Whether to ignore LLM callbacks."""
        return self.ignore_llm_

    @property
    def ignore_chain(self) -> bool:
        """Whether to ignore chain callbacks."""
        return self.ignore_chain_

    @property
    def ignore_agent(self) -> bool:
        """Whether to ignore agent callbacks."""
        return self.ignore_agent_

    @property
    def ignore_retriever(self) -> bool:
        """Whether to ignore retriever callbacks."""
        return self.ignore_retriever_

    def on_llm_start(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_llm_start_common()

    def on_llm_new_token(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_llm_new_token_common()

    def on_llm_end(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_llm_end_common()

    def on_llm_error(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_llm_error_common()

    def on_chain_start(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_chain_start_common()

    def on_chain_end(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_chain_end_common()

    def on_chain_error(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_chain_error_common()

    def on_tool_start(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_tool_start_common()

    def on_tool_end(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_tool_end_common()

    def on_tool_error(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_tool_error_common()

    def on_agent_action(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_agent_action_common()

    def on_agent_finish(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_agent_finish_common()

    def on_text(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_text_common()

    def on_retriever_start(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_retriever_start_common()

    def on_retriever_end(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_retriever_end_common()

    def on_retriever_error(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.on_retriever_error_common()

    def __deepcopy__(self, memo: dict) -> "FakeCallbackHandler":
        return self


class FakeCallbackHandlerWithChatStart(FakeCallbackHandler):
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        assert all(isinstance(m, BaseMessage) for m in chain(*messages))
        self.on_chat_model_start_common()


class FakeAsyncCallbackHandler(AsyncCallbackHandler, BaseFakeCallbackHandlerMixin):
    """Fake async callback handler for testing."""

    @property
    def ignore_llm(self) -> bool:
        """Whether to ignore LLM callbacks."""
        return self.ignore_llm_

    @property
    def ignore_chain(self) -> bool:
        """Whether to ignore chain callbacks."""
        return self.ignore_chain_

    @property
    def ignore_agent(self) -> bool:
        """Whether to ignore agent callbacks."""
        return self.ignore_agent_

    async def on_llm_start(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_llm_start_common()

    async def on_llm_new_token(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_llm_new_token_common()

    async def on_llm_end(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_llm_end_common()

    async def on_llm_error(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_llm_error_common()

    async def on_chain_start(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_chain_start_common()

    async def on_chain_end(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_chain_end_common()

    async def on_chain_error(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_chain_error_common()

    async def on_tool_start(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_tool_start_common()

    async def on_tool_end(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_tool_end_common()

    async def on_tool_error(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_tool_error_common()

    async def on_agent_action(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_agent_action_common()

    async def on_agent_finish(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_agent_finish_common()

    async def on_text(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.on_text_common()

    def __deepcopy__(self, memo: dict) -> "FakeAsyncCallbackHandler":
        return self
