from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar, Union
from uuid import UUID, uuid4

from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseMessage,
    Document,
    LLMResult,
    get_buffer_string,
)
from langchain.schema.callbacks.base import (
    BaseCallbackHandler,
    BaseCallbackManager,
    ChainManagerMixin,
    LLMManagerMixin,
    RetrieverManagerMixin,
    RunManagerMixin,
    ToolManagerMixin,
)

logger = logging.getLogger(__name__)


def _handle_event(
    handlers: List[BaseCallbackHandler],
    event_name: str,
    ignore_condition_name: Optional[str],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Generic event handler for CallbackManager."""
    message_strings: Optional[List[str]] = None
    for handler in handlers:
        try:
            if ignore_condition_name is None or not getattr(
                handler, ignore_condition_name
            ):
                getattr(handler, event_name)(*args, **kwargs)
        except NotImplementedError as e:
            if event_name == "on_chat_model_start":
                if message_strings is None:
                    message_strings = [get_buffer_string(m) for m in args[1]]
                _handle_event(
                    [handler],
                    "on_llm_start",
                    "ignore_llm",
                    args[0],
                    message_strings,
                    *args[2:],
                    **kwargs,
                )
            else:
                logger.warning(
                    f"Error in {handler.__class__.__name__}.{event_name} callback: {e}"
                )
        except Exception as e:
            logger.warning(
                f"Error in {handler.__class__.__name__}.{event_name} callback: {e}"
            )
            if handler.raise_error:
                raise e


async def _ahandle_event_for_handler(
    handler: BaseCallbackHandler,
    event_name: str,
    ignore_condition_name: Optional[str],
    *args: Any,
    **kwargs: Any,
) -> None:
    try:
        if ignore_condition_name is None or not getattr(handler, ignore_condition_name):
            event = getattr(handler, event_name)
            if asyncio.iscoroutinefunction(event):
                await event(*args, **kwargs)
            else:
                if handler.run_inline:
                    event(*args, **kwargs)
                else:
                    await asyncio.get_event_loop().run_in_executor(
                        None, functools.partial(event, *args, **kwargs)
                    )
    except NotImplementedError as e:
        if event_name == "on_chat_model_start":
            message_strings = [get_buffer_string(m) for m in args[1]]
            await _ahandle_event_for_handler(
                handler,
                "on_llm_start",
                "ignore_llm",
                args[0],
                message_strings,
                *args[2:],
                **kwargs,
            )
        else:
            logger.warning(
                f"Error in {handler.__class__.__name__}.{event_name} callback: {e}"
            )
    except Exception as e:
        logger.warning(
            f"Error in {handler.__class__.__name__}.{event_name} callback: {e}"
        )
        if handler.raise_error:
            raise e


async def _ahandle_event(
    handlers: List[BaseCallbackHandler],
    event_name: str,
    ignore_condition_name: Optional[str],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Generic event handler for AsyncCallbackManager."""
    for handler in [h for h in handlers if h.run_inline]:
        await _ahandle_event_for_handler(
            handler, event_name, ignore_condition_name, *args, **kwargs
        )
    await asyncio.gather(
        *(
            _ahandle_event_for_handler(
                handler, event_name, ignore_condition_name, *args, **kwargs
            )
            for handler in handlers
            if not handler.run_inline
        )
    )


BRM = TypeVar("BRM", bound="BaseRunManager")


class BaseRunManager(RunManagerMixin):
    """Base class for run manager (a bound callback manager)."""

    def __init__(
        self,
        *,
        run_id: UUID,
        handlers: List[BaseCallbackHandler],
        inheritable_handlers: List[BaseCallbackHandler],
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        inheritable_tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inheritable_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the run manager.

        Args:
            run_id (UUID): The ID of the run.
            handlers (List[BaseCallbackHandler]): The list of handlers.
            inheritable_handlers (List[BaseCallbackHandler]):
                The list of inheritable handlers.
            parent_run_id (UUID, optional): The ID of the parent run.
                Defaults to None.
            tags (Optional[List[str]]): The list of tags.
            inheritable_tags (Optional[List[str]]): The list of inheritable tags.
            metadata (Optional[Dict[str, Any]]): The metadata.
            inheritable_metadata (Optional[Dict[str, Any]]): The inheritable metadata.
        """
        self.run_id = run_id
        self.handlers = handlers
        self.inheritable_handlers = inheritable_handlers
        self.parent_run_id = parent_run_id
        self.tags = tags or []
        self.inheritable_tags = inheritable_tags or []
        self.metadata = metadata or {}
        self.inheritable_metadata = inheritable_metadata or {}

    @classmethod
    def get_noop_manager(cls: Type[BRM]) -> BRM:
        """Return a manager that doesn't perform any operations.

        Returns:
            BaseRunManager: The noop manager.
        """
        return cls(
            run_id=uuid4(),
            handlers=[],
            inheritable_handlers=[],
            tags=[],
            inheritable_tags=[],
            metadata={},
            inheritable_metadata={},
        )


class RunManager(BaseRunManager):
    """Sync Run Manager."""

    def on_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> Any:
        """Run when text is received.

        Args:
            text (str): The received text.

        Returns:
            Any: The result of the callback.
        """
        _handle_event(
            self.handlers,
            "on_text",
            None,
            text,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class ParentRunManager(RunManager):
    """Sync Parent Run Manager."""

    def get_child(self, tag: Optional[str] = None) -> CallbackManager:
        """Get a child callback manager.

        Args:
            tag (str, optional): The tag for the child callback manager.
                Defaults to None.

        Returns:
            CallbackManager: The child callback manager.
        """
        manager = CallbackManager(handlers=[], parent_run_id=self.run_id)
        manager.set_handlers(self.inheritable_handlers)
        manager.add_tags(self.inheritable_tags)
        manager.add_metadata(self.inheritable_metadata)
        if tag is not None:
            manager.add_tags([tag], False)
        return manager


class AsyncRunManager(BaseRunManager):
    """Async Run Manager."""

    async def on_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> Any:
        """Run when text is received.

        Args:
            text (str): The received text.

        Returns:
            Any: The result of the callback.
        """
        await _ahandle_event(
            self.handlers,
            "on_text",
            None,
            text,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class AsyncParentRunManager(AsyncRunManager):
    """Async Parent Run Manager."""

    def get_child(self, tag: Optional[str] = None) -> AsyncCallbackManager:
        """Get a child callback manager.

        Args:
            tag (str, optional): The tag for the child callback manager.
                Defaults to None.

        Returns:
            AsyncCallbackManager: The child callback manager.
        """
        manager = AsyncCallbackManager(handlers=[], parent_run_id=self.run_id)
        manager.set_handlers(self.inheritable_handlers)
        manager.add_tags(self.inheritable_tags)
        manager.add_metadata(self.inheritable_metadata)
        if tag is not None:
            manager.add_tags([tag], False)
        return manager


class CallbackManagerForLLMRun(RunManager, LLMManagerMixin):
    """Callback manager for LLM run."""

    def on_llm_new_token(
        self,
        token: str,
        **kwargs: Any,
    ) -> None:
        """Run when LLM generates a new token.

        Args:
            token (str): The new token.
        """
        _handle_event(
            self.handlers,
            "on_llm_new_token",
            "ignore_llm",
            token=token,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running.

        Args:
            response (LLMResult): The LLM result.
        """
        _handle_event(
            self.handlers,
            "on_llm_end",
            "ignore_llm",
            response,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors.

        Args:
            error (Exception or KeyboardInterrupt): The error.
        """
        _handle_event(
            self.handlers,
            "on_llm_error",
            "ignore_llm",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class AsyncCallbackManagerForLLMRun(AsyncRunManager, LLMManagerMixin):
    """Async callback manager for LLM run."""

    async def on_llm_new_token(
        self,
        token: str,
        **kwargs: Any,
    ) -> None:
        """Run when LLM generates a new token.

        Args:
            token (str): The new token.
        """
        await _ahandle_event(
            self.handlers,
            "on_llm_new_token",
            "ignore_llm",
            token,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running.

        Args:
            response (LLMResult): The LLM result.
        """
        await _ahandle_event(
            self.handlers,
            "on_llm_end",
            "ignore_llm",
            response,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    async def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors.

        Args:
            error (Exception or KeyboardInterrupt): The error.
        """
        await _ahandle_event(
            self.handlers,
            "on_llm_error",
            "ignore_llm",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class CallbackManagerForChainRun(ParentRunManager, ChainManagerMixin):
    """Callback manager for chain run."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running.

        Args:
            outputs (Dict[str, Any]): The outputs of the chain.
        """
        _handle_event(
            self.handlers,
            "on_chain_end",
            "ignore_chain",
            outputs,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        """Run when chain errors.

        Args:
            error (Exception or KeyboardInterrupt): The error.
        """
        _handle_event(
            self.handlers,
            "on_chain_error",
            "ignore_chain",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run when agent action is received.

        Args:
            action (AgentAction): The agent action.

        Returns:
            Any: The result of the callback.
        """
        _handle_event(
            self.handlers,
            "on_agent_action",
            "ignore_agent",
            action,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run when agent finish is received.

        Args:
            finish (AgentFinish): The agent finish.

        Returns:
            Any: The result of the callback.
        """
        _handle_event(
            self.handlers,
            "on_agent_finish",
            "ignore_agent",
            finish,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class AsyncCallbackManagerForChainRun(AsyncParentRunManager, ChainManagerMixin):
    """Async callback manager for chain run."""

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running.

        Args:
            outputs (Dict[str, Any]): The outputs of the chain.
        """
        await _ahandle_event(
            self.handlers,
            "on_chain_end",
            "ignore_chain",
            outputs,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    async def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        """Run when chain errors.

        Args:
            error (Exception or KeyboardInterrupt): The error.
        """
        await _ahandle_event(
            self.handlers,
            "on_chain_error",
            "ignore_chain",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run when agent action is received.

        Args:
            action (AgentAction): The agent action.

        Returns:
            Any: The result of the callback.
        """
        await _ahandle_event(
            self.handlers,
            "on_agent_action",
            "ignore_agent",
            action,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run when agent finish is received.

        Args:
            finish (AgentFinish): The agent finish.

        Returns:
            Any: The result of the callback.
        """
        await _ahandle_event(
            self.handlers,
            "on_agent_finish",
            "ignore_agent",
            finish,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class CallbackManagerForToolRun(ParentRunManager, ToolManagerMixin):
    """Callback manager for tool run."""

    def on_tool_end(
        self,
        output: str,
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running.

        Args:
            output (str): The output of the tool.
        """
        _handle_event(
            self.handlers,
            "on_tool_end",
            "ignore_agent",
            output,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        """Run when tool errors.

        Args:
            error (Exception or KeyboardInterrupt): The error.
        """
        _handle_event(
            self.handlers,
            "on_tool_error",
            "ignore_agent",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class AsyncCallbackManagerForToolRun(AsyncParentRunManager, ToolManagerMixin):
    """Async callback manager for tool run."""

    async def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running.

        Args:
            output (str): The output of the tool.
        """
        await _ahandle_event(
            self.handlers,
            "on_tool_end",
            "ignore_agent",
            output,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    async def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        """Run when tool errors.

        Args:
            error (Exception or KeyboardInterrupt): The error.
        """
        await _ahandle_event(
            self.handlers,
            "on_tool_error",
            "ignore_agent",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class CallbackManagerForRetrieverRun(ParentRunManager, RetrieverManagerMixin):
    """Callback manager for retriever run."""

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        **kwargs: Any,
    ) -> None:
        """Run when retriever ends running."""
        _handle_event(
            self.handlers,
            "on_retriever_end",
            "ignore_retriever",
            documents,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_retriever_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        """Run when retriever errors."""
        _handle_event(
            self.handlers,
            "on_retriever_error",
            "ignore_retriever",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class AsyncCallbackManagerForRetrieverRun(
    AsyncParentRunManager,
    RetrieverManagerMixin,
):
    """Async callback manager for retriever run."""

    async def on_retriever_end(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> None:
        """Run when retriever ends running."""
        await _ahandle_event(
            self.handlers,
            "on_retriever_end",
            "ignore_retriever",
            documents,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    async def on_retriever_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        """Run when retriever errors."""
        await _ahandle_event(
            self.handlers,
            "on_retriever_error",
            "ignore_retriever",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )


class CallbackManager(BaseCallbackManager):
    """Callback manager that can be used to handle callbacks from langchain."""

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> List[CallbackManagerForLLMRun]:
        """Run when LLM starts running.

        Args:
            serialized (Dict[str, Any]): The serialized LLM.
            prompts (List[str]): The list of prompts.
            run_id (UUID, optional): The ID of the run. Defaults to None.

        Returns:
            List[CallbackManagerForLLMRun]: A callback manager for each
                prompt as an LLM run.
        """
        managers = []
        for prompt in prompts:
            run_id_ = uuid4()
            _handle_event(
                self.handlers,
                "on_llm_start",
                "ignore_llm",
                serialized,
                [prompt],
                run_id=run_id_,
                parent_run_id=self.parent_run_id,
                tags=self.tags,
                metadata=self.metadata,
                **kwargs,
            )

            managers.append(
                CallbackManagerForLLMRun(
                    run_id=run_id_,
                    handlers=self.handlers,
                    inheritable_handlers=self.inheritable_handlers,
                    parent_run_id=self.parent_run_id,
                    tags=self.tags,
                    inheritable_tags=self.inheritable_tags,
                    metadata=self.metadata,
                    inheritable_metadata=self.inheritable_metadata,
                )
            )

        return managers

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> List[CallbackManagerForLLMRun]:
        """Run when LLM starts running.

        Args:
            serialized (Dict[str, Any]): The serialized LLM.
            messages (List[List[BaseMessage]]): The list of messages.
            run_id (UUID, optional): The ID of the run. Defaults to None.

        Returns:
            List[CallbackManagerForLLMRun]: A callback manager for each
                list of messages as an LLM run.
        """

        managers = []
        for message_list in messages:
            run_id_ = uuid4()
            _handle_event(
                self.handlers,
                "on_chat_model_start",
                "ignore_chat_model",
                serialized,
                [message_list],
                run_id=run_id_,
                parent_run_id=self.parent_run_id,
                tags=self.tags,
                metadata=self.metadata,
                **kwargs,
            )

            managers.append(
                CallbackManagerForLLMRun(
                    run_id=run_id_,
                    handlers=self.handlers,
                    inheritable_handlers=self.inheritable_handlers,
                    parent_run_id=self.parent_run_id,
                    tags=self.tags,
                    inheritable_tags=self.inheritable_tags,
                    metadata=self.metadata,
                    inheritable_metadata=self.inheritable_metadata,
                )
            )

        return managers

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> CallbackManagerForChainRun:
        """Run when chain starts running.

        Args:
            serialized (Dict[str, Any]): The serialized chain.
            inputs (Dict[str, Any]): The inputs to the chain.
            run_id (UUID, optional): The ID of the run. Defaults to None.

        Returns:
            CallbackManagerForChainRun: The callback manager for the chain run.
        """
        if run_id is None:
            run_id = uuid4()

        _handle_event(
            self.handlers,
            "on_chain_start",
            "ignore_chain",
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            metadata=self.metadata,
            **kwargs,
        )

        return CallbackManagerForChainRun(
            run_id=run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> CallbackManagerForToolRun:
        """Run when tool starts running.

        Args:
            serialized (Dict[str, Any]): The serialized tool.
            input_str (str): The input to the tool.
            run_id (UUID, optional): The ID of the run. Defaults to None.
            parent_run_id (UUID, optional): The ID of the parent run. Defaults to None.

        Returns:
            CallbackManagerForToolRun: The callback manager for the tool run.
        """
        if run_id is None:
            run_id = uuid4()

        _handle_event(
            self.handlers,
            "on_tool_start",
            "ignore_agent",
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            metadata=self.metadata,
            **kwargs,
        )

        return CallbackManagerForToolRun(
            run_id=run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> CallbackManagerForRetrieverRun:
        """Run when retriever starts running."""
        if run_id is None:
            run_id = uuid4()

        _handle_event(
            self.handlers,
            "on_retriever_start",
            "ignore_retriever",
            serialized,
            query,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            metadata=self.metadata,
            **kwargs,
        )

        return CallbackManagerForRetrieverRun(
            run_id=run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    @classmethod
    def configure(
        cls,
        inheritable_callbacks: Callbacks = None,
        local_callbacks: Callbacks = None,
        verbose: bool = False,
        inheritable_tags: Optional[List[str]] = None,
        local_tags: Optional[List[str]] = None,
        inheritable_metadata: Optional[Dict[str, Any]] = None,
        local_metadata: Optional[Dict[str, Any]] = None,
    ) -> CallbackManager:
        """Configure the callback manager.

        Args:
            inheritable_callbacks (Optional[Callbacks], optional): The inheritable
                callbacks. Defaults to None.
            local_callbacks (Optional[Callbacks], optional): The local callbacks.
                Defaults to None.
            verbose (bool, optional): Whether to enable verbose mode. Defaults to False.
            inheritable_tags (Optional[List[str]], optional): The inheritable tags.
                Defaults to None.
            local_tags (Optional[List[str]], optional): The local tags.
                Defaults to None.
            inheritable_metadata (Optional[Dict[str, Any]], optional): The inheritable
                metadata. Defaults to None.
            local_metadata (Optional[Dict[str, Any]], optional): The local metadata.
                Defaults to None.

        Returns:
            CallbackManager: The configured callback manager.
        """

        return _configure(
            cls,
            inheritable_callbacks,
            local_callbacks,
            verbose,
            inheritable_tags,
            local_tags,
            inheritable_metadata,
            local_metadata,
        )


class AsyncCallbackManager(BaseCallbackManager):
    """Async callback manager that can be used to handle callbacks from LangChain."""

    @property
    def is_async(self) -> bool:
        """Return whether the handler is async."""
        return True

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> List[AsyncCallbackManagerForLLMRun]:
        """Run when LLM starts running.

        Args:
            serialized (Dict[str, Any]): The serialized LLM.
            prompts (List[str]): The list of prompts.
            run_id (UUID, optional): The ID of the run. Defaults to None.

        Returns:
            List[AsyncCallbackManagerForLLMRun]: The list of async
                callback managers, one for each LLM Run corresponding
                to each prompt.
        """

        tasks = []
        managers = []

        for prompt in prompts:
            run_id_ = uuid4()

            tasks.append(
                _ahandle_event(
                    self.handlers,
                    "on_llm_start",
                    "ignore_llm",
                    serialized,
                    [prompt],
                    run_id=run_id_,
                    parent_run_id=self.parent_run_id,
                    tags=self.tags,
                    metadata=self.metadata,
                    **kwargs,
                )
            )

            managers.append(
                AsyncCallbackManagerForLLMRun(
                    run_id=run_id_,
                    handlers=self.handlers,
                    inheritable_handlers=self.inheritable_handlers,
                    parent_run_id=self.parent_run_id,
                    tags=self.tags,
                    inheritable_tags=self.inheritable_tags,
                    metadata=self.metadata,
                    inheritable_metadata=self.inheritable_metadata,
                )
            )

        await asyncio.gather(*tasks)

        return managers

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> Any:
        """Run when LLM starts running.

        Args:
            serialized (Dict[str, Any]): The serialized LLM.
            messages (List[List[BaseMessage]]): The list of messages.
            run_id (UUID, optional): The ID of the run. Defaults to None.

        Returns:
            List[AsyncCallbackManagerForLLMRun]: The list of
                async callback managers, one for each LLM Run
                corresponding to each inner  message list.
        """
        tasks = []
        managers = []

        for message_list in messages:
            run_id_ = uuid4()

            tasks.append(
                _ahandle_event(
                    self.handlers,
                    "on_chat_model_start",
                    "ignore_chat_model",
                    serialized,
                    [message_list],
                    run_id=run_id_,
                    parent_run_id=self.parent_run_id,
                    tags=self.tags,
                    metadata=self.metadata,
                    **kwargs,
                )
            )

            managers.append(
                AsyncCallbackManagerForLLMRun(
                    run_id=run_id_,
                    handlers=self.handlers,
                    inheritable_handlers=self.inheritable_handlers,
                    parent_run_id=self.parent_run_id,
                    tags=self.tags,
                    inheritable_tags=self.inheritable_tags,
                    metadata=self.metadata,
                    inheritable_metadata=self.inheritable_metadata,
                )
            )

        await asyncio.gather(*tasks)
        return managers

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> AsyncCallbackManagerForChainRun:
        """Run when chain starts running.

        Args:
            serialized (Dict[str, Any]): The serialized chain.
            inputs (Dict[str, Any]): The inputs to the chain.
            run_id (UUID, optional): The ID of the run. Defaults to None.

        Returns:
            AsyncCallbackManagerForChainRun: The async callback manager
                for the chain run.
        """
        if run_id is None:
            run_id = uuid4()

        await _ahandle_event(
            self.handlers,
            "on_chain_start",
            "ignore_chain",
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            metadata=self.metadata,
            **kwargs,
        )

        return AsyncCallbackManagerForChainRun(
            run_id=run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> AsyncCallbackManagerForToolRun:
        """Run when tool starts running.

        Args:
            serialized (Dict[str, Any]): The serialized tool.
            input_str (str): The input to the tool.
            run_id (UUID, optional): The ID of the run. Defaults to None.
            parent_run_id (UUID, optional): The ID of the parent run.
                Defaults to None.

        Returns:
            AsyncCallbackManagerForToolRun: The async callback manager
                for the tool run.
        """
        if run_id is None:
            run_id = uuid4()

        await _ahandle_event(
            self.handlers,
            "on_tool_start",
            "ignore_agent",
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            metadata=self.metadata,
            **kwargs,
        )

        return AsyncCallbackManagerForToolRun(
            run_id=run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    async def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> AsyncCallbackManagerForRetrieverRun:
        """Run when retriever starts running."""
        if run_id is None:
            run_id = uuid4()

        await _ahandle_event(
            self.handlers,
            "on_retriever_start",
            "ignore_retriever",
            serialized,
            query,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            metadata=self.metadata,
            **kwargs,
        )

        return AsyncCallbackManagerForRetrieverRun(
            run_id=run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    @classmethod
    def configure(
        cls,
        inheritable_callbacks: Callbacks = None,
        local_callbacks: Callbacks = None,
        verbose: bool = False,
        inheritable_tags: Optional[List[str]] = None,
        local_tags: Optional[List[str]] = None,
        inheritable_metadata: Optional[Dict[str, Any]] = None,
        local_metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncCallbackManager:
        """Configure the async callback manager.

        Args:
            inheritable_callbacks (Optional[Callbacks], optional): The inheritable
                callbacks. Defaults to None.
            local_callbacks (Optional[Callbacks], optional): The local callbacks.
                Defaults to None.
            verbose (bool, optional): Whether to enable verbose mode. Defaults to False.
            inheritable_tags (Optional[List[str]], optional): The inheritable tags.
                Defaults to None.
            local_tags (Optional[List[str]], optional): The local tags.
                Defaults to None.
            inheritable_metadata (Optional[Dict[str, Any]], optional): The inheritable
                metadata. Defaults to None.
            local_metadata (Optional[Dict[str, Any]], optional): The local metadata.
                Defaults to None.

        Returns:
            AsyncCallbackManager: The configured async callback manager.
        """

        return _configure(
            cls,
            inheritable_callbacks,
            local_callbacks,
            verbose,
            inheritable_tags,
            local_tags,
            inheritable_metadata,
            local_metadata,
        )


CM = TypeVar("CM", CallbackManager, AsyncCallbackManager)


def _configure(
    callback_manager_cls: Type[CM],
    inheritable_callbacks: Callbacks = None,
    local_callbacks: Callbacks = None,
    verbose: bool = False,
    inheritable_tags: Optional[List[str]] = None,
    local_tags: Optional[List[str]] = None,
    inheritable_metadata: Optional[Dict[str, Any]] = None,
    local_metadata: Optional[Dict[str, Any]] = None,
) -> CM:
    """Configure the callback manager.

    Args:
        callback_manager_cls (Type[T]): The callback manager class.
        inheritable_callbacks (Optional[Callbacks], optional): The inheritable
            callbacks. Defaults to None.
        local_callbacks (Optional[Callbacks], optional): The local callbacks.
            Defaults to None.
        verbose (bool, optional): Whether to enable verbose mode. Defaults to False.
        inheritable_tags (Optional[List[str]], optional): The inheritable tags.
            Defaults to None.
        local_tags (Optional[List[str]], optional): The local tags. Defaults to None.
        inheritable_metadata (Optional[Dict[str, Any]], optional): The inheritable
            metadata. Defaults to None.
        local_metadata (Optional[Dict[str, Any]], optional): The local metadata.
            Defaults to None.

    Returns:
        T: The configured callback manager.
    """
    callback_manager = callback_manager_cls(handlers=[])
    if inheritable_callbacks or local_callbacks:
        if isinstance(inheritable_callbacks, list) or inheritable_callbacks is None:
            inheritable_callbacks_ = inheritable_callbacks or []
            callback_manager = callback_manager_cls(
                handlers=inheritable_callbacks_.copy(),
                inheritable_handlers=inheritable_callbacks_.copy(),
            )
        else:
            callback_manager = callback_manager_cls(
                handlers=inheritable_callbacks.handlers,
                inheritable_handlers=inheritable_callbacks.inheritable_handlers,
                parent_run_id=inheritable_callbacks.parent_run_id,
                tags=inheritable_callbacks.tags,
                inheritable_tags=inheritable_callbacks.inheritable_tags,
                metadata=inheritable_callbacks.metadata,
                inheritable_metadata=inheritable_callbacks.inheritable_metadata,
            )
        local_handlers_ = (
            local_callbacks
            if isinstance(local_callbacks, list)
            else (local_callbacks.handlers if local_callbacks else [])
        )
        for handler in local_handlers_:
            callback_manager.add_handler(handler, False)
    if inheritable_tags or local_tags:
        callback_manager.add_tags(inheritable_tags or [])
        callback_manager.add_tags(local_tags or [], False)
    if inheritable_metadata or local_metadata:
        callback_manager.add_metadata(inheritable_metadata or {})
        callback_manager.add_metadata(local_metadata or {}, False)

    try:
        from langchain.callbacks.context import add_handlers_from_context

        add_handlers_from_context(callback_manager, verbose)
    except ImportError:
        pass

    return callback_manager


Callbacks = Optional[Union[List[BaseCallbackHandler], BaseCallbackManager]]
