from __future__ import annotations

import asyncio
import copy
import functools
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from langchain.callbacks.base import (
    BaseCallbackHandler,
    BaseCallbackManager,
    ChainManagerMixin,
    LLMManagerMixin,
    RunManagerMixin,
    ToolManagerMixin,
)
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.schema import AgentAction, AgentFinish, LLMResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()],
)

Callbacks = Optional[Union[List[BaseCallbackHandler], BaseCallbackManager]]


def _handle_event(
    handlers: List[BaseCallbackHandler],
    event_name: str,
    ignore_condition_name: Optional[str],
    *args: Any,
    **kwargs: Any,
) -> None:
    for handler in handlers:
        try:
            if ignore_condition_name is None or not getattr(
                handler, ignore_condition_name
            ):
                getattr(handler, event_name)(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {event_name} callback: {e}")


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
                await asyncio.get_event_loop().run_in_executor(
                    None, functools.partial(event, *args, **kwargs)
                )
    except Exception as e:
        logging.error(f"Error in {event_name} callback: {e}")


async def _ahandle_event(
    handlers: List[BaseCallbackHandler],
    event_name: str,
    ignore_condition_name: Optional[str],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Generic event handler for AsyncCallbackManager."""
    await asyncio.gather(
        *(
            _ahandle_event_for_handler(
                handler, event_name, ignore_condition_name, *args, **kwargs
            )
            for handler in handlers
        )
    )


class BaseRunManager(RunManagerMixin):
    """Base class for run manager (a bound callback manager)."""

    def __init__(
        self,
        run_id: str,
        handlers: List[BaseCallbackHandler],
        inheritable_handlers: List[BaseCallbackHandler],
        parent_run_id: str,
    ) -> None:
        """Initialize run manager."""
        self.run_id = run_id
        self.handlers = handlers
        self.inheritable_handlers = inheritable_handlers
        self.parent_run_id = parent_run_id


class RunManager(BaseRunManager):
    """Sync Run Manager."""

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run when text is received."""
        _handle_event(self.handlers, "on_text", None, False, text, **kwargs)


class AsyncRunManager(BaseRunManager):
    """Async Run Manager."""

    async def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run when text is received."""
        await _ahandle_event(self.handlers, "on_text", None, False, text, **kwargs)


class CallbackManagerForLLMRun(RunManager, LLMManagerMixin):
    """Callback manager for LLM run."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run when LLM generates a new token."""
        _handle_event(
            self.handlers,
            "on_llm_new_token",
            "ignore_llm",
            token,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        _handle_event(
            self.handlers,
            "on_llm_end",
            "ignore_llm",
            response,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors."""
        _handle_event(
            self.handlers,
            "on_llm_error",
            "ignore_llm",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )


class AsyncCallbackManagerForLLMRun(AsyncRunManager, LLMManagerMixin):
    """Async callback manager for LLM run."""

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run when LLM generates a new token."""
        await _ahandle_event(
            self.handlers,
            "on_llm_new_token",
            "ignore_llm",
            token,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        await _ahandle_event(
            self.handlers,
            "on_llm_end",
            "ignore_llm",
            response,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    async def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors."""
        await _ahandle_event(
            self.handlers,
            "on_llm_error",
            "ignore_llm",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )


class CallbackManagerForChainRun(RunManager, ChainManagerMixin):
    """Callback manager for chain run."""

    def get_child(self) -> CallbackManager:
        """Get a child callback manager."""
        manager = CallbackManager([], parent_run_id=self.run_id)
        manager.set_handlers(self.inheritable_handlers)
        return manager

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        _handle_event(
            self.handlers,
            "on_chain_end",
            "ignore_chain",
            outputs,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        """Run when chain errors."""
        _handle_event(
            self.handlers,
            "on_chain_error",
            "ignore_chain",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run when agent action is received."""
        _handle_event(
            self.handlers,
            "on_agent_action",
            "ignore_agent",
            action,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run when agent finish is received."""
        _handle_event(
            self.handlers,
            "on_agent_finish",
            "ignore_agent",
            finish,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )


class AsyncCallbackManagerForChainRun(AsyncRunManager, ChainManagerMixin):
    """Async callback manager for chain run."""

    def get_child(self) -> AsyncCallbackManager:
        """Get a child callback manager."""
        manager = AsyncCallbackManager([], parent_run_id=self.run_id)
        manager.set_handlers(self.inheritable_handlers)
        return manager

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        await _ahandle_event(
            self.handlers,
            "on_chain_end",
            "ignore_chain",
            outputs,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    async def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        """Run when chain errors."""
        await _ahandle_event(
            self.handlers,
            "on_chain_error",
            "ignore_chain",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run when agent action is received."""
        await _ahandle_event(
            self.handlers,
            "on_agent_action",
            "ignore_agent",
            action,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run when agent finish is received."""
        await _ahandle_event(
            self.handlers,
            "on_agent_finish",
            "ignore_agent",
            finish,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )


class CallbackManagerForToolRun(RunManager, ToolManagerMixin):
    """Callback manager for tool run."""

    def get_child(self) -> CallbackManager:
        """Get a child callback manager."""
        manager = CallbackManager([], parent_run_id=self.run_id)
        manager.set_handlers(self.inheritable_handlers)
        return manager

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        _handle_event(
            self.handlers,
            "on_tool_end",
            "ignore_agent",
            output,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        """Run when tool errors."""
        _handle_event(
            self.handlers,
            "on_tool_error",
            "ignore_agent",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )


class AsyncCallbackManagerForToolRun(AsyncRunManager, ToolManagerMixin):
    """Async callback manager for tool run."""

    def get_child(self) -> AsyncCallbackManager:
        """Get a child callback manager."""
        manager = AsyncCallbackManager([], parent_run_id=self.run_id)
        manager.set_handlers(self.inheritable_handlers)
        return manager

    async def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        await _ahandle_event(
            self.handlers,
            "on_tool_end",
            "ignore_agent",
            output,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    async def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        """Run when tool errors."""
        await _ahandle_event(
            self.handlers,
            "on_tool_error",
            "ignore_agent",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )


class CallbackManager(BaseCallbackManager):
    """Callback manager that can be used to handle callbacks from langchain."""

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CallbackManagerForLLMRun:
        """Run when LLM starts running."""
        if run_id is None:
            run_id = uuid.uuid4()

        _handle_event(
            self.handlers,
            "on_llm_start",
            "ignore_llm",
            serialized,
            prompts,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

        return CallbackManagerForLLMRun(
            run_id, self.handlers, self.inheritable_handlers, self.parent_run_id
        )

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CallbackManagerForChainRun:
        """Run when chain starts running."""
        if run_id is None:
            run_id = uuid.uuid4()

        _handle_event(
            self.handlers,
            "on_chain_start",
            "ignore_chain",
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

        return CallbackManagerForChainRun(
            run_id, self.handlers, self.inheritable_handlers, self.parent_run_id
        )

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        run_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CallbackManagerForToolRun:
        """Run when tool starts running."""
        if run_id is None:
            run_id = uuid.uuid4()

        _handle_event(
            self.handlers,
            "on_tool_start",
            "ignore_agent",
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )

        return CallbackManagerForToolRun(
            run_id, self.handlers, self.inheritable_handlers, self.parent_run_id
        )

    @classmethod
    def configure(
        cls,
        inheritable_callbacks: Optional[
            Union[BaseCallbackManager, List[BaseCallbackHandler]]
        ] = None,
        local_callbacks: Optional[
            Union[BaseCallbackManager, List[BaseCallbackHandler]]
        ] = None,
        verbose: bool = False,
    ) -> Optional[BaseCallbackManager]:
        """Configure the callback manager."""
        return _configure(cls, inheritable_callbacks, local_callbacks, verbose)


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
        run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncCallbackManagerForLLMRun:
        """Run when LLM starts running."""
        if run_id is None:
            run_id = uuid.uuid4()

        await _ahandle_event(
            self.handlers,
            "on_llm_start",
            "ignore_llm",
            serialized,
            prompts,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

        return AsyncCallbackManagerForLLMRun(
            run_id, self.handlers, self.inheritable_handlers, self.parent_run_id
        )

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncCallbackManagerForChainRun:
        """Run when chain starts running."""
        if run_id is None:
            run_id = uuid.uuid4()

        await _ahandle_event(
            self.handlers,
            "on_chain_start",
            "ignore_chain",
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

        return AsyncCallbackManagerForChainRun(
            run_id, self.handlers, self.inheritable_handlers, self.parent_run_id
        )

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        run_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncCallbackManagerForToolRun:
        """Run when tool starts running."""
        if run_id is None:
            run_id = uuid.uuid4()

        await _ahandle_event(
            self.handlers,
            "on_tool_start",
            "ignore_agent",
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )

        return AsyncCallbackManagerForToolRun(
            run_id, self.handlers, self.inheritable_handlers, self.parent_run_id
        )

    @classmethod
    def configure(
        cls,
        inheritable_callbacks: Optional[
            Union[BaseCallbackManager, List[BaseCallbackHandler]]
        ] = None,
        local_callbacks: Optional[
            Union[BaseCallbackManager, List[BaseCallbackHandler]]
        ] = None,
        verbose: bool = False,
    ) -> Optional[BaseCallbackManager]:
        """Configure the callback manager."""
        return _configure(cls, inheritable_callbacks, local_callbacks, verbose)


T = TypeVar("T", CallbackManager, AsyncCallbackManager)


def _configure(
    callback_manager_cls: Type[T],
    inheritable_callbacks: Optional[Union[T, List[BaseCallbackHandler]]] = None,
    local_callbacks: Optional[Union[T, List[BaseCallbackHandler]]] = None,
    verbose: bool = False,
) -> T:
    """Configure the callback manager."""
    callback_manager: Optional[T] = None
    if inheritable_callbacks or local_callbacks:
        if isinstance(inheritable_callbacks, list) or not inheritable_callbacks:
            callback_manager = callback_manager_cls(
                handlers=inheritable_callbacks if inheritable_callbacks else [],
                inheritable_handlers=inheritable_callbacks
                if inheritable_callbacks
                else [],
            )
        else:
            callback_manager = inheritable_callbacks
        callback_manager = copy.deepcopy(callback_manager)
        local_handlers_ = (
            local_callbacks
            if isinstance(local_callbacks, list)
            else (local_callbacks.handlers if local_callbacks else [])
        )
        [
            callback_manager.add_handler(copy.deepcopy(handler), False)
            for handler in local_handlers_
        ]

    if not callback_manager:
        callback_manager = callback_manager_cls([])
    tracing_enabled = os.environ.get("LANGCHAIN_TRACING") is not None
    if verbose or tracing_enabled:
        if verbose and not any(
            isinstance(handler, StdOutCallbackHandler)
            for handler in callback_manager.handlers
        ):
            callback_manager.add_handler(StdOutCallbackHandler(), False)

        if tracing_enabled and not any(
            isinstance(handler, LangChainTracer)
            for handler in callback_manager.handlers
        ):
            handler = LangChainTracer()
            handler.load_default_session()
            callback_manager.add_handler(handler, True)

    return callback_manager
