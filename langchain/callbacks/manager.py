from __future__ import annotations

import asyncio
import functools
import logging
import os
import warnings
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)
from uuid import UUID, uuid4

import langchain
from langchain.callbacks.base import (
    BaseCallbackHandler,
    BaseCallbackManager,
    ChainManagerMixin,
    LLMManagerMixin,
    RunManagerMixin,
    ToolManagerMixin,
)
from langchain.callbacks.openai_info import OpenAICallbackHandler
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.callbacks.tracers.langchain_v1 import LangChainTracerV1, TracerSessionV1
from langchain.callbacks.tracers.stdout import ConsoleCallbackHandler
from langchain.callbacks.tracers.wandb import WandbTracer
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseMessage,
    LLMResult,
    get_buffer_string,
)

logger = logging.getLogger(__name__)
Callbacks = Optional[Union[List[BaseCallbackHandler], BaseCallbackManager]]

openai_callback_var: ContextVar[Optional[OpenAICallbackHandler]] = ContextVar(
    "openai_callback", default=None
)
tracing_callback_var: ContextVar[
    Optional[LangChainTracerV1]
] = ContextVar(  # noqa: E501
    "tracing_callback", default=None
)
wandb_tracing_callback_var: ContextVar[
    Optional[WandbTracer]
] = ContextVar(  # noqa: E501
    "tracing_wandb_callback", default=None
)

tracing_v2_callback_var: ContextVar[
    Optional[LangChainTracer]
] = ContextVar(  # noqa: E501
    "tracing_callback_v2", default=None
)


def _get_debug() -> bool:
    return langchain.debug


@contextmanager
def get_openai_callback() -> Generator[OpenAICallbackHandler, None, None]:
    """Get OpenAI callback handler in a context manager."""
    cb = OpenAICallbackHandler()
    openai_callback_var.set(cb)
    yield cb
    openai_callback_var.set(None)


@contextmanager
def tracing_enabled(
    session_name: str = "default",
) -> Generator[TracerSessionV1, None, None]:
    """Get Tracer in a context manager."""
    cb = LangChainTracerV1()
    session = cast(TracerSessionV1, cb.load_session(session_name))
    tracing_callback_var.set(cb)
    yield session
    tracing_callback_var.set(None)


@contextmanager
def wandb_tracing_enabled(
    session_name: str = "default",
) -> Generator[None, None, None]:
    """Get WandbTracer in a context manager."""
    cb = WandbTracer()
    wandb_tracing_callback_var.set(cb)
    yield None
    wandb_tracing_callback_var.set(None)


@contextmanager
def tracing_v2_enabled(
    session_name: Optional[str] = None,
    *,
    example_id: Optional[Union[str, UUID]] = None,
) -> Generator[None, None, None]:
    """Get the experimental tracer handler in a context manager."""
    # Issue a warning that this is experimental
    warnings.warn(
        "The tracing v2 API is in development. "
        "This is not yet stable and may change in the future."
    )
    if isinstance(example_id, str):
        example_id = UUID(example_id)
    cb = LangChainTracer(
        example_id=example_id,
        session_name=session_name,
    )
    tracing_v2_callback_var.set(cb)
    yield
    tracing_v2_callback_var.set(None)


@contextmanager
def trace_as_chain_group(
    group_name: str,
    *,
    session_name: Optional[str] = None,
    example_id: Optional[Union[str, UUID]] = None,
    tenant_id: Optional[str] = None,
    session_extra: Optional[Dict[str, Any]] = None,
) -> Generator[CallbackManager, None, None]:
    """Get a callback manager for a chain group in a context manager."""
    cb = LangChainTracer(
        tenant_id=tenant_id,
        session_name=session_name,
        example_id=example_id,
        session_extra=session_extra,
    )
    cm = CallbackManager.configure(
        inheritable_callbacks=[cb],
    )

    run_manager = cm.on_chain_start({"name": group_name}, {})
    yield run_manager.get_child()
    run_manager.on_chain_end({})


@asynccontextmanager
async def atrace_as_chain_group(
    group_name: str,
    *,
    session_name: Optional[str] = None,
    example_id: Optional[Union[str, UUID]] = None,
    tenant_id: Optional[str] = None,
    session_extra: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[AsyncCallbackManager, None]:
    """Get a callback manager for a chain group in a context manager."""
    cb = LangChainTracer(
        tenant_id=tenant_id,
        session_name=session_name,
        example_id=example_id,
        session_extra=session_extra,
    )
    cm = AsyncCallbackManager.configure(
        inheritable_callbacks=[cb],
    )

    run_manager = await cm.on_chain_start({"name": group_name}, {})
    try:
        yield run_manager.get_child()
    finally:
        await run_manager.on_chain_end({})


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
                logger.warning(f"Error in {event_name} callback: {e}")
        except Exception as e:
            if handler.raise_error:
                raise e
            logging.warning(f"Error in {event_name} callback: {e}")


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
            logger.warning(f"Error in {event_name} callback: {e}")
    except Exception as e:
        logger.warning(f"Error in {event_name} callback: {e}")


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


BRM = TypeVar("BRM", bound="BaseRunManager")


class BaseRunManager(RunManagerMixin):
    """Base class for run manager (a bound callback manager)."""

    def __init__(
        self,
        run_id: UUID,
        handlers: List[BaseCallbackHandler],
        inheritable_handlers: List[BaseCallbackHandler],
        parent_run_id: Optional[UUID] = None,
    ) -> None:
        """Initialize run manager."""
        self.run_id = run_id
        self.handlers = handlers
        self.inheritable_handlers = inheritable_handlers
        self.parent_run_id = parent_run_id

    @classmethod
    def get_noop_manager(cls: Type[BRM]) -> BRM:
        """Return a manager that doesn't perform any operations."""
        return cls(uuid4(), [], [])


class RunManager(BaseRunManager):
    """Sync Run Manager."""

    def on_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> Any:
        """Run when text is received."""
        _handle_event(
            self.handlers,
            "on_text",
            None,
            text,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )


class AsyncRunManager(BaseRunManager):
    """Async Run Manager."""

    async def on_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> Any:
        """Run when text is received."""
        await _ahandle_event(
            self.handlers,
            "on_text",
            None,
            text,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )


class CallbackManagerForLLMRun(RunManager, LLMManagerMixin):
    """Callback manager for LLM run."""

    def on_llm_new_token(
        self,
        token: str,
        **kwargs: Any,
    ) -> None:
        """Run when LLM generates a new token."""
        _handle_event(
            self.handlers,
            "on_llm_new_token",
            "ignore_llm",
            token=token,
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

    async def on_llm_new_token(
        self,
        token: str,
        **kwargs: Any,
    ) -> None:
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

    def on_tool_end(
        self,
        output: str,
        **kwargs: Any,
    ) -> None:
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
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> CallbackManagerForLLMRun:
        """Run when LLM starts running."""
        if run_id is None:
            run_id = uuid4()

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

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> CallbackManagerForLLMRun:
        """Run when LLM starts running."""
        if run_id is None:
            run_id = uuid4()
        _handle_event(
            self.handlers,
            "on_chat_model_start",
            "ignore_chat_model",
            serialized,
            messages,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

        # Re-use the LLM Run Manager since the outputs are treated
        # the same for now
        return CallbackManagerForLLMRun(
            run_id, self.handlers, self.inheritable_handlers, self.parent_run_id
        )

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> CallbackManagerForChainRun:
        """Run when chain starts running."""
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
            **kwargs,
        )

        return CallbackManagerForChainRun(
            run_id, self.handlers, self.inheritable_handlers, self.parent_run_id
        )

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> CallbackManagerForToolRun:
        """Run when tool starts running."""
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
            **kwargs,
        )

        return CallbackManagerForToolRun(
            run_id, self.handlers, self.inheritable_handlers, self.parent_run_id
        )

    @classmethod
    def configure(
        cls,
        inheritable_callbacks: Callbacks = None,
        local_callbacks: Callbacks = None,
        verbose: bool = False,
    ) -> CallbackManager:
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
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> AsyncCallbackManagerForLLMRun:
        """Run when LLM starts running."""
        if run_id is None:
            run_id = uuid4()

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

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if run_id is None:
            run_id = uuid4()

        await _ahandle_event(
            self.handlers,
            "on_chat_model_start",
            "ignore_chat_model",
            serialized,
            messages,
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
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> AsyncCallbackManagerForChainRun:
        """Run when chain starts running."""
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
            **kwargs,
        )

        return AsyncCallbackManagerForChainRun(
            run_id, self.handlers, self.inheritable_handlers, self.parent_run_id
        )

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> AsyncCallbackManagerForToolRun:
        """Run when tool starts running."""
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
            **kwargs,
        )

        return AsyncCallbackManagerForToolRun(
            run_id, self.handlers, self.inheritable_handlers, self.parent_run_id
        )

    @classmethod
    def configure(
        cls,
        inheritable_callbacks: Callbacks = None,
        local_callbacks: Callbacks = None,
        verbose: bool = False,
    ) -> AsyncCallbackManager:
        """Configure the callback manager."""
        return _configure(cls, inheritable_callbacks, local_callbacks, verbose)


T = TypeVar("T", CallbackManager, AsyncCallbackManager)


def env_var_is_set(env_var: str) -> bool:
    """Check if an environment variable is set."""
    return env_var in os.environ and os.environ[env_var] not in (
        "",
        "0",
        "false",
        "False",
    )


def _configure(
    callback_manager_cls: Type[T],
    inheritable_callbacks: Callbacks = None,
    local_callbacks: Callbacks = None,
    verbose: bool = False,
) -> T:
    """Configure the callback manager."""
    callback_manager = callback_manager_cls([])
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
            )
        local_handlers_ = (
            local_callbacks
            if isinstance(local_callbacks, list)
            else (local_callbacks.handlers if local_callbacks else [])
        )
        for handler in local_handlers_:
            callback_manager.add_handler(handler, False)

    tracer = tracing_callback_var.get()
    wandb_tracer = wandb_tracing_callback_var.get()
    open_ai = openai_callback_var.get()
    tracing_enabled_ = (
        env_var_is_set("LANGCHAIN_TRACING")
        or tracer is not None
        or env_var_is_set("LANGCHAIN_HANDLER")
    )
    wandb_tracing_enabled_ = (
        env_var_is_set("LANGCHAIN_WANDB_TRACING") or wandb_tracer is not None
    )

    tracer_v2 = tracing_v2_callback_var.get()
    tracing_v2_enabled_ = (
        env_var_is_set("LANGCHAIN_TRACING_V2") or tracer_v2 is not None
    )
    tracer_session = os.environ.get("LANGCHAIN_SESSION")
    debug = _get_debug()
    if tracer_session is None:
        tracer_session = "default"
    if (
        verbose
        or debug
        or tracing_enabled_
        or tracing_v2_enabled_
        or wandb_tracing_enabled_
        or open_ai is not None
    ):
        if verbose and not any(
            isinstance(handler, StdOutCallbackHandler)
            for handler in callback_manager.handlers
        ):
            if debug:
                pass
            else:
                callback_manager.add_handler(StdOutCallbackHandler(), False)
        if debug and not any(
            isinstance(handler, ConsoleCallbackHandler)
            for handler in callback_manager.handlers
        ):
            callback_manager.add_handler(ConsoleCallbackHandler(), True)
        if tracing_enabled_ and not any(
            isinstance(handler, LangChainTracerV1)
            for handler in callback_manager.handlers
        ):
            if tracer:
                callback_manager.add_handler(tracer, True)
            else:
                handler = LangChainTracerV1()
                handler.load_session(tracer_session)
                callback_manager.add_handler(handler, True)
        if wandb_tracing_enabled_ and not any(
            isinstance(handler, WandbTracer) for handler in callback_manager.handlers
        ):
            if wandb_tracer:
                callback_manager.add_handler(wandb_tracer, True)
            else:
                handler = WandbTracer()
                callback_manager.add_handler(handler, True)
        if tracing_v2_enabled_ and not any(
            isinstance(handler, LangChainTracer)
            for handler in callback_manager.handlers
        ):
            if tracer_v2:
                callback_manager.add_handler(tracer_v2, True)
            else:
                try:
                    handler = LangChainTracer(session_name=tracer_session)
                    callback_manager.add_handler(handler, True)
                except Exception as e:
                    logger.warning(
                        "Unable to load requested LangChainTracer."
                        " To disable this warning,"
                        " unset the  LANGCHAIN_TRACING_V2 environment variables.",
                        e,
                    )
        if open_ai is not None and not any(
            isinstance(handler, OpenAICallbackHandler)
            for handler in callback_manager.handlers
        ):
            callback_manager.add_handler(open_ai, True)
    return callback_manager
