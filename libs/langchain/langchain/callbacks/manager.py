from __future__ import annotations

import asyncio
import functools
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Coroutine,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)
from uuid import UUID

from langsmith.run_helpers import get_run_tree_context
from tenacity import RetryCallState

from langchain.callbacks.base import (
    BaseCallbackHandler,
    BaseCallbackManager,
    Callbacks,
    ChainManagerMixin,
    LLMManagerMixin,
    RetrieverManagerMixin,
    RunManagerMixin,
    ToolManagerMixin,
)
from langchain.callbacks.openai_info import OpenAICallbackHandler
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.tracers import run_collector
from langchain.callbacks.tracers.langchain import (
    LangChainTracer,
)
from langchain.callbacks.tracers.langchain_v1 import LangChainTracerV1, TracerSessionV1
from langchain.callbacks.tracers.stdout import ConsoleCallbackHandler
from langchain.callbacks.tracers.wandb import WandbTracer
from langchain.schema import (
    AgentAction,
    AgentFinish,
    Document,
    LLMResult,
)
from langchain.schema.messages import BaseMessage, get_buffer_string
from langchain.schema.output import ChatGenerationChunk, GenerationChunk

if TYPE_CHECKING:
    from langsmith import Client as LangSmithClient

logger = logging.getLogger(__name__)

openai_callback_var: ContextVar[Optional[OpenAICallbackHandler]] = ContextVar(
    "openai_callback", default=None
)
tracing_callback_var: ContextVar[Optional[LangChainTracerV1]] = ContextVar(  # noqa: E501
    "tracing_callback", default=None
)
wandb_tracing_callback_var: ContextVar[Optional[WandbTracer]] = ContextVar(  # noqa: E501
    "tracing_wandb_callback", default=None
)

tracing_v2_callback_var: ContextVar[Optional[LangChainTracer]] = ContextVar(  # noqa: E501
    "tracing_callback_v2", default=None
)
run_collector_var: ContextVar[
    Optional[run_collector.RunCollectorCallbackHandler]
] = ContextVar(  # noqa: E501
    "run_collector", default=None
)


def _get_debug() -> bool:
    from langchain.globals import get_debug

    return get_debug()


@contextmanager
def get_openai_callback() -> Generator[OpenAICallbackHandler, None, None]:
    """Get the OpenAI callback handler in a context manager.
    which conveniently exposes token and cost information.

    Returns:
        OpenAICallbackHandler: The OpenAI callback handler.

    Example:
        >>> with get_openai_callback() as cb:
        ...     # Use the OpenAI callback handler
    """
    cb = OpenAICallbackHandler()
    openai_callback_var.set(cb)
    yield cb
    openai_callback_var.set(None)


@contextmanager
def tracing_enabled(
    session_name: str = "default",
) -> Generator[TracerSessionV1, None, None]:
    """Get the Deprecated LangChainTracer in a context manager.

    Args:
        session_name (str, optional): The name of the session.
          Defaults to "default".

    Returns:
        TracerSessionV1: The LangChainTracer session.

    Example:
        >>> with tracing_enabled() as session:
        ...     # Use the LangChainTracer session
    """
    cb = LangChainTracerV1()
    session = cast(TracerSessionV1, cb.load_session(session_name))
    try:
        tracing_callback_var.set(cb)
        yield session
    finally:
        tracing_callback_var.set(None)


@contextmanager
def wandb_tracing_enabled(
    session_name: str = "default",
) -> Generator[None, None, None]:
    """Get the WandbTracer in a context manager.

    Args:
        session_name (str, optional): The name of the session.
            Defaults to "default".

    Returns:
        None

    Example:
        >>> with wandb_tracing_enabled() as session:
        ...     # Use the WandbTracer session
    """
    cb = WandbTracer()
    wandb_tracing_callback_var.set(cb)
    yield None
    wandb_tracing_callback_var.set(None)


@contextmanager
def tracing_v2_enabled(
    project_name: Optional[str] = None,
    *,
    example_id: Optional[Union[str, UUID]] = None,
    tags: Optional[List[str]] = None,
    client: Optional[LangSmithClient] = None,
) -> Generator[LangChainTracer, None, None]:
    """Instruct LangChain to log all runs in context to LangSmith.

    Args:
        project_name (str, optional): The name of the project.
            Defaults to "default".
        example_id (str or UUID, optional): The ID of the example.
            Defaults to None.
        tags (List[str], optional): The tags to add to the run.
            Defaults to None.

    Returns:
        None

    Example:
        >>> with tracing_v2_enabled():
        ...     # LangChain code will automatically be traced

        You can use this to fetch the LangSmith run URL:

        >>> with tracing_v2_enabled() as cb:
        ...     chain.invoke("foo")
        ...     run_url = cb.get_run_url()
    """
    if isinstance(example_id, str):
        example_id = UUID(example_id)
    cb = LangChainTracer(
        example_id=example_id,
        project_name=project_name,
        tags=tags,
        client=client,
    )
    try:
        tracing_v2_callback_var.set(cb)
        yield cb
    finally:
        tracing_v2_callback_var.set(None)


@contextmanager
def collect_runs() -> Generator[run_collector.RunCollectorCallbackHandler, None, None]:
    """Collect all run traces in context.

    Returns:
        run_collector.RunCollectorCallbackHandler: The run collector callback handler.

    Example:
        >>> with collect_runs() as runs_cb:
                chain.invoke("foo")
                run_id = runs_cb.traced_runs[0].id
    """
    cb = run_collector.RunCollectorCallbackHandler()
    run_collector_var.set(cb)
    yield cb
    run_collector_var.set(None)


def _get_trace_callbacks(
    project_name: Optional[str] = None,
    example_id: Optional[Union[str, UUID]] = None,
    callback_manager: Optional[Union[CallbackManager, AsyncCallbackManager]] = None,
) -> Callbacks:
    if _tracing_v2_is_enabled():
        project_name_ = project_name or _get_tracer_project()
        tracer = tracing_v2_callback_var.get() or LangChainTracer(
            project_name=project_name_,
            example_id=example_id,
        )
        if callback_manager is None:
            cb = cast(Callbacks, [tracer])
        else:
            if not any(
                isinstance(handler, LangChainTracer)
                for handler in callback_manager.handlers
            ):
                callback_manager.add_handler(tracer, True)
                # If it already has a LangChainTracer, we don't need to add another one.
                # this would likely mess up the trace hierarchy.
            cb = callback_manager
    else:
        cb = None
    return cb


@contextmanager
def trace_as_chain_group(
    group_name: str,
    callback_manager: Optional[CallbackManager] = None,
    *,
    inputs: Optional[Dict[str, Any]] = None,
    project_name: Optional[str] = None,
    example_id: Optional[Union[str, UUID]] = None,
    run_id: Optional[UUID] = None,
    tags: Optional[List[str]] = None,
) -> Generator[CallbackManagerForChainGroup, None, None]:
    """Get a callback manager for a chain group in a context manager.
    Useful for grouping different calls together as a single run even if
    they aren't composed in a single chain.

    Args:
        group_name (str): The name of the chain group.
        callback_manager (CallbackManager, optional): The callback manager to use.
        inputs (Dict[str, Any], optional): The inputs to the chain group.
        project_name (str, optional): The name of the project.
            Defaults to None.
        example_id (str or UUID, optional): The ID of the example.
            Defaults to None.
        run_id (UUID, optional): The ID of the run.
        tags (List[str], optional): The inheritable tags to apply to all runs.
            Defaults to None.

    Note: must have LANGCHAIN_TRACING_V2 env var set to true to see the trace in LangSmith.

    Returns:
        CallbackManagerForChainGroup: The callback manager for the chain group.

    Example:
        .. code-block:: python

            llm_input = "Foo"
            with trace_as_chain_group("group_name", inputs={"input": llm_input}) as manager:
                # Use the callback manager for the chain group
                res = llm.predict(llm_input, callbacks=manager)
                manager.on_chain_end({"output": res})
    """  # noqa: E501
    cb = _get_trace_callbacks(
        project_name, example_id, callback_manager=callback_manager
    )
    cm = CallbackManager.configure(
        inheritable_callbacks=cb,
        inheritable_tags=tags,
    )

    run_manager = cm.on_chain_start({"name": group_name}, inputs or {}, run_id=run_id)
    child_cm = run_manager.get_child()
    group_cm = CallbackManagerForChainGroup(
        child_cm.handlers,
        child_cm.inheritable_handlers,
        child_cm.parent_run_id,
        parent_run_manager=run_manager,
        tags=child_cm.tags,
        inheritable_tags=child_cm.inheritable_tags,
        metadata=child_cm.metadata,
        inheritable_metadata=child_cm.inheritable_metadata,
    )
    try:
        yield group_cm
    except Exception as e:
        if not group_cm.ended:
            run_manager.on_chain_error(e)
        raise e
    else:
        if not group_cm.ended:
            run_manager.on_chain_end({})


@asynccontextmanager
async def atrace_as_chain_group(
    group_name: str,
    callback_manager: Optional[AsyncCallbackManager] = None,
    *,
    inputs: Optional[Dict[str, Any]] = None,
    project_name: Optional[str] = None,
    example_id: Optional[Union[str, UUID]] = None,
    run_id: Optional[UUID] = None,
    tags: Optional[List[str]] = None,
) -> AsyncGenerator[AsyncCallbackManagerForChainGroup, None]:
    """Get an async callback manager for a chain group in a context manager.
    Useful for grouping different async calls together as a single run even if
    they aren't composed in a single chain.

    Args:
        group_name (str): The name of the chain group.
        callback_manager (AsyncCallbackManager, optional): The async callback manager to use,
            which manages tracing and other callback behavior.
        project_name (str, optional): The name of the project.
            Defaults to None.
        example_id (str or UUID, optional): The ID of the example.
            Defaults to None.
        run_id (UUID, optional): The ID of the run.
        tags (List[str], optional): The inheritable tags to apply to all runs.
            Defaults to None.
    Returns:
        AsyncCallbackManager: The async callback manager for the chain group.

    Note: must have LANGCHAIN_TRACING_V2 env var set to true to see the trace in LangSmith.

    Example:
        .. code-block:: python

            llm_input = "Foo"
            async with atrace_as_chain_group("group_name", inputs={"input": llm_input}) as manager:
                # Use the async callback manager for the chain group
                res = await llm.apredict(llm_input, callbacks=manager)
                await manager.on_chain_end({"output": res})
    """  # noqa: E501
    cb = _get_trace_callbacks(
        project_name, example_id, callback_manager=callback_manager
    )
    cm = AsyncCallbackManager.configure(inheritable_callbacks=cb, inheritable_tags=tags)

    run_manager = await cm.on_chain_start(
        {"name": group_name}, inputs or {}, run_id=run_id
    )
    child_cm = run_manager.get_child()
    group_cm = AsyncCallbackManagerForChainGroup(
        child_cm.handlers,
        child_cm.inheritable_handlers,
        child_cm.parent_run_id,
        parent_run_manager=run_manager,
        tags=child_cm.tags,
        inheritable_tags=child_cm.inheritable_tags,
        metadata=child_cm.metadata,
        inheritable_metadata=child_cm.inheritable_metadata,
    )
    try:
        yield group_cm
    except Exception as e:
        if not group_cm.ended:
            await run_manager.on_chain_error(e)
        raise e
    else:
        if not group_cm.ended:
            await run_manager.on_chain_end({})


def handle_event(
    handlers: List[BaseCallbackHandler],
    event_name: str,
    ignore_condition_name: Optional[str],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Generic event handler for CallbackManager.

    Note: This function is used by langserve to handle events.

    Args:
        handlers: The list of handlers that will handle the event
        event_name: The name of the event (e.g., "on_llm_start")
        ignore_condition_name: Name of the attribute defined on handler
            that if True will cause the handler to be skipped for the given event
        *args: The arguments to pass to the event handler
        **kwargs: The keyword arguments to pass to the event handler
    """
    coros: List[Coroutine[Any, Any, Any]] = []

    try:
        message_strings: Optional[List[str]] = None
        for handler in handlers:
            try:
                if ignore_condition_name is None or not getattr(
                    handler, ignore_condition_name
                ):
                    event = getattr(handler, event_name)(*args, **kwargs)
                    if asyncio.iscoroutine(event):
                        coros.append(event)
            except NotImplementedError as e:
                if event_name == "on_chat_model_start":
                    if message_strings is None:
                        message_strings = [get_buffer_string(m) for m in args[1]]
                    handle_event(
                        [handler],
                        "on_llm_start",
                        "ignore_llm",
                        args[0],
                        message_strings,
                        *args[2:],
                        **kwargs,
                    )
                else:
                    handler_name = handler.__class__.__name__
                    logger.warning(
                        f"NotImplementedError in {handler_name}.{event_name}"
                        f" callback: {repr(e)}"
                    )
            except Exception as e:
                logger.warning(
                    f"Error in {handler.__class__.__name__}.{event_name} callback:"
                    f" {repr(e)}"
                )
                if handler.raise_error:
                    raise e
    finally:
        if coros:
            try:
                # Raises RuntimeError if there is no current event loop.
                asyncio.get_running_loop()
                loop_running = True
            except RuntimeError:
                loop_running = False

            if loop_running:
                # If we try to submit this coroutine to the running loop
                # we end up in a deadlock, as we'd have gotten here from a
                # running coroutine, which we cannot interrupt to run this one.
                # The solution is to create a new loop in a new thread.
                with ThreadPoolExecutor(1) as executor:
                    executor.submit(_run_coros, coros).result()
            else:
                _run_coros(coros)


def _run_coros(coros: List[Coroutine[Any, Any, Any]]) -> None:
    if hasattr(asyncio, "Runner"):
        # Python 3.11+
        # Run the coroutines in a new event loop, taking care to
        # - install signal handlers
        # - run pending tasks scheduled by `coros`
        # - close asyncgens and executors
        # - close the loop
        with asyncio.Runner() as runner:
            # Run the coroutine, get the result
            for coro in coros:
                runner.run(coro)

            # Run pending tasks scheduled by coros until they are all done
            while pending := asyncio.all_tasks(runner.get_loop()):
                runner.run(asyncio.wait(pending))
    else:
        # Before Python 3.11 we need to run each coroutine in a new event loop
        # as the Runner api is not available.
        for coro in coros:
            asyncio.run(coro)


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
                f"NotImplementedError in {handler.__class__.__name__}.{event_name}"
                f" callback: {repr(e)}"
            )
    except Exception as e:
        logger.warning(
            f"Error in {handler.__class__.__name__}.{event_name} callback:"
            f" {repr(e)}"
        )
        if handler.raise_error:
            raise e


async def ahandle_event(
    handlers: List[BaseCallbackHandler],
    event_name: str,
    ignore_condition_name: Optional[str],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Generic event handler for AsyncCallbackManager.

    Note: This function is used by langserve to handle events.

    Args:
        handlers: The list of handlers that will handle the event
        event_name: The name of the event (e.g., "on_llm_start")
        ignore_condition_name: Name of the attribute defined on handler
            that if True will cause the handler to be skipped for the given event
        *args: The arguments to pass to the event handler
        **kwargs: The keyword arguments to pass to the event handler
    """
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
            run_id=uuid.uuid4(),
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
        handle_event(
            self.handlers,
            "on_text",
            None,
            text,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    def on_retry(
        self,
        retry_state: RetryCallState,
        **kwargs: Any,
    ) -> None:
        handle_event(
            self.handlers,
            "on_retry",
            "ignore_retry",
            retry_state,
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
        await ahandle_event(
            self.handlers,
            "on_text",
            None,
            text,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            **kwargs,
        )

    async def on_retry(
        self,
        retry_state: RetryCallState,
        **kwargs: Any,
    ) -> None:
        await ahandle_event(
            self.handlers,
            "on_retry",
            "ignore_retry",
            retry_state,
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
        *,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM generates a new token.

        Args:
            token (str): The new token.
        """
        handle_event(
            self.handlers,
            "on_llm_new_token",
            "ignore_llm",
            token=token,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            chunk=chunk,
            **kwargs,
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running.

        Args:
            response (LLMResult): The LLM result.
        """
        handle_event(
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
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors.

        Args:
            error (Exception or KeyboardInterrupt): The error.
        """
        handle_event(
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
        *,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM generates a new token.

        Args:
            token (str): The new token.
        """
        await ahandle_event(
            self.handlers,
            "on_llm_new_token",
            "ignore_llm",
            token,
            chunk=chunk,
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
        await ahandle_event(
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
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors.

        Args:
            error (Exception or KeyboardInterrupt): The error.
        """
        await ahandle_event(
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

    def on_chain_end(self, outputs: Union[Dict[str, Any], Any], **kwargs: Any) -> None:
        """Run when chain ends running.

        Args:
            outputs (Union[Dict[str, Any], Any]): The outputs of the chain.
        """
        handle_event(
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
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors.

        Args:
            error (Exception or KeyboardInterrupt): The error.
        """
        handle_event(
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
        handle_event(
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
        handle_event(
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

    async def on_chain_end(
        self, outputs: Union[Dict[str, Any], Any], **kwargs: Any
    ) -> None:
        """Run when chain ends running.

        Args:
            outputs (Union[Dict[str, Any], Any]): The outputs of the chain.
        """
        await ahandle_event(
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
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors.

        Args:
            error (Exception or KeyboardInterrupt): The error.
        """
        await ahandle_event(
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
        await ahandle_event(
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
        await ahandle_event(
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
        handle_event(
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
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors.

        Args:
            error (Exception or KeyboardInterrupt): The error.
        """
        handle_event(
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
        await ahandle_event(
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
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors.

        Args:
            error (Exception or KeyboardInterrupt): The error.
        """
        await ahandle_event(
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
        handle_event(
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
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when retriever errors."""
        handle_event(
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
        await ahandle_event(
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
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when retriever errors."""
        await ahandle_event(
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
    """Callback manager that handles callbacks from LangChain."""

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
            run_id_ = uuid.uuid4()
            handle_event(
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
            run_id_ = uuid.uuid4()
            handle_event(
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
        inputs: Union[Dict[str, Any], Any],
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> CallbackManagerForChainRun:
        """Run when chain starts running.

        Args:
            serialized (Dict[str, Any]): The serialized chain.
            inputs (Union[Dict[str, Any], Any]): The inputs to the chain.
            run_id (UUID, optional): The ID of the run. Defaults to None.

        Returns:
            CallbackManagerForChainRun: The callback manager for the chain run.
        """
        if run_id is None:
            run_id = uuid.uuid4()
        handle_event(
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
            run_id = uuid.uuid4()

        handle_event(
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
            run_id = uuid.uuid4()

        handle_event(
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


class CallbackManagerForChainGroup(CallbackManager):
    """Callback manager for the chain group."""

    def __init__(
        self,
        handlers: List[BaseCallbackHandler],
        inheritable_handlers: Optional[List[BaseCallbackHandler]] = None,
        parent_run_id: Optional[UUID] = None,
        *,
        parent_run_manager: CallbackManagerForChainRun,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            handlers,
            inheritable_handlers,
            parent_run_id,
            **kwargs,
        )
        self.parent_run_manager = parent_run_manager
        self.ended = False

    def copy(self) -> CallbackManagerForChainGroup:
        return self.__class__(
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
            parent_run_manager=self.parent_run_manager,
        )

    def on_chain_end(self, outputs: Union[Dict[str, Any], Any], **kwargs: Any) -> None:
        """Run when traced chain group ends.

        Args:
            outputs (Union[Dict[str, Any], Any]): The outputs of the chain.
        """
        self.ended = True
        return self.parent_run_manager.on_chain_end(outputs, **kwargs)

    def on_chain_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors.

        Args:
            error (Exception or KeyboardInterrupt): The error.
        """
        self.ended = True
        return self.parent_run_manager.on_chain_error(error, **kwargs)


class AsyncCallbackManager(BaseCallbackManager):
    """Async callback manager that handles callbacks from LangChain."""

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
            run_id_ = uuid.uuid4()

            tasks.append(
                ahandle_event(
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
    ) -> List[AsyncCallbackManagerForLLMRun]:
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
            run_id_ = uuid.uuid4()

            tasks.append(
                ahandle_event(
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
        inputs: Union[Dict[str, Any], Any],
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> AsyncCallbackManagerForChainRun:
        """Run when chain starts running.

        Args:
            serialized (Dict[str, Any]): The serialized chain.
            inputs (Union[Dict[str, Any], Any]): The inputs to the chain.
            run_id (UUID, optional): The ID of the run. Defaults to None.

        Returns:
            AsyncCallbackManagerForChainRun: The async callback manager
                for the chain run.
        """
        if run_id is None:
            run_id = uuid.uuid4()

        await ahandle_event(
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
            run_id = uuid.uuid4()

        await ahandle_event(
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
            run_id = uuid.uuid4()

        await ahandle_event(
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


class AsyncCallbackManagerForChainGroup(AsyncCallbackManager):
    """Async callback manager for the chain group."""

    def __init__(
        self,
        handlers: List[BaseCallbackHandler],
        inheritable_handlers: Optional[List[BaseCallbackHandler]] = None,
        parent_run_id: Optional[UUID] = None,
        *,
        parent_run_manager: AsyncCallbackManagerForChainRun,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            handlers,
            inheritable_handlers,
            parent_run_id,
            **kwargs,
        )
        self.parent_run_manager = parent_run_manager
        self.ended = False

    def copy(self) -> AsyncCallbackManagerForChainGroup:
        return self.__class__(
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
            parent_run_manager=self.parent_run_manager,
        )

    async def on_chain_end(
        self, outputs: Union[Dict[str, Any], Any], **kwargs: Any
    ) -> None:
        """Run when traced chain group ends.

        Args:
            outputs (Union[Dict[str, Any], Any]): The outputs of the chain.
        """
        self.ended = True
        await self.parent_run_manager.on_chain_end(outputs, **kwargs)

    async def on_chain_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors.

        Args:
            error (Exception or KeyboardInterrupt): The error.
        """
        self.ended = True
        await self.parent_run_manager.on_chain_error(error, **kwargs)


T = TypeVar("T", CallbackManager, AsyncCallbackManager)


def env_var_is_set(env_var: str) -> bool:
    """Check if an environment variable is set.

    Args:
        env_var (str): The name of the environment variable.

    Returns:
        bool: True if the environment variable is set, False otherwise.
    """
    return env_var in os.environ and os.environ[env_var] not in (
        "",
        "0",
        "false",
        "False",
    )


def _tracing_v2_is_enabled() -> bool:
    return (
        env_var_is_set("LANGCHAIN_TRACING_V2")
        or tracing_v2_callback_var.get() is not None
        or get_run_tree_context() is not None
    )


def _get_tracer_project() -> str:
    run_tree = get_run_tree_context()
    return getattr(
        run_tree,
        "session_name",
        getattr(
            # Note, if people are trying to nest @traceable functions and the
            # tracing_v2_enabled context manager, this will likely mess up the
            # tree structure.
            tracing_v2_callback_var.get(),
            "project",
            os.environ.get(
                "LANGCHAIN_PROJECT", os.environ.get("LANGCHAIN_SESSION", "default")
            ),
        ),
    )


def _configure(
    callback_manager_cls: Type[T],
    inheritable_callbacks: Callbacks = None,
    local_callbacks: Callbacks = None,
    verbose: bool = False,
    inheritable_tags: Optional[List[str]] = None,
    local_tags: Optional[List[str]] = None,
    inheritable_metadata: Optional[Dict[str, Any]] = None,
    local_metadata: Optional[Dict[str, Any]] = None,
) -> T:
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
    run_tree = get_run_tree_context()
    parent_run_id = None if run_tree is None else getattr(run_tree, "id")
    callback_manager = callback_manager_cls(handlers=[], parent_run_id=parent_run_id)
    if inheritable_callbacks or local_callbacks:
        if isinstance(inheritable_callbacks, list) or inheritable_callbacks is None:
            inheritable_callbacks_ = inheritable_callbacks or []
            callback_manager = callback_manager_cls(
                handlers=inheritable_callbacks_.copy(),
                inheritable_handlers=inheritable_callbacks_.copy(),
                parent_run_id=parent_run_id,
            )
        else:
            callback_manager = callback_manager_cls(
                handlers=inheritable_callbacks.handlers.copy(),
                inheritable_handlers=inheritable_callbacks.inheritable_handlers.copy(),
                parent_run_id=inheritable_callbacks.parent_run_id,
                tags=inheritable_callbacks.tags.copy(),
                inheritable_tags=inheritable_callbacks.inheritable_tags.copy(),
                metadata=inheritable_callbacks.metadata.copy(),
                inheritable_metadata=inheritable_callbacks.inheritable_metadata.copy(),
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
    tracing_v2_enabled_ = _tracing_v2_is_enabled()
    tracer_project = _get_tracer_project()
    run_collector_ = run_collector_var.get()
    debug = _get_debug()
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
                handler.load_session(tracer_project)
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
                    handler = LangChainTracer(project_name=tracer_project)
                    callback_manager.add_handler(handler, True)
                except Exception as e:
                    logger.warning(
                        "Unable to load requested LangChainTracer."
                        " To disable this warning,"
                        " unset the  LANGCHAIN_TRACING_V2 environment variables.",
                        e,
                    )
        if open_ai is not None and not any(
            handler is open_ai  # direct pointer comparison
            for handler in callback_manager.handlers
        ):
            callback_manager.add_handler(open_ai, True)
    if run_collector_ is not None and not any(
        handler is run_collector_  # direct pointer comparison
        for handler in callback_manager.handlers
    ):
        callback_manager.add_handler(run_collector_, False)
    return callback_manager
