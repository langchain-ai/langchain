from __future__ import annotations

import asyncio
import functools
import logging
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from contextvars import copy_context
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
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

from langchain_core.callbacks.base import (
    BaseCallbackHandler,
    BaseCallbackManager,
    Callbacks,
    ChainManagerMixin,
    LLMManagerMixin,
    RetrieverManagerMixin,
    RunManagerMixin,
    ToolManagerMixin,
)
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.tracers.schemas import Run
from langchain_core.utils.env import env_var_is_set

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.documents import Document
    from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult
    from langchain_core.runnables.config import RunnableConfig

logger = logging.getLogger(__name__)


def _get_debug() -> bool:
    from langchain_core.globals import get_debug

    return get_debug()


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
    metadata: Optional[Dict[str, Any]] = None,
) -> Generator[CallbackManagerForChainGroup, None, None]:
    """Get a callback manager for a chain group in a context manager.
    Useful for grouping different calls together as a single run even if
    they aren't composed in a single chain.

    Args:
        group_name (str): The name of the chain group.
        callback_manager (CallbackManager, optional): The callback manager to use.
            Defaults to None.
        inputs (Dict[str, Any], optional): The inputs to the chain group.
            Defaults to None.
        project_name (str, optional): The name of the project.
            Defaults to None.
        example_id (str or UUID, optional): The ID of the example.
            Defaults to None.
        run_id (UUID, optional): The ID of the run.
        tags (List[str], optional): The inheritable tags to apply to all runs.
            Defaults to None.
        metadata (Dict[str, Any], optional): The metadata to apply to all runs.
            Defaults to None.

    Note: must have LANGCHAIN_TRACING_V2 env var set to true to see the trace in LangSmith.

    Returns:
        CallbackManagerForChainGroup: The callback manager for the chain group.

    Example:
        .. code-block:: python

            llm_input = "Foo"
            with trace_as_chain_group("group_name", inputs={"input": llm_input}) as manager:
                # Use the callback manager for the chain group
                res = llm.invoke(llm_input, {"callbacks": manager})
                manager.on_chain_end({"output": res})
    """  # noqa: E501
    from langchain_core.tracers.context import _get_trace_callbacks

    cb = _get_trace_callbacks(
        project_name, example_id, callback_manager=callback_manager
    )
    cm = CallbackManager.configure(
        inheritable_callbacks=cb,
        inheritable_tags=tags,
        inheritable_metadata=metadata,
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
    metadata: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[AsyncCallbackManagerForChainGroup, None]:
    """Get an async callback manager for a chain group in a context manager.
    Useful for grouping different async calls together as a single run even if
    they aren't composed in a single chain.

    Args:
        group_name (str): The name of the chain group.
        callback_manager (AsyncCallbackManager, optional): The async callback manager to use,
            which manages tracing and other callback behavior. Defaults to None.
        inputs (Dict[str, Any], optional): The inputs to the chain group.
            Defaults to None.
        project_name (str, optional): The name of the project.
            Defaults to None.
        example_id (str or UUID, optional): The ID of the example.
            Defaults to None.
        run_id (UUID, optional): The ID of the run.
        tags (List[str], optional): The inheritable tags to apply to all runs.
            Defaults to None.
        metadata (Dict[str, Any], optional): The metadata to apply to all runs.
            Defaults to None.
    Returns:
        AsyncCallbackManager: The async callback manager for the chain group.

    Note: must have LANGCHAIN_TRACING_V2 env var set to true to see the trace in LangSmith.

    Example:
        .. code-block:: python

            llm_input = "Foo"
            async with atrace_as_chain_group("group_name", inputs={"input": llm_input}) as manager:
                # Use the async callback manager for the chain group
                res = await llm.ainvoke(llm_input, {"callbacks": manager})
                await manager.on_chain_end({"output": res})
    """  # noqa: E501
    from langchain_core.tracers.context import _get_trace_callbacks

    cb = _get_trace_callbacks(
        project_name, example_id, callback_manager=callback_manager
    )
    cm = AsyncCallbackManager.configure(
        inheritable_callbacks=cb, inheritable_tags=tags, inheritable_metadata=metadata
    )

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


Func = TypeVar("Func", bound=Callable)


def shielded(func: Func) -> Func:
    """
    Makes so an awaitable method is always shielded from cancellation.

    Args:
        func (Callable): The function to shield.

    Returns:
        Callable: The shielded function
    """

    @functools.wraps(func)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        return await asyncio.shield(func(*args, **kwargs))

    return cast(Func, wrapped)


def handle_event(
    handlers: List[BaseCallbackHandler],
    event_name: str,
    ignore_condition_name: Optional[str],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Generic event handler for CallbackManager.

    Note: This function is used by LangServe to handle events.

    Args:
        handlers: The list of handlers that will handle the event.
        event_name: The name of the event (e.g., "on_llm_start").
        ignore_condition_name: Name of the attribute defined on handler
            that if True will cause the handler to be skipped for the given event.
        *args: The arguments to pass to the event handler.
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
                    executor.submit(
                        cast(Callable, copy_context().run), _run_coros, coros
                    ).result()
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
                try:
                    runner.run(coro)
                except Exception as e:
                    logger.warning(f"Error in callback coroutine: {repr(e)}")

            # Run pending tasks scheduled by coros until they are all done
            while pending := asyncio.all_tasks(runner.get_loop()):
                runner.run(asyncio.wait(pending))
    else:
        # Before Python 3.11 we need to run each coroutine in a new event loop
        # as the Runner api is not available.
        for coro in coros:
            try:
                asyncio.run(coro)
            except Exception as e:
                logger.warning(f"Error in callback coroutine: {repr(e)}")


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
                        None,
                        cast(
                            Callable,
                            functools.partial(
                                copy_context().run, event, *args, **kwargs
                            ),
                        ),
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
    """Async generic event handler for AsyncCallbackManager.

    Note: This function is used by LangServe to handle events.

    Args:
        handlers: The list of handlers that will handle the event.
        event_name: The name of the event (e.g., "on_llm_start").
        ignore_condition_name: Name of the attribute defined on handler
            that if True will cause the handler to be skipped for the given event.
        *args: The arguments to pass to the event handler.
        **kwargs: The keyword arguments to pass to the event handler.
    """
    for handler in [h for h in handlers if h.run_inline]:
        await _ahandle_event_for_handler(
            handler, event_name, ignore_condition_name, *args, **kwargs
        )
    await asyncio.gather(
        *(
            _ahandle_event_for_handler(
                handler,
                event_name,
                ignore_condition_name,
                *args,
                **kwargs,
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
            tags (Optional[List[str]]): The list of tags. Defaults to None.
            inheritable_tags (Optional[List[str]]): The list of inheritable tags.
                Defaults to None.
            metadata (Optional[Dict[str, Any]]): The metadata.
                Defaults to None.
            inheritable_metadata (Optional[Dict[str, Any]]): The inheritable metadata.
                Defaults to None.
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
        """Run when a text is received.

        Args:
            text (str): The received text.
            **kwargs (Any): Additional keyword arguments.

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
        """Run when a retry is received.

        Args:
            retry_state (RetryCallState): The retry state.
            **kwargs (Any): Additional keyword arguments.
        """
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


class AsyncRunManager(BaseRunManager, ABC):
    """Async Run Manager."""

    @abstractmethod
    def get_sync(self) -> RunManager:
        """Get the equivalent sync RunManager.

        Returns:
            RunManager: The sync RunManager.
        """

    async def on_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> Any:
        """Run when a text is received.

        Args:
            text (str): The received text.
            **kwargs (Any): Additional keyword arguments.

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
        """Async run when a retry is received.

        Args:
            retry_state (RetryCallState): The retry state.
            **kwargs (Any): Additional keyword arguments.
        """
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
            chunk (Optional[Union[GenerationChunk, ChatGenerationChunk]], optional):
                The chunk. Defaults to None.
            **kwargs (Any): Additional keyword arguments.
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
            **kwargs (Any): Additional keyword arguments.
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
            kwargs (Any): Additional keyword arguments.
                - response (LLMResult): The response which was generated before
                    the error occurred.
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

    def get_sync(self) -> CallbackManagerForLLMRun:
        """Get the equivalent sync RunManager.

        Returns:
            CallbackManagerForLLMRun: The sync RunManager.
        """
        return CallbackManagerForLLMRun(
            run_id=self.run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    @shielded
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
            chunk (Optional[Union[GenerationChunk, ChatGenerationChunk]], optional):
                The chunk. Defaults to None.
            **kwargs (Any): Additional keyword arguments.
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

    @shielded
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running.

        Args:
            response (LLMResult): The LLM result.
            **kwargs (Any): Additional keyword arguments.
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

    @shielded
    async def on_llm_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors.

        Args:
            error (Exception or KeyboardInterrupt): The error.
            kwargs (Any): Additional keyword arguments.
                - response (LLMResult): The response which was generated before
                    the error occurred.



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
            **kwargs (Any): Additional keyword arguments.
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
            **kwargs (Any): Additional keyword arguments.
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
            **kwargs (Any): Additional keyword arguments.

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
            **kwargs (Any): Additional keyword arguments.

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

    def get_sync(self) -> CallbackManagerForChainRun:
        """Get the equivalent sync RunManager.

        Returns:
            CallbackManagerForChainRun: The sync RunManager.
        """
        return CallbackManagerForChainRun(
            run_id=self.run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    @shielded
    async def on_chain_end(
        self, outputs: Union[Dict[str, Any], Any], **kwargs: Any
    ) -> None:
        """Run when a chain ends running.

        Args:
            outputs (Union[Dict[str, Any], Any]): The outputs of the chain.
            **kwargs (Any): Additional keyword arguments.
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

    @shielded
    async def on_chain_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors.

        Args:
            error (Exception or KeyboardInterrupt): The error.
            **kwargs (Any): Additional keyword arguments.
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

    @shielded
    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run when agent action is received.

        Args:
            action (AgentAction): The agent action.
            **kwargs (Any): Additional keyword arguments.

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

    @shielded
    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run when agent finish is received.

        Args:
            finish (AgentFinish): The agent finish.
            **kwargs (Any): Additional keyword arguments.

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
        output: Any,
        **kwargs: Any,
    ) -> None:
        """Run when the tool ends running.

        Args:
            output (Any): The output of the tool.
            **kwargs (Any): Additional keyword arguments.
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
            **kwargs (Any): Additional keyword arguments.
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

    def get_sync(self) -> CallbackManagerForToolRun:
        """Get the equivalent sync RunManager.

        Returns:
            CallbackManagerForToolRun: The sync RunManager.
        """
        return CallbackManagerForToolRun(
            run_id=self.run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    @shielded
    async def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Async run when the tool ends running.

        Args:
            output (Any): The output of the tool.
            **kwargs (Any): Additional keyword arguments.
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

    @shielded
    async def on_tool_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors.

        Args:
            error (Exception or KeyboardInterrupt): The error.
            **kwargs (Any): Additional keyword arguments.
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
        """Run when retriever ends running.

        Args:
            documents (Sequence[Document]): The retrieved documents.
            **kwargs (Any): Additional keyword arguments.
        """
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
        """Run when retriever errors.

        Args:
            error (BaseException): The error.
            **kwargs (Any): Additional keyword arguments.
        """
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

    def get_sync(self) -> CallbackManagerForRetrieverRun:
        """Get the equivalent sync RunManager.

        Returns:
            CallbackManagerForRetrieverRun: The sync RunManager.
        """
        return CallbackManagerForRetrieverRun(
            run_id=self.run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata,
        )

    @shielded
    async def on_retriever_end(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> None:
        """Run when the retriever ends running.

        Args:
            documents (Sequence[Document]): The retrieved documents.
            **kwargs (Any): Additional keyword arguments.
        """
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

    @shielded
    async def on_retriever_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """Run when retriever errors.

        Args:
            error (BaseException): The error.
            **kwargs (Any): Additional keyword arguments.
        """
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
    """Callback manager for LangChain."""

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> List[CallbackManagerForLLMRun]:
        """Run when LLM starts running.

        Args:
            serialized (Dict[str, Any]): The serialized LLM.
            prompts (List[str]): The list of prompts.
            run_id (UUID, optional): The ID of the run. Defaults to None.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            List[CallbackManagerForLLMRun]: A callback manager for each
                prompt as an LLM run.
        """
        managers = []
        for i, prompt in enumerate(prompts):
            # Can't have duplicate runs with the same run ID (if provided)
            run_id_ = run_id if i == 0 and run_id is not None else uuid.uuid4()
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
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> List[CallbackManagerForLLMRun]:
        """Run when LLM starts running.

        Args:
            serialized (Dict[str, Any]): The serialized LLM.
            messages (List[List[BaseMessage]]): The list of messages.
            run_id (UUID, optional): The ID of the run. Defaults to None.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            List[CallbackManagerForLLMRun]: A callback manager for each
                list of messages as an LLM run.
        """

        managers = []
        for message_list in messages:
            if run_id is not None:
                run_id_ = run_id
                run_id = None
            else:
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
            **kwargs (Any): Additional keyword arguments.

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
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> CallbackManagerForToolRun:
        """Run when tool starts running.

        Args:
            serialized: Serialized representation of the tool.
            input_str: The  input to the tool as a string.
                Non-string inputs are cast to strings.
            run_id: ID for the run. Defaults to None.
            parent_run_id: The ID of the parent run. Defaults to None.
            inputs: The original input to the tool if provided.
                Recommended for usage instead of input_str when the original
                input is needed.
                If provided, the inputs are expected to be formatted as a dict.
                The keys will correspond to the named-arguments in the tool.
            **kwargs (Any): Additional keyword arguments.

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
            inputs=inputs,
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
        """Run when the retriever starts running.

        Args:
            serialized (Dict[str, Any]): The serialized retriever.
            query (str): The query.
            run_id (UUID, optional): The ID of the run. Defaults to None.
            parent_run_id (UUID, optional): The ID of the parent run. Defaults to None.
            **kwargs (Any): Additional keyword arguments.
        """
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

    def on_custom_event(
        self,
        name: str,
        data: Any,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Dispatch an adhoc event to the handlers (async version).

        This event should NOT be used in any internal LangChain code. The event
        is meant specifically for users of the library to dispatch custom
        events that are tailored to their application.

        Args:
            name: The name of the adhoc event.
            data: The data for the adhoc event.
            run_id: The ID of the run. Defaults to None.

        .. versionadded:: 0.2.14
        """
        if kwargs:
            raise ValueError(
                "The dispatcher API does not accept additional keyword arguments."
                "Please do not pass any additional keyword arguments, instead "
                "include them in the data field."
            )
        if run_id is None:
            run_id = uuid.uuid4()

        handle_event(
            self.handlers,
            "on_custom_event",
            "ignore_custom_event",
            name,
            data,
            run_id=run_id,
            tags=self.tags,
            metadata=self.metadata,
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
        """Initialize the callback manager.

        Args:
            handlers (List[BaseCallbackHandler]): The list of handlers.
            inheritable_handlers (Optional[List[BaseCallbackHandler]]): The list of
                inheritable handlers. Defaults to None.
            parent_run_id (Optional[UUID]): The ID of the parent run. Defaults to None.
            parent_run_manager (CallbackManagerForChainRun): The parent run manager.
            **kwargs (Any): Additional keyword arguments.
        """
        super().__init__(
            handlers,
            inheritable_handlers,
            parent_run_id,
            **kwargs,
        )
        self.parent_run_manager = parent_run_manager
        self.ended = False

    def copy(self) -> CallbackManagerForChainGroup:
        """Copy the callback manager."""
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
            **kwargs (Any): Additional keyword arguments.
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
            **kwargs (Any): Additional keyword arguments.
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
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> List[AsyncCallbackManagerForLLMRun]:
        """Run when LLM starts running.

        Args:
            serialized (Dict[str, Any]): The serialized LLM.
            prompts (List[str]): The list of prompts.
            run_id (UUID, optional): The ID of the run. Defaults to None.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            List[AsyncCallbackManagerForLLMRun]: The list of async
                callback managers, one for each LLM Run corresponding
                to each prompt.
        """

        tasks = []
        managers = []

        for prompt in prompts:
            if run_id is not None:
                run_id_ = run_id
                run_id = None
            else:
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
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> List[AsyncCallbackManagerForLLMRun]:
        """Async run when LLM starts running.

        Args:
            serialized (Dict[str, Any]): The serialized LLM.
            messages (List[List[BaseMessage]]): The list of messages.
            run_id (UUID, optional): The ID of the run. Defaults to None.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            List[AsyncCallbackManagerForLLMRun]: The list of
                async callback managers, one for each LLM Run
                corresponding to each inner  message list.
        """
        tasks = []
        managers = []

        for message_list in messages:
            if run_id is not None:
                run_id_ = run_id
                run_id = None
            else:
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
        """Async run when chain starts running.

        Args:
            serialized (Dict[str, Any]): The serialized chain.
            inputs (Union[Dict[str, Any], Any]): The inputs to the chain.
            run_id (UUID, optional): The ID of the run. Defaults to None.
            **kwargs (Any): Additional keyword arguments.

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
        """Run when the tool starts running.

        Args:
            serialized (Dict[str, Any]): The serialized tool.
            input_str (str): The input to the tool.
            run_id (UUID, optional): The ID of the run. Defaults to None.
            parent_run_id (UUID, optional): The ID of the parent run.
                Defaults to None.
            **kwargs (Any): Additional keyword arguments.

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

    async def on_custom_event(
        self,
        name: str,
        data: Any,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Dispatch an adhoc event to the handlers (async version).

        This event should NOT be used in any internal LangChain code. The event
        is meant specifically for users of the library to dispatch custom
        events that are tailored to their application.

        Args:
            name: The name of the adhoc event.
            data: The data for the adhoc event.
            run_id: The ID of the run. Defaults to None.

        .. versionadded:: 0.2.14
        """
        if run_id is None:
            run_id = uuid.uuid4()

        if kwargs:
            raise ValueError(
                "The dispatcher API does not accept additional keyword arguments."
                "Please do not pass any additional keyword arguments, instead "
                "include them in the data field."
            )
        await ahandle_event(
            self.handlers,
            "on_custom_event",
            "ignore_custom_event",
            name,
            data,
            run_id=run_id,
            tags=self.tags,
            metadata=self.metadata,
        )

    async def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> AsyncCallbackManagerForRetrieverRun:
        """Run when the retriever starts running.

        Args:
            serialized (Dict[str, Any]): The serialized retriever.
            query (str): The query.
            run_id (UUID, optional): The ID of the run. Defaults to None.
            parent_run_id (UUID, optional): The ID of the parent run. Defaults to None.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            AsyncCallbackManagerForRetrieverRun: The async callback manager
                for the retriever run.
        """
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
        """Initialize the async callback manager.

        Args:
            handlers (List[BaseCallbackHandler]): The list of handlers.
            inheritable_handlers (Optional[List[BaseCallbackHandler]]): The list of
                inheritable handlers. Defaults to None.
            parent_run_id (Optional[UUID]): The ID of the parent run. Defaults to None.
            parent_run_manager (AsyncCallbackManagerForChainRun):
                The parent run manager.
            **kwargs (Any): Additional keyword arguments.
        """
        super().__init__(
            handlers,
            inheritable_handlers,
            parent_run_id,
            **kwargs,
        )
        self.parent_run_manager = parent_run_manager
        self.ended = False

    def copy(self) -> AsyncCallbackManagerForChainGroup:
        """Copy the async callback manager."""
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
            **kwargs (Any): Additional keyword arguments.
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
            **kwargs (Any): Additional keyword arguments.
        """
        self.ended = True
        await self.parent_run_manager.on_chain_error(error, **kwargs)


T = TypeVar("T", CallbackManager, AsyncCallbackManager)


H = TypeVar("H", bound=BaseCallbackHandler, covariant=True)


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
    from langchain_core.tracers.context import (
        _configure_hooks,
        _get_tracer_project,
        _tracing_v2_is_enabled,
        tracing_v2_callback_var,
    )

    run_tree = get_run_tree_context()
    parent_run_id = None if run_tree is None else run_tree.id
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
            parent_run_id_ = inheritable_callbacks.parent_run_id
            # Break ties between the external tracing context and inherited context
            if parent_run_id is not None:
                if parent_run_id_ is None:
                    parent_run_id_ = parent_run_id
                # If the LC parent has already been reflected
                # in the run tree, we know the run_tree is either the
                # same parent or a child of the parent.
                elif run_tree and str(parent_run_id_) in run_tree.dotted_order:
                    parent_run_id_ = parent_run_id
                # Otherwise, we assume the LC context has progressed
                # beyond the run tree and we should not inherit the parent.
            callback_manager = callback_manager_cls(
                handlers=inheritable_callbacks.handlers.copy(),
                inheritable_handlers=inheritable_callbacks.inheritable_handlers.copy(),
                parent_run_id=parent_run_id_,
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

    v1_tracing_enabled_ = env_var_is_set("LANGCHAIN_TRACING") or env_var_is_set(
        "LANGCHAIN_HANDLER"
    )

    tracer_v2 = tracing_v2_callback_var.get()
    tracing_v2_enabled_ = _tracing_v2_is_enabled()

    if v1_tracing_enabled_ and not tracing_v2_enabled_:
        # if both are enabled, can silently ignore the v1 tracer
        raise RuntimeError(
            "Tracing using LangChainTracerV1 is no longer supported. "
            "Please set the LANGCHAIN_TRACING_V2 environment variable to enable "
            "tracing instead."
        )

    tracer_project = _get_tracer_project()
    debug = _get_debug()
    if verbose or debug or tracing_v2_enabled_:
        from langchain_core.tracers.langchain import LangChainTracer
        from langchain_core.tracers.stdout import ConsoleCallbackHandler

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
        if tracing_v2_enabled_ and not any(
            isinstance(handler, LangChainTracer)
            for handler in callback_manager.handlers
        ):
            if tracer_v2:
                callback_manager.add_handler(tracer_v2, True)
            else:
                try:
                    handler = LangChainTracer(
                        project_name=tracer_project,
                        client=run_tree.client if run_tree is not None else None,
                    )
                    callback_manager.add_handler(handler, True)
                except Exception as e:
                    logger.warning(
                        "Unable to load requested LangChainTracer."
                        " To disable this warning,"
                        " unset the LANGCHAIN_TRACING_V2 environment variables.\n"
                        f"{repr(e)}",
                    )
        if run_tree is not None:
            for handler in callback_manager.handlers:
                if isinstance(handler, LangChainTracer):
                    handler.order_map[run_tree.id] = (
                        run_tree.trace_id,
                        run_tree.dotted_order,
                    )
                    handler.run_map[str(run_tree.id)] = cast(Run, run_tree)
    for var, inheritable, handler_class, env_var in _configure_hooks:
        create_one = (
            env_var is not None
            and env_var_is_set(env_var)
            and handler_class is not None
        )
        if var.get() is not None or create_one:
            var_handler = var.get() or cast(Type[BaseCallbackHandler], handler_class)()
            if handler_class is None:
                if not any(
                    handler is var_handler  # direct pointer comparison
                    for handler in callback_manager.handlers
                ):
                    callback_manager.add_handler(var_handler, inheritable)
            else:
                if not any(
                    isinstance(handler, handler_class)
                    for handler in callback_manager.handlers
                ):
                    callback_manager.add_handler(var_handler, inheritable)
    return callback_manager


async def adispatch_custom_event(
    name: str, data: Any, *, config: Optional[RunnableConfig] = None
) -> None:
    """Dispatch an adhoc event to the handlers.

    Args:
        name: The name of the adhoc event.
        data: The data for the adhoc event. Free form data. Ideally should be
              JSON serializable to avoid serialization issues downstream, but
              this is not enforced.
        config: Optional config object. Mirrors the async API but not strictly needed.

    Example:

        .. code-block:: python

            from langchain_core.callbacks import (
                AsyncCallbackHandler,
                adispatch_custom_event
            )
            from langchain_core.runnable import RunnableLambda

            class CustomCallbackManager(AsyncCallbackHandler):
                async def on_custom_event(
                    self,
                    name: str,
                    data: Any,
                    *,
                    run_id: UUID,
                    tags: Optional[List[str]] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    **kwargs: Any,
                ) -> None:
                    print(f"Received custom event: {name} with data: {data}")

            callback = CustomCallbackManager()

            async def foo(inputs):
                await adispatch_custom_event("my_event", {"bar": "buzz})
                return inputs

            foo_ = RunnableLambda(foo)
            await foo_.ainvoke({"a": "1"}, {"callbacks": [CustomCallbackManager()]})

    Example: Use with astream events

        .. code-block:: python

            from langchain_core.callbacks import (
                AsyncCallbackHandler,
                adispatch_custom_event
            )
            from langchain_core.runnable import RunnableLambda

            class CustomCallbackManager(AsyncCallbackHandler):
                async def on_custom_event(
                    self,
                    name: str,
                    data: Any,
                    *,
                    run_id: UUID,
                    tags: Optional[List[str]] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    **kwargs: Any,
                ) -> None:
                    print(f"Received custom event: {name} with data: {data}")

            callback = CustomCallbackManager()

            async def foo(inputs):
                await adispatch_custom_event("event_type_1", {"bar": "buzz})
                await adispatch_custom_event("event_type_2", 5)
                return inputs

            foo_ = RunnableLambda(foo)

            async for event in foo_.ainvoke_stream(
                {"a": "1"},
                version="v2",
                config={"callbacks": [CustomCallbackManager()]}
            ):
                print(event)

    .. warning: If using python <= 3.10 and async, you MUST
        specify the `config` parameter or the function will raise an error.
        This is due to a limitation in asyncio for python <= 3.10 that prevents
        LangChain from automatically propagating the config object on the user's
        behalf.

    .. versionadded:: 0.2.15
    """
    from langchain_core.runnables.config import (
        ensure_config,
        get_async_callback_manager_for_config,
    )

    config = ensure_config(config)
    callback_manager = get_async_callback_manager_for_config(config)
    # We want to get the callback manager for the parent run.
    # This is a work-around for now to be able to dispatch adhoc events from
    # within a tool or a lambda and have the metadata events associated
    # with the parent run rather than have a new run id generated for each.
    if callback_manager.parent_run_id is None:
        raise RuntimeError(
            "Unable to dispatch an adhoc event without a parent run id."
            "This function can only be called from within an existing run (e.g.,"
            "inside a tool or a RunnableLambda or a RunnableGenerator.)"
            "If you are doing that and still seeing this error, try explicitly"
            "passing the config parameter to this function."
        )

    await callback_manager.on_custom_event(
        name,
        data,
        run_id=callback_manager.parent_run_id,
    )


def dispatch_custom_event(
    name: str, data: Any, *, config: Optional[RunnableConfig] = None
) -> None:
    """Dispatch an adhoc event.

    Args:
        name: The name of the adhoc event.
        data: The data for the adhoc event. Free form data. Ideally should be
              JSON serializable to avoid serialization issues downstream, but
              this is not enforced.
        config: Optional config object. Mirrors the async API but not strictly needed.

    Example:

        .. code-block:: python

            from langchain_core.callbacks import BaseCallbackHandler
            from langchain_core.callbacks import dispatch_custom_event
            from langchain_core.runnable import RunnableLambda

            class CustomCallbackManager(BaseCallbackHandler):
                def on_custom_event(
                    self,
                    name: str,
                    data: Any,
                    *,
                    run_id: UUID,
                    tags: Optional[List[str]] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    **kwargs: Any,
                ) -> None:
                    print(f"Received custom event: {name} with data: {data}")

            def foo(inputs):
                dispatch_custom_event("my_event", {"bar": "buzz})
                return inputs

            foo_ = RunnableLambda(foo)
            foo_.invoke({"a": "1"}, {"callbacks": [CustomCallbackManager()]})

    .. versionadded:: 0.2.15
    """
    from langchain_core.runnables.config import (
        ensure_config,
        get_callback_manager_for_config,
    )

    config = ensure_config(config)
    callback_manager = get_callback_manager_for_config(config)
    # We want to get the callback manager for the parent run.
    # This is a work-around for now to be able to dispatch adhoc events from
    # within a tool or a lambda and have the metadata events associated
    # with the parent run rather than have a new run id generated for each.
    if callback_manager.parent_run_id is None:
        raise RuntimeError(
            "Unable to dispatch an adhoc event without a parent run id."
            "This function can only be called from within an existing run (e.g.,"
            "inside a tool or a RunnableLambda or a RunnableGenerator.)"
            "If you are doing that and still seeing this error, try explicitly"
            "passing the config parameter to this function."
        )
    callback_manager.on_custom_event(
        name,
        data,
        run_id=callback_manager.parent_run_id,
    )
