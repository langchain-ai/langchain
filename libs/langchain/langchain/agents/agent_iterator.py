from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from asyncio import CancelledError
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    NoReturn,
    Optional,
    Tuple,
    Type,
    Union,
)

from langchain.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
    CallbackManager,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain.load.dump import dumpd
from langchain.schema import RUN_KEY, AgentAction, AgentFinish, RunInfo
from langchain.tools import BaseTool
from langchain.utilities.asyncio import asyncio_timeout
from langchain.utils.input import get_color_mapping

if TYPE_CHECKING:
    from langchain.agents.agent import AgentExecutor

logger = logging.getLogger(__name__)


class BaseAgentExecutorIterator(ABC):
    """Base class for AgentExecutorIterator."""

    @abstractmethod
    def build_callback_manager(self) -> None:
        pass


def rebuild_callback_manager_on_set(
    setter_method: Callable[..., None]
) -> Callable[..., None]:
    """Decorator to force setters to rebuild callback mgr"""

    @wraps(setter_method)
    def wrapper(self: BaseAgentExecutorIterator, *args: Any, **kwargs: Any) -> None:
        setter_method(self, *args, **kwargs)
        self.build_callback_manager()

    return wrapper


class AgentExecutorIterator(BaseAgentExecutorIterator):
    """Iterator for AgentExecutor."""

    def __init__(
        self,
        agent_executor: AgentExecutor,
        inputs: Any,
        callbacks: Callbacks = None,
        *,
        tags: Optional[list[str]] = None,
        include_run_info: bool = False,
        async_: bool = False,
    ):
        """
        Initialize the AgentExecutorIterator with the given AgentExecutor,
        inputs, and optional callbacks.
        """
        self._agent_executor = agent_executor
        self.inputs = inputs
        self.async_ = async_
        # build callback manager on tags setter
        self._callbacks = callbacks
        self.tags = tags
        self.include_run_info = include_run_info
        self.run_manager = None
        self.reset()

    _callback_manager: Union[AsyncCallbackManager, CallbackManager]
    _inputs: dict[str, str]
    _final_outputs: Optional[dict[str, str]]
    run_manager: Optional[
        Union[AsyncCallbackManagerForChainRun, CallbackManagerForChainRun]
    ]
    timeout_manager: Any  # TODO: Fix a type here; the shim makes it tricky.

    @property
    def inputs(self) -> dict[str, str]:
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: Any) -> None:
        self._inputs = self.agent_executor.prep_inputs(inputs)

    @property
    def callbacks(self) -> Callbacks:
        return self._callbacks

    @callbacks.setter
    @rebuild_callback_manager_on_set
    def callbacks(self, callbacks: Callbacks) -> None:
        """When callbacks are changed after __init__, rebuild callback mgr"""
        self._callbacks = callbacks

    @property
    def tags(self) -> Optional[List[str]]:
        return self._tags

    @tags.setter
    @rebuild_callback_manager_on_set
    def tags(self, tags: Optional[List[str]]) -> None:
        """When tags are changed after __init__, rebuild callback mgr"""
        self._tags = tags

    @property
    def agent_executor(self) -> AgentExecutor:
        return self._agent_executor

    @agent_executor.setter
    @rebuild_callback_manager_on_set
    def agent_executor(self, agent_executor: AgentExecutor) -> None:
        self._agent_executor = agent_executor
        # force re-prep inputs in case agent_executor's prep_inputs fn changed
        self.inputs = self.inputs

    @property
    def callback_manager(self) -> Union[AsyncCallbackManager, CallbackManager]:
        return self._callback_manager

    def build_callback_manager(self) -> None:
        """
        Create and configure the callback manager based on the current
        callbacks and tags.
        """
        CallbackMgr: Union[Type[AsyncCallbackManager], Type[CallbackManager]] = (
            AsyncCallbackManager if self.async_ else CallbackManager
        )
        self._callback_manager = CallbackMgr.configure(
            self.callbacks,
            self.agent_executor.callbacks,
            self.agent_executor.verbose,
            self.tags,
            self.agent_executor.tags,
        )

    @property
    def name_to_tool_map(self) -> dict[str, BaseTool]:
        return {tool.name: tool for tool in self.agent_executor.tools}

    @property
    def color_mapping(self) -> dict[str, str]:
        return get_color_mapping(
            [tool.name for tool in self.agent_executor.tools],
            excluded_colors=["green", "red"],
        )

    def reset(self) -> None:
        """
        Reset the iterator to its initial state, clearing intermediate steps,
        iterations, and time elapsed.
        """
        logger.debug("(Re)setting AgentExecutorIterator to fresh state")
        self.intermediate_steps: list[tuple[AgentAction, str]] = []
        self.iterations = 0
        # maybe better to start these on the first __anext__ call?
        self.time_elapsed = 0.0
        self.start_time = time.time()
        self._final_outputs = None

    def update_iterations(self) -> None:
        """
        Increment the number of iterations and update the time elapsed.
        """
        self.iterations += 1
        self.time_elapsed = time.time() - self.start_time
        logger.debug(
            f"Agent Iterations: {self.iterations} ({self.time_elapsed:.2f}s elapsed)"
        )

    def raise_stopiteration(self, output: Any) -> NoReturn:
        """
        Raise a StopIteration exception with the given output.
        """
        logger.debug("Chain end: stop iteration")
        raise StopIteration(output)

    async def raise_stopasynciteration(self, output: Any) -> NoReturn:
        """
        Raise a StopAsyncIteration exception with the given output.
        Close the timeout context manager.
        """
        logger.debug("Chain end: stop async iteration")
        if self.timeout_manager is not None:
            await self.timeout_manager.__aexit__(None, None, None)
        raise StopAsyncIteration(output)

    @property
    def final_outputs(self) -> Optional[dict[str, Any]]:
        return self._final_outputs

    @final_outputs.setter
    def final_outputs(self, outputs: Optional[Dict[str, Any]]) -> None:
        # have access to intermediate steps by design in iterator,
        # so return only outputs may as well always be true.

        self._final_outputs = None
        if outputs:
            prepared_outputs: dict[str, Any] = self.agent_executor.prep_outputs(
                self.inputs, outputs, return_only_outputs=True
            )
            if self.include_run_info and self.run_manager is not None:
                logger.debug("Assign run key")
                prepared_outputs[RUN_KEY] = RunInfo(run_id=self.run_manager.run_id)
            self._final_outputs = prepared_outputs

    def __iter__(self: "AgentExecutorIterator") -> "AgentExecutorIterator":
        logger.debug("Initialising AgentExecutorIterator")
        self.reset()
        assert isinstance(self.callback_manager, CallbackManager)
        self.run_manager = self.callback_manager.on_chain_start(
            dumpd(self.agent_executor),
            self.inputs,
        )
        return self

    def __aiter__(self) -> "AgentExecutorIterator":
        """
        N.B. __aiter__ must be a normal method, so need to initialise async run manager
        on first __anext__ call where we can await it
        """
        logger.debug("Initialising AgentExecutorIterator (async)")
        self.reset()
        if self.agent_executor.max_execution_time:
            self.timeout_manager = asyncio_timeout(
                self.agent_executor.max_execution_time
            )
        else:
            self.timeout_manager = None
        return self

    def _on_first_step(self) -> None:
        """
        Perform any necessary setup for the first step of the synchronous iterator.
        """
        pass

    async def _on_first_async_step(self) -> None:
        """
        Perform any necessary setup for the first step of the asynchronous iterator.
        """
        # on first step, need to await callback manager and start async timeout ctxmgr
        if self.iterations == 0:
            assert isinstance(self.callback_manager, AsyncCallbackManager)
            self.run_manager = await self.callback_manager.on_chain_start(
                dumpd(self.agent_executor),
                self.inputs,
            )
            if self.timeout_manager:
                await self.timeout_manager.__aenter__()

    def __next__(self) -> dict[str, Any]:
        """
        AgentExecutor               AgentExecutorIterator
        __call__                    (__iter__ ->) __next__
            _call              <=>      _call_next
                _take_next_step             _take_next_step
        """
        # first step
        if self.iterations == 0:
            self._on_first_step()
        # N.B. timeout taken care of by "_should_continue" in sync case
        try:
            return self._call_next()
        except StopIteration:
            raise
        except (KeyboardInterrupt, Exception) as e:
            if self.run_manager:
                self.run_manager.on_chain_error(e)
            raise

    async def __anext__(self) -> dict[str, Any]:
        """
        AgentExecutor               AgentExecutorIterator
        acall                       (__aiter__ ->) __anext__
            _acall              <=>     _acall_next
                _atake_next_step            _atake_next_step
        """
        if self.iterations == 0:
            await self._on_first_async_step()
        try:
            return await self._acall_next()
        except StopAsyncIteration:
            raise
        except (TimeoutError, CancelledError):
            await self.timeout_manager.__aexit__(None, None, None)
            self.timeout_manager = None
            return await self._astop()
        except (KeyboardInterrupt, Exception) as e:
            if self.run_manager:
                assert isinstance(self.run_manager, AsyncCallbackManagerForChainRun)
                await self.run_manager.on_chain_error(e)
            raise

    def _execute_next_step(
        self, run_manager: Optional[CallbackManagerForChainRun]
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """
        Execute the next step in the chain using the
        AgentExecutor's _take_next_step method.
        """
        return self.agent_executor._take_next_step(
            self.name_to_tool_map,
            self.color_mapping,
            self.inputs,
            self.intermediate_steps,
            run_manager=run_manager,
        )

    async def _execute_next_async_step(
        self, run_manager: Optional[AsyncCallbackManagerForChainRun]
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """
        Execute the next step in the chain using the
        AgentExecutor's _atake_next_step method.
        """
        return await self.agent_executor._atake_next_step(
            self.name_to_tool_map,
            self.color_mapping,
            self.inputs,
            self.intermediate_steps,
            run_manager=run_manager,
        )

    def _process_next_step_output(
        self,
        next_step_output: Union[AgentFinish, List[Tuple[AgentAction, str]]],
        run_manager: Optional[CallbackManagerForChainRun],
    ) -> Dict[str, Union[str, List[Tuple[AgentAction, str]]]]:
        """
        Process the output of the next step,
        handling AgentFinish and tool return cases.
        """
        logger.debug("Processing output of Agent loop step")
        if isinstance(next_step_output, AgentFinish):
            logger.debug(
                "Hit AgentFinish: _return -> on_chain_end -> run final output logic"
            )
            output = self.agent_executor._return(
                next_step_output, self.intermediate_steps, run_manager=run_manager
            )
            if self.run_manager:
                self.run_manager.on_chain_end(output)
            self.final_outputs = output
            return output

        self.intermediate_steps.extend(next_step_output)
        logger.debug("Updated intermediate_steps with step output")

        # Check for tool return
        if len(next_step_output) == 1:
            next_step_action = next_step_output[0]
            tool_return = self.agent_executor._get_tool_return(next_step_action)
            if tool_return is not None:
                output = self.agent_executor._return(
                    tool_return, self.intermediate_steps, run_manager=run_manager
                )
                if self.run_manager:
                    self.run_manager.on_chain_end(output)
                self.final_outputs = output
                return output

        output = {"intermediate_step": next_step_output}
        return output

    async def _aprocess_next_step_output(
        self,
        next_step_output: Union[AgentFinish, List[Tuple[AgentAction, str]]],
        run_manager: Optional[AsyncCallbackManagerForChainRun],
    ) -> Dict[str, Union[str, List[Tuple[AgentAction, str]]]]:
        """
        Process the output of the next async step,
        handling AgentFinish and tool return cases.
        """
        logger.debug("Processing output of async Agent loop step")
        if isinstance(next_step_output, AgentFinish):
            logger.debug(
                "Hit AgentFinish: _areturn -> on_chain_end -> run final output logic"
            )
            output = await self.agent_executor._areturn(
                next_step_output, self.intermediate_steps, run_manager=run_manager
            )
            if run_manager:
                await run_manager.on_chain_end(output)
            self.final_outputs = output
            return output

        self.intermediate_steps.extend(next_step_output)
        logger.debug("Updated intermediate_steps with step output")

        # Check for tool return
        if len(next_step_output) == 1:
            next_step_action = next_step_output[0]
            tool_return = self.agent_executor._get_tool_return(next_step_action)
            if tool_return is not None:
                output = await self.agent_executor._areturn(
                    tool_return, self.intermediate_steps, run_manager=run_manager
                )
                if run_manager:
                    await run_manager.on_chain_end(output)
                self.final_outputs = output
                return output

        output = {"intermediate_step": next_step_output}
        return output

    def _stop(self) -> dict[str, Any]:
        """
        Stop the iterator and raise a StopIteration exception with the stopped response.
        """
        logger.warning("Stopping agent prematurely due to triggering stop condition")
        # this manually constructs agent finish with output key
        output = self.agent_executor.agent.return_stopped_response(
            self.agent_executor.early_stopping_method,
            self.intermediate_steps,
            **self.inputs,
        )
        assert (
            isinstance(self.run_manager, CallbackManagerForChainRun)
            or self.run_manager is None
        )
        returned_output = self.agent_executor._return(
            output, self.intermediate_steps, run_manager=self.run_manager
        )
        self.final_outputs = returned_output
        return returned_output

    async def _astop(self) -> dict[str, Any]:
        """
        Stop the async iterator and raise a StopAsyncIteration exception with
        the stopped response.
        """
        logger.warning("Stopping agent prematurely due to triggering stop condition")
        output = self.agent_executor.agent.return_stopped_response(
            self.agent_executor.early_stopping_method,
            self.intermediate_steps,
            **self.inputs,
        )
        assert (
            isinstance(self.run_manager, AsyncCallbackManagerForChainRun)
            or self.run_manager is None
        )
        returned_output = await self.agent_executor._areturn(
            output, self.intermediate_steps, run_manager=self.run_manager
        )
        self.final_outputs = returned_output
        return returned_output

    def _call_next(self) -> dict[str, Any]:
        """
        Perform a single iteration of the synchronous AgentExecutorIterator.
        """
        # final output already reached: stopiteration (final output)
        if self.final_outputs is not None:
            self.raise_stopiteration(self.final_outputs)
        # timeout/max iterations: stopiteration (stopped response)
        if not self.agent_executor._should_continue(self.iterations, self.time_elapsed):
            return self._stop()
        assert (
            isinstance(self.run_manager, CallbackManagerForChainRun)
            or self.run_manager is None
        )
        next_step_output = self._execute_next_step(self.run_manager)
        output = self._process_next_step_output(next_step_output, self.run_manager)
        self.update_iterations()
        return output

    async def _acall_next(self) -> dict[str, Any]:
        """
        Perform a single iteration of the asynchronous AgentExecutorIterator.
        """
        # final output already reached: stopiteration (final output)
        if self.final_outputs is not None:
            await self.raise_stopasynciteration(self.final_outputs)
        # timeout/max iterations: stopiteration (stopped response)
        if not self.agent_executor._should_continue(self.iterations, self.time_elapsed):
            return await self._astop()
        assert (
            isinstance(self.run_manager, AsyncCallbackManagerForChainRun)
            or self.run_manager is None
        )
        next_step_output = await self._execute_next_async_step(self.run_manager)
        output = await self._aprocess_next_step_output(
            next_step_output, self.run_manager
        )
        self.update_iterations()
        return output
