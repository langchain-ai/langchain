from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Iterator
from typing import (
    TYPE_CHECKING,
    Any,
)
from uuid import UUID

from langchain_core.agents import (
    AgentAction,
    AgentFinish,
    AgentStep,
)
from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
    CallbackManager,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain_core.load.dump import dumpd
from langchain_core.outputs import RunInfo
from langchain_core.runnables.utils import AddableDict
from langchain_core.tools import BaseTool
from langchain_core.utils.input import get_color_mapping

from langchain_classic.schema import RUN_KEY
from langchain_classic.utilities.asyncio import asyncio_timeout

if TYPE_CHECKING:
    from langchain_classic.agents.agent import AgentExecutor, NextStepOutput

logger = logging.getLogger(__name__)


class AgentExecutorIterator:
    """Iterator for AgentExecutor."""

    def __init__(
        self,
        agent_executor: AgentExecutor,
        inputs: Any,
        callbacks: Callbacks = None,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        run_name: str | None = None,
        run_id: UUID | None = None,
        include_run_info: bool = False,
        yield_actions: bool = False,
    ):
        """Initialize the `AgentExecutorIterator`.

        Initialize the `AgentExecutorIterator` with the given `AgentExecutor`,
        inputs, and optional callbacks.

        Args:
            agent_executor: The `AgentExecutor` to iterate over.
            inputs: The inputs to the `AgentExecutor`.
            callbacks: The callbacks to use during iteration.
            tags: The tags to use during iteration.
            metadata: The metadata to use during iteration.
            run_name: The name of the run.
            run_id: The ID of the run.
            include_run_info: Whether to include run info in the output.
            yield_actions: Whether to yield actions as they are generated.
        """
        self._agent_executor = agent_executor
        self.inputs = inputs
        self.callbacks = callbacks
        self.tags = tags
        self.metadata = metadata
        self.run_name = run_name
        self.run_id = run_id
        self.include_run_info = include_run_info
        self.yield_actions = yield_actions
        self.reset()

    _inputs: dict[str, str]
    callbacks: Callbacks
    tags: list[str] | None
    metadata: dict[str, Any] | None
    run_name: str | None
    run_id: UUID | None
    include_run_info: bool
    yield_actions: bool

    @property
    def inputs(self) -> dict[str, str]:
        """The inputs to the `AgentExecutor`."""
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: Any) -> None:
        self._inputs = self.agent_executor.prep_inputs(inputs)

    @property
    def agent_executor(self) -> AgentExecutor:
        """The `AgentExecutor` to iterate over."""
        return self._agent_executor

    @agent_executor.setter
    def agent_executor(self, agent_executor: AgentExecutor) -> None:
        self._agent_executor = agent_executor
        # force re-prep inputs in case agent_executor's prep_inputs fn changed
        self.inputs = self.inputs

    @property
    def name_to_tool_map(self) -> dict[str, BaseTool]:
        """A mapping of tool names to tools."""
        return {tool.name: tool for tool in self.agent_executor.tools}

    @property
    def color_mapping(self) -> dict[str, str]:
        """A mapping of tool names to colors."""
        return get_color_mapping(
            [tool.name for tool in self.agent_executor.tools],
            excluded_colors=["green", "red"],
        )

    def reset(self) -> None:
        """Reset the iterator to its initial state.

        Reset the iterator to its initial state, clearing intermediate steps,
        iterations, and time elapsed.
        """
        logger.debug("(Re)setting AgentExecutorIterator to fresh state")
        self.intermediate_steps: list[tuple[AgentAction, str]] = []
        self.iterations = 0
        # maybe better to start these on the first __anext__ call?
        self.time_elapsed = 0.0
        self.start_time = time.time()

    def update_iterations(self) -> None:
        """Increment the number of iterations and update the time elapsed."""
        self.iterations += 1
        self.time_elapsed = time.time() - self.start_time
        logger.debug(
            "Agent Iterations: %s (%.2fs elapsed)",
            self.iterations,
            self.time_elapsed,
        )

    def make_final_outputs(
        self,
        outputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | AsyncCallbackManagerForChainRun,
    ) -> AddableDict:
        """Make final outputs for the iterator.

        Args:
            outputs: The outputs from the agent executor.
            run_manager: The run manager to use for callbacks.
        """
        # have access to intermediate steps by design in iterator,
        # so return only outputs may as well always be true.

        prepared_outputs = AddableDict(
            self.agent_executor.prep_outputs(
                self.inputs,
                outputs,
                return_only_outputs=True,
            ),
        )
        if self.include_run_info:
            prepared_outputs[RUN_KEY] = RunInfo(run_id=run_manager.run_id)
        return prepared_outputs

    def __iter__(self: AgentExecutorIterator) -> Iterator[AddableDict]:
        """Create an async iterator for the `AgentExecutor`."""
        logger.debug("Initialising AgentExecutorIterator")
        self.reset()
        callback_manager = CallbackManager.configure(
            self.callbacks,
            self.agent_executor.callbacks,
            self.agent_executor.verbose,
            self.tags,
            self.agent_executor.tags,
            self.metadata,
            self.agent_executor.metadata,
        )
        run_manager = callback_manager.on_chain_start(
            dumpd(self.agent_executor),
            self.inputs,
            self.run_id,
            name=self.run_name,
        )
        try:
            while self.agent_executor._should_continue(  # noqa: SLF001
                self.iterations,
                self.time_elapsed,
            ):
                # take the next step: this plans next action, executes it,
                # yielding action and observation as they are generated
                next_step_seq: NextStepOutput = []
                for chunk in self.agent_executor._iter_next_step(  # noqa: SLF001
                    self.name_to_tool_map,
                    self.color_mapping,
                    self.inputs,
                    self.intermediate_steps,
                    run_manager,
                ):
                    next_step_seq.append(chunk)
                    # if we're yielding actions, yield them as they come
                    # do not yield AgentFinish, which will be handled below
                    if self.yield_actions:
                        if isinstance(chunk, AgentAction):
                            yield AddableDict(actions=[chunk], messages=chunk.messages)
                        elif isinstance(chunk, AgentStep):
                            yield AddableDict(steps=[chunk], messages=chunk.messages)

                # convert iterator output to format handled by _process_next_step_output
                next_step = self.agent_executor._consume_next_step(next_step_seq)  # noqa: SLF001
                # update iterations and time elapsed
                self.update_iterations()
                # decide if this is the final output
                output = self._process_next_step_output(next_step, run_manager)
                is_final = "intermediate_step" not in output
                # yield the final output always
                # for backwards compat, yield int. output if not yielding actions
                if not self.yield_actions or is_final:
                    yield output
                # if final output reached, stop iteration
                if is_final:
                    return
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise

        # if we got here means we exhausted iterations or time
        yield self._stop(run_manager)

    async def __aiter__(self) -> AsyncIterator[AddableDict]:
        """Create an async iterator for the `AgentExecutor`.

        N.B. __aiter__ must be a normal method, so need to initialize async run manager
        on first __anext__ call where we can await it.
        """
        logger.debug("Initialising AgentExecutorIterator (async)")
        self.reset()
        callback_manager = AsyncCallbackManager.configure(
            self.callbacks,
            self.agent_executor.callbacks,
            self.agent_executor.verbose,
            self.tags,
            self.agent_executor.tags,
            self.metadata,
            self.agent_executor.metadata,
        )
        run_manager = await callback_manager.on_chain_start(
            dumpd(self.agent_executor),
            self.inputs,
            self.run_id,
            name=self.run_name,
        )
        try:
            async with asyncio_timeout(self.agent_executor.max_execution_time):
                while self.agent_executor._should_continue(  # noqa: SLF001
                    self.iterations,
                    self.time_elapsed,
                ):
                    # take the next step: this plans next action, executes it,
                    # yielding action and observation as they are generated
                    next_step_seq: NextStepOutput = []
                    async for chunk in self.agent_executor._aiter_next_step(  # noqa: SLF001
                        self.name_to_tool_map,
                        self.color_mapping,
                        self.inputs,
                        self.intermediate_steps,
                        run_manager,
                    ):
                        next_step_seq.append(chunk)
                        # if we're yielding actions, yield them as they come
                        # do not yield AgentFinish, which will be handled below
                        if self.yield_actions:
                            if isinstance(chunk, AgentAction):
                                yield AddableDict(
                                    actions=[chunk],
                                    messages=chunk.messages,
                                )
                            elif isinstance(chunk, AgentStep):
                                yield AddableDict(
                                    steps=[chunk],
                                    messages=chunk.messages,
                                )

                    # convert iterator output to format handled by _process_next_step
                    next_step = self.agent_executor._consume_next_step(next_step_seq)  # noqa: SLF001
                    # update iterations and time elapsed
                    self.update_iterations()
                    # decide if this is the final output
                    output = await self._aprocess_next_step_output(
                        next_step,
                        run_manager,
                    )
                    is_final = "intermediate_step" not in output
                    # yield the final output always
                    # for backwards compat, yield int. output if not yielding actions
                    if not self.yield_actions or is_final:
                        yield output
                    # if final output reached, stop iteration
                    if is_final:
                        return
        except (TimeoutError, asyncio.TimeoutError):
            yield await self._astop(run_manager)
            return
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise

        # if we got here means we exhausted iterations or time
        yield await self._astop(run_manager)

    def _process_next_step_output(
        self,
        next_step_output: AgentFinish | list[tuple[AgentAction, str]],
        run_manager: CallbackManagerForChainRun,
    ) -> AddableDict:
        """Process the output of the next step.

        Process the output of the next step,
        handling AgentFinish and tool return cases.
        """
        logger.debug("Processing output of Agent loop step")
        if isinstance(next_step_output, AgentFinish):
            logger.debug(
                "Hit AgentFinish: _return -> on_chain_end -> run final output logic",
            )
            return self._return(next_step_output, run_manager=run_manager)

        self.intermediate_steps.extend(next_step_output)
        logger.debug("Updated intermediate_steps with step output")

        # Check for tool return
        if len(next_step_output) == 1:
            next_step_action = next_step_output[0]
            tool_return = self.agent_executor._get_tool_return(next_step_action)  # noqa: SLF001
            if tool_return is not None:
                return self._return(tool_return, run_manager=run_manager)

        return AddableDict(intermediate_step=next_step_output)

    async def _aprocess_next_step_output(
        self,
        next_step_output: AgentFinish | list[tuple[AgentAction, str]],
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> AddableDict:
        """Process the output of the next async step.

        Process the output of the next async step,
        handling AgentFinish and tool return cases.
        """
        logger.debug("Processing output of async Agent loop step")
        if isinstance(next_step_output, AgentFinish):
            logger.debug(
                "Hit AgentFinish: _areturn -> on_chain_end -> run final output logic",
            )
            return await self._areturn(next_step_output, run_manager=run_manager)

        self.intermediate_steps.extend(next_step_output)
        logger.debug("Updated intermediate_steps with step output")

        # Check for tool return
        if len(next_step_output) == 1:
            next_step_action = next_step_output[0]
            tool_return = self.agent_executor._get_tool_return(next_step_action)  # noqa: SLF001
            if tool_return is not None:
                return await self._areturn(tool_return, run_manager=run_manager)

        return AddableDict(intermediate_step=next_step_output)

    def _stop(self, run_manager: CallbackManagerForChainRun) -> AddableDict:
        """Stop the iterator.

        Stop the iterator and raise a StopIteration exception with the stopped response.
        """
        logger.warning("Stopping agent prematurely due to triggering stop condition")
        # this manually constructs agent finish with output key
        output = self.agent_executor._action_agent.return_stopped_response(  # noqa: SLF001
            self.agent_executor.early_stopping_method,
            self.intermediate_steps,
            **self.inputs,
        )
        return self._return(output, run_manager=run_manager)

    async def _astop(self, run_manager: AsyncCallbackManagerForChainRun) -> AddableDict:
        """Stop the async iterator.

        Stop the async iterator and raise a StopAsyncIteration exception with
        the stopped response.
        """
        logger.warning("Stopping agent prematurely due to triggering stop condition")
        output = self.agent_executor._action_agent.return_stopped_response(  # noqa: SLF001
            self.agent_executor.early_stopping_method,
            self.intermediate_steps,
            **self.inputs,
        )
        return await self._areturn(output, run_manager=run_manager)

    def _return(
        self,
        output: AgentFinish,
        run_manager: CallbackManagerForChainRun,
    ) -> AddableDict:
        """Return the final output of the iterator."""
        returned_output = self.agent_executor._return(  # noqa: SLF001
            output,
            self.intermediate_steps,
            run_manager=run_manager,
        )
        returned_output["messages"] = output.messages
        run_manager.on_chain_end(returned_output)
        return self.make_final_outputs(returned_output, run_manager)

    async def _areturn(
        self,
        output: AgentFinish,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> AddableDict:
        """Return the final output of the async iterator."""
        returned_output = await self.agent_executor._areturn(  # noqa: SLF001
            output,
            self.intermediate_steps,
            run_manager=run_manager,
        )
        returned_output["messages"] = output.messages
        await run_manager.on_chain_end(returned_output)
        return self.make_final_outputs(returned_output, run_manager)
