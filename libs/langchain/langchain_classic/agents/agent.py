"""Chain that takes in an input and produces an action and action input."""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import json
import logging
import time
from abc import abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from pathlib import Path
from typing import (
    Any,
    cast,
)

import yaml
from langchain_core._api import deprecated
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForToolRun,
    BaseCallbackManager,
    CallbackManagerForChainRun,
    CallbackManagerForToolRun,
    Callbacks,
)
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, ensure_config
from langchain_core.runnables.utils import AddableDict
from langchain_core.tools import BaseTool
from langchain_core.utils.input import get_color_mapping
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self, override

from langchain_classic._api.deprecation import AGENT_DEPRECATION_WARNING
from langchain_classic.agents.agent_iterator import AgentExecutorIterator
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents.tools import InvalidTool
from langchain_classic.chains.base import Chain
from langchain_classic.chains.llm import LLMChain
from langchain_classic.utilities.asyncio import asyncio_timeout

logger = logging.getLogger(__name__)


class BaseSingleActionAgent(BaseModel):
    """Base Single Action Agent class."""

    @property
    def return_values(self) -> list[str]:
        """Return values of the agent."""
        return ["output"]

    def get_allowed_tools(self) -> list[str] | None:
        """Get allowed tools."""
        return None

    @abstractmethod
    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """

    @abstractmethod
    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """Async given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """

    @property
    @abstractmethod
    def input_keys(self) -> list[str]:
        """Return the input keys.

        :meta private:
        """

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: list[tuple[AgentAction, str]],  # noqa: ARG002
        **_: Any,
    ) -> AgentFinish:
        """Return response when agent has been stopped due to max iterations.

        Args:
            early_stopping_method: Method to use for early stopping.
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.

        Returns:
            Agent finish object.

        Raises:
            ValueError: If `early_stopping_method` is not supported.
        """
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish(
                {"output": "Agent stopped due to iteration limit or time limit."},
                "",
            )
        msg = f"Got unsupported early_stopping_method `{early_stopping_method}`"
        raise ValueError(msg)

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: BaseCallbackManager | None = None,
        **kwargs: Any,
    ) -> BaseSingleActionAgent:
        """Construct an agent from an LLM and tools.

        Args:
            llm: Language model to use.
            tools: Tools to use.
            callback_manager: Callback manager to use.
            kwargs: Additional arguments.

        Returns:
            Agent object.
        """
        raise NotImplementedError

    @property
    def _agent_type(self) -> str:
        """Return Identifier of an agent type."""
        raise NotImplementedError

    @override
    def dict(self, **kwargs: Any) -> builtins.dict:
        """Return dictionary representation of agent.

        Returns:
            Dictionary representation of agent.
        """
        _dict = super().model_dump()
        try:
            _type = self._agent_type
        except NotImplementedError:
            _type = None
        if isinstance(_type, AgentType):
            _dict["_type"] = str(_type.value)
        elif _type is not None:
            _dict["_type"] = _type
        return _dict

    def save(self, file_path: Path | str) -> None:
        """Save the agent.

        Args:
            file_path: Path to file to save the agent to.

        Example:
        ```python
        # If working with agent executor
        agent.agent.save(file_path="path/agent.yaml")
        ```
        """
        # Convert file to Path object.
        save_path = Path(file_path) if isinstance(file_path, str) else file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        # Fetch dictionary to save
        agent_dict = self.dict()
        if "_type" not in agent_dict:
            msg = f"Agent {self} does not support saving"
            raise NotImplementedError(msg)

        if save_path.suffix == ".json":
            with save_path.open("w") as f:
                json.dump(agent_dict, f, indent=4)
        elif save_path.suffix.endswith((".yaml", ".yml")):
            with save_path.open("w") as f:
                yaml.dump(agent_dict, f, default_flow_style=False)
        else:
            msg = f"{save_path} must be json or yaml"
            raise ValueError(msg)

    def tool_run_logging_kwargs(self) -> builtins.dict:
        """Return logging kwargs for tool run."""
        return {}


class BaseMultiActionAgent(BaseModel):
    """Base Multi Action Agent class."""

    @property
    def return_values(self) -> list[str]:
        """Return values of the agent."""
        return ["output"]

    def get_allowed_tools(self) -> list[str] | None:
        """Get allowed tools.

        Returns:
            Allowed tools.
        """
        return None

    @abstractmethod
    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> list[AgentAction] | AgentFinish:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with the observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Actions specifying what tool to use.
        """

    @abstractmethod
    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> list[AgentAction] | AgentFinish:
        """Async given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with the observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Actions specifying what tool to use.
        """

    @property
    @abstractmethod
    def input_keys(self) -> list[str]:
        """Return the input keys.

        :meta private:
        """

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: list[tuple[AgentAction, str]],  # noqa: ARG002
        **_: Any,
    ) -> AgentFinish:
        """Return response when agent has been stopped due to max iterations.

        Args:
            early_stopping_method: Method to use for early stopping.
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.

        Returns:
            Agent finish object.

        Raises:
            ValueError: If `early_stopping_method` is not supported.
        """
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish({"output": "Agent stopped due to max iterations."}, "")
        msg = f"Got unsupported early_stopping_method `{early_stopping_method}`"
        raise ValueError(msg)

    @property
    def _agent_type(self) -> str:
        """Return Identifier of an agent type."""
        raise NotImplementedError

    @override
    def dict(self, **kwargs: Any) -> builtins.dict:
        """Return dictionary representation of agent."""
        _dict = super().model_dump()
        with contextlib.suppress(NotImplementedError):
            _dict["_type"] = str(self._agent_type)
        return _dict

    def save(self, file_path: Path | str) -> None:
        """Save the agent.

        Args:
            file_path: Path to file to save the agent to.

        Raises:
            NotImplementedError: If agent does not support saving.
            ValueError: If `file_path` is not json or yaml.

        Example:
        ```python
        # If working with agent executor
        agent.agent.save(file_path="path/agent.yaml")
        ```
        """
        # Convert file to Path object.
        save_path = Path(file_path) if isinstance(file_path, str) else file_path

        # Fetch dictionary to save
        agent_dict = self.dict()
        if "_type" not in agent_dict:
            msg = f"Agent {self} does not support saving."
            raise NotImplementedError(msg)

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == ".json":
            with save_path.open("w") as f:
                json.dump(agent_dict, f, indent=4)
        elif save_path.suffix.endswith((".yaml", ".yml")):
            with save_path.open("w") as f:
                yaml.dump(agent_dict, f, default_flow_style=False)
        else:
            msg = f"{save_path} must be json or yaml"
            raise ValueError(msg)

    def tool_run_logging_kwargs(self) -> builtins.dict:
        """Return logging kwargs for tool run."""
        return {}


class AgentOutputParser(BaseOutputParser[AgentAction | AgentFinish]):
    """Base class for parsing agent output into agent action/finish."""

    @abstractmethod
    def parse(self, text: str) -> AgentAction | AgentFinish:
        """Parse text into agent action/finish."""


class MultiActionAgentOutputParser(
    BaseOutputParser[list[AgentAction] | AgentFinish],
):
    """Base class for parsing agent output into agent actions/finish.

    This is used for agents that can return multiple actions.
    """

    @abstractmethod
    def parse(self, text: str) -> list[AgentAction] | AgentFinish:
        """Parse text into agent actions/finish.

        Args:
            text: Text to parse.

        Returns:
            List of agent actions or agent finish.
        """


class RunnableAgent(BaseSingleActionAgent):
    """Agent powered by Runnables."""

    runnable: Runnable[dict, AgentAction | AgentFinish]
    """Runnable to call to get agent action."""
    input_keys_arg: list[str] = []
    return_keys_arg: list[str] = []
    stream_runnable: bool = True
    """Whether to stream from the runnable or not.

    If `True` then underlying LLM is invoked in a streaming fashion to make it possible
        to get access to the individual LLM tokens when using stream_log with the Agent
        Executor. If `False` then LLM is invoked in a non-streaming fashion and
        individual LLM tokens will not be available in stream_log.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    def return_values(self) -> list[str]:
        """Return values of the agent."""
        return self.return_keys_arg

    @property
    def input_keys(self) -> list[str]:
        """Return the input keys."""
        return self.input_keys_arg

    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """Based on past history and current inputs, decide what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with the observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        inputs = {**kwargs, "intermediate_steps": intermediate_steps}
        final_output: Any = None
        if self.stream_runnable:
            # Use streaming to make sure that the underlying LLM is invoked in a
            # streaming
            # fashion to make it possible to get access to the individual LLM tokens
            # when using stream_log with the Agent Executor.
            # Because the response from the plan is not a generator, we need to
            # accumulate the output into final output and return that.
            for chunk in self.runnable.stream(inputs, config={"callbacks": callbacks}):
                if final_output is None:
                    final_output = chunk
                else:
                    final_output += chunk
        else:
            final_output = self.runnable.invoke(inputs, config={"callbacks": callbacks})

        return final_output

    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """Async based on past history and current inputs, decide what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        inputs = {**kwargs, "intermediate_steps": intermediate_steps}
        final_output: Any = None
        if self.stream_runnable:
            # Use streaming to make sure that the underlying LLM is invoked in a
            # streaming
            # fashion to make it possible to get access to the individual LLM tokens
            # when using stream_log with the Agent Executor.
            # Because the response from the plan is not a generator, we need to
            # accumulate the output into final output and return that.
            async for chunk in self.runnable.astream(
                inputs,
                config={"callbacks": callbacks},
            ):
                if final_output is None:
                    final_output = chunk
                else:
                    final_output += chunk
        else:
            final_output = await self.runnable.ainvoke(
                inputs,
                config={"callbacks": callbacks},
            )
        return final_output


class RunnableMultiActionAgent(BaseMultiActionAgent):
    """Agent powered by Runnables."""

    runnable: Runnable[dict, list[AgentAction] | AgentFinish]
    """Runnable to call to get agent actions."""
    input_keys_arg: list[str] = []
    return_keys_arg: list[str] = []
    stream_runnable: bool = True
    """Whether to stream from the runnable or not.

    If `True` then underlying LLM is invoked in a streaming fashion to make it possible
        to get access to the individual LLM tokens when using stream_log with the Agent
        Executor. If `False` then LLM is invoked in a non-streaming fashion and
        individual LLM tokens will not be available in stream_log.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    def return_values(self) -> list[str]:
        """Return values of the agent."""
        return self.return_keys_arg

    @property
    def input_keys(self) -> list[str]:
        """Return the input keys.

        Returns:
            List of input keys.
        """
        return self.input_keys_arg

    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> list[AgentAction] | AgentFinish:
        """Based on past history and current inputs, decide what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with the observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        inputs = {**kwargs, "intermediate_steps": intermediate_steps}
        final_output: Any = None
        if self.stream_runnable:
            # Use streaming to make sure that the underlying LLM is invoked in a
            # streaming
            # fashion to make it possible to get access to the individual LLM tokens
            # when using stream_log with the Agent Executor.
            # Because the response from the plan is not a generator, we need to
            # accumulate the output into final output and return that.
            for chunk in self.runnable.stream(inputs, config={"callbacks": callbacks}):
                if final_output is None:
                    final_output = chunk
                else:
                    final_output += chunk
        else:
            final_output = self.runnable.invoke(inputs, config={"callbacks": callbacks})

        return final_output

    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> list[AgentAction] | AgentFinish:
        """Async based on past history and current inputs, decide what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        inputs = {**kwargs, "intermediate_steps": intermediate_steps}
        final_output: Any = None
        if self.stream_runnable:
            # Use streaming to make sure that the underlying LLM is invoked in a
            # streaming
            # fashion to make it possible to get access to the individual LLM tokens
            # when using stream_log with the Agent Executor.
            # Because the response from the plan is not a generator, we need to
            # accumulate the output into final output and return that.
            async for chunk in self.runnable.astream(
                inputs,
                config={"callbacks": callbacks},
            ):
                if final_output is None:
                    final_output = chunk
                else:
                    final_output += chunk
        else:
            final_output = await self.runnable.ainvoke(
                inputs,
                config={"callbacks": callbacks},
            )

        return final_output


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="1.0",
)
class LLMSingleActionAgent(BaseSingleActionAgent):
    """Base class for single action agents."""

    llm_chain: LLMChain
    """LLMChain to use for agent."""
    output_parser: AgentOutputParser
    """Output parser to use for agent."""
    stop: list[str]
    """List of strings to stop on."""

    @property
    def input_keys(self) -> list[str]:
        """Return the input keys.

        Returns:
            List of input keys.
        """
        return list(set(self.llm_chain.input_keys) - {"intermediate_steps"})

    @override
    def dict(self, **kwargs: Any) -> builtins.dict:
        """Return dictionary representation of agent."""
        _dict = super().dict()
        del _dict["output_parser"]
        return _dict

    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with the observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        output = self.llm_chain.run(
            intermediate_steps=intermediate_steps,
            stop=self.stop,
            callbacks=callbacks,
            **kwargs,
        )
        return self.output_parser.parse(output)

    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """Async given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        output = await self.llm_chain.arun(
            intermediate_steps=intermediate_steps,
            stop=self.stop,
            callbacks=callbacks,
            **kwargs,
        )
        return self.output_parser.parse(output)

    def tool_run_logging_kwargs(self) -> builtins.dict:
        """Return logging kwargs for tool run."""
        return {
            "llm_prefix": "",
            "observation_prefix": "" if len(self.stop) == 0 else self.stop[0],
        }


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="1.0",
)
class Agent(BaseSingleActionAgent):
    """Agent that calls the language model and deciding the action.

    This is driven by a LLMChain. The prompt in the LLMChain MUST include
    a variable called "agent_scratchpad" where the agent can put its
    intermediary work.
    """

    llm_chain: LLMChain
    """LLMChain to use for agent."""
    output_parser: AgentOutputParser
    """Output parser to use for agent."""
    allowed_tools: list[str] | None = None
    """Allowed tools for the agent. If `None`, all tools are allowed."""

    @override
    def dict(self, **kwargs: Any) -> builtins.dict:
        """Return dictionary representation of agent."""
        _dict = super().dict()
        del _dict["output_parser"]
        return _dict

    def get_allowed_tools(self) -> list[str] | None:
        """Get allowed tools."""
        return self.allowed_tools

    @property
    def return_values(self) -> list[str]:
        """Return values of the agent."""
        return ["output"]

    @property
    def _stop(self) -> list[str]:
        return [
            f"\n{self.observation_prefix.rstrip()}",
            f"\n\t{self.observation_prefix.rstrip()}",
        ]

    def _construct_scratchpad(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
    ) -> str | list[BaseMessage]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
        return thoughts

    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        full_output = self.llm_chain.predict(callbacks=callbacks, **full_inputs)
        return self.output_parser.parse(full_output)

    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentAction | AgentFinish:
        """Async given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        full_output = await self.llm_chain.apredict(callbacks=callbacks, **full_inputs)
        return await self.output_parser.aparse(full_output)

    def get_full_inputs(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> builtins.dict[str, Any]:
        """Create the full inputs for the LLMChain from intermediate steps.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            **kwargs: User inputs.

        Returns:
            Full inputs for the LLMChain.
        """
        thoughts = self._construct_scratchpad(intermediate_steps)
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        return {**kwargs, **new_inputs}

    @property
    def input_keys(self) -> list[str]:
        """Return the input keys.

        :meta private:
        """
        return list(set(self.llm_chain.input_keys) - {"agent_scratchpad"})

    @model_validator(mode="after")
    def validate_prompt(self) -> Self:
        """Validate that prompt matches format.

        Args:
            values: Values to validate.

        Returns:
            Validated values.

        Raises:
            ValueError: If `agent_scratchpad` is not in prompt.input_variables
             and prompt is not a FewShotPromptTemplate or a PromptTemplate.
        """
        prompt = self.llm_chain.prompt
        if "agent_scratchpad" not in prompt.input_variables:
            logger.warning(
                "`agent_scratchpad` should be a variable in prompt.input_variables."
                " Did not find it, so adding it at the end.",
            )
            prompt.input_variables.append("agent_scratchpad")
            if isinstance(prompt, PromptTemplate):
                prompt.template += "\n{agent_scratchpad}"
            elif isinstance(prompt, FewShotPromptTemplate):
                prompt.suffix += "\n{agent_scratchpad}"
            else:
                msg = f"Got unexpected prompt type {type(prompt)}"
                raise ValueError(msg)
        return self

    @property
    @abstractmethod
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""

    @property
    @abstractmethod
    def llm_prefix(self) -> str:
        """Prefix to append the LLM call with."""

    @classmethod
    @abstractmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        """Create a prompt for this class.

        Args:
            tools: Tools to use.

        Returns:
            Prompt template.
        """

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        """Validate that appropriate tools are passed in.

        Args:
            tools: Tools to use.
        """

    @classmethod
    @abstractmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        """Get default output parser for this class."""

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: BaseCallbackManager | None = None,
        output_parser: AgentOutputParser | None = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools.

        Args:
            llm: Language model to use.
            tools: Tools to use.
            callback_manager: Callback manager to use.
            output_parser: Output parser to use.
            kwargs: Additional arguments.

        Returns:
            Agent object.
        """
        cls._validate_tools(tools)
        llm_chain = LLMChain(
            llm=llm,
            prompt=cls.create_prompt(tools),
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        _output_parser = output_parser or cls._get_default_output_parser()
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: list[tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> AgentFinish:
        """Return response when agent has been stopped due to max iterations.

        Args:
            early_stopping_method: Method to use for early stopping.
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            **kwargs: User inputs.

        Returns:
            Agent finish object.

        Raises:
            ValueError: If `early_stopping_method` is not in ['force', 'generate'].
        """
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish(
                {"output": "Agent stopped due to iteration limit or time limit."},
                "",
            )
        if early_stopping_method == "generate":
            # Generate does one final forward pass
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += (
                    f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
                )
            # Adding to the previous steps, we now tell the LLM to make a final pred
            thoughts += (
                "\n\nI now need to return a final answer based on the previous steps:"
            )
            new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
            full_inputs = {**kwargs, **new_inputs}
            full_output = self.llm_chain.predict(**full_inputs)
            # We try to extract a final answer
            parsed_output = self.output_parser.parse(full_output)
            if isinstance(parsed_output, AgentFinish):
                # If we can extract, we send the correct stuff
                return parsed_output
            # If we can extract, but the tool is not the final tool,
            # we just return the full output
            return AgentFinish({"output": full_output}, full_output)
        msg = (
            "early_stopping_method should be one of `force` or `generate`, "
            f"got {early_stopping_method}"
        )
        raise ValueError(msg)

    def tool_run_logging_kwargs(self) -> builtins.dict:
        """Return logging kwargs for tool run."""
        return {
            "llm_prefix": self.llm_prefix,
            "observation_prefix": self.observation_prefix,
        }


class ExceptionTool(BaseTool):
    """Tool that just returns the query."""

    name: str = "_Exception"
    """Name of the tool."""
    description: str = "Exception tool"
    """Description of the tool."""

    @override
    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return query

    @override
    async def _arun(
        self,
        query: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return query


NextStepOutput = list[AgentFinish | AgentAction | AgentStep]
RunnableAgentType = RunnableAgent | RunnableMultiActionAgent


class AgentExecutor(Chain):
    """Agent that is using tools."""

    agent: BaseSingleActionAgent | BaseMultiActionAgent | Runnable
    """The agent to run for creating a plan and determining actions
    to take at each step of the execution loop."""
    tools: Sequence[BaseTool]
    """The valid tools the agent can call."""
    return_intermediate_steps: bool = False
    """Whether to return the agent's trajectory of intermediate steps
    at the end in addition to the final output."""
    max_iterations: int | None = 15
    """The maximum number of steps to take before ending the execution
    loop.

    Setting to 'None' could lead to an infinite loop."""
    max_execution_time: float | None = None
    """The maximum amount of wall clock time to spend in the execution
    loop.
    """
    early_stopping_method: str = "force"
    """The method to use for early stopping if the agent never
    returns `AgentFinish`. Either 'force' or 'generate'.

    `"force"` returns a string saying that it stopped because it met a
        time or iteration limit.

    `"generate"` calls the agent's LLM Chain one final time to generate
        a final answer based on the previous steps.
    """
    handle_parsing_errors: bool | str | Callable[[OutputParserException], str] = False
    """How to handle errors raised by the agent's output parser.
    Defaults to `False`, which raises the error.
    If `true`, the error will be sent back to the LLM as an observation.
    If a string, the string itself will be sent to the LLM as an observation.
    If a callable function, the function will be called with the exception as an
    argument, and the result of that function will be passed to the agent as an
    observation.
    """
    trim_intermediate_steps: (
        int | Callable[[list[tuple[AgentAction, str]]], list[tuple[AgentAction, str]]]
    ) = -1
    """How to trim the intermediate steps before returning them.
    Defaults to -1, which means no trimming.
    """

    @classmethod
    def from_agent_and_tools(
        cls,
        agent: BaseSingleActionAgent | BaseMultiActionAgent | Runnable,
        tools: Sequence[BaseTool],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentExecutor:
        """Create from agent and tools.

        Args:
            agent: Agent to use.
            tools: Tools to use.
            callbacks: Callbacks to use.
            kwargs: Additional arguments.

        Returns:
            Agent executor object.
        """
        return cls(
            agent=agent,
            tools=tools,
            callbacks=callbacks,
            **kwargs,
        )

    @model_validator(mode="after")
    def validate_tools(self) -> Self:
        """Validate that tools are compatible with agent.

        Args:
            values: Values to validate.

        Returns:
            Validated values.

        Raises:
            ValueError: If allowed tools are different than provided tools.
        """
        agent = self.agent
        tools = self.tools
        allowed_tools = agent.get_allowed_tools()  # type: ignore[union-attr]
        if allowed_tools is not None and set(allowed_tools) != {
            tool.name for tool in tools
        }:
            msg = (
                f"Allowed tools ({allowed_tools}) different than "
                f"provided tools ({[tool.name for tool in tools]})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="before")
    @classmethod
    def validate_runnable_agent(cls, values: dict) -> Any:
        """Convert runnable to agent if passed in.

        Args:
            values: Values to validate.

        Returns:
            Validated values.
        """
        agent = values.get("agent")
        if agent and isinstance(agent, Runnable):
            try:
                output_type = agent.OutputType
            except TypeError:
                multi_action = False
            except Exception:
                logger.exception("Unexpected error getting OutputType from agent")
                multi_action = False
            else:
                multi_action = output_type == list[AgentAction] | AgentFinish

            stream_runnable = values.pop("stream_runnable", True)
            if multi_action:
                values["agent"] = RunnableMultiActionAgent(
                    runnable=agent,
                    stream_runnable=stream_runnable,
                )
            else:
                values["agent"] = RunnableAgent(
                    runnable=agent,
                    stream_runnable=stream_runnable,
                )
        return values

    @property
    def _action_agent(self) -> BaseSingleActionAgent | BaseMultiActionAgent:
        """Type cast self.agent.

        If the `agent` attribute is a Runnable, it will be converted one of
        RunnableAgentType in the validate_runnable_agent root_validator.

        To support instantiating with a Runnable, here we explicitly cast the type
        to reflect the changes made in the root_validator.
        """
        if isinstance(self.agent, Runnable):
            return cast("RunnableAgentType", self.agent)
        return self.agent

    @override
    def save(self, file_path: Path | str) -> None:
        """Raise error - saving not supported for Agent Executors.

        Args:
            file_path: Path to save to.

        Raises:
            ValueError: Saving not supported for agent executors.
        """
        msg = (
            "Saving not supported for agent executors. "
            "If you are trying to save the agent, please use the "
            "`.save_agent(...)`"
        )
        raise ValueError(msg)

    def save_agent(self, file_path: Path | str) -> None:
        """Save the underlying agent.

        Args:
            file_path: Path to save to.
        """
        return self._action_agent.save(file_path)

    def iter(
        self,
        inputs: Any,
        callbacks: Callbacks = None,
        *,
        include_run_info: bool = False,
        async_: bool = False,  # noqa: ARG002 arg kept for backwards compat, but ignored
    ) -> AgentExecutorIterator:
        """Enables iteration over steps taken to reach final output.

        Args:
            inputs: Inputs to the agent.
            callbacks: Callbacks to run.
            include_run_info: Whether to include run info.
            async_: Whether to run async. (Ignored)

        Returns:
            Agent executor iterator object.
        """
        return AgentExecutorIterator(
            self,
            inputs,
            callbacks,
            tags=self.tags,
            include_run_info=include_run_info,
        )

    @property
    def input_keys(self) -> list[str]:
        """Return the input keys.

        :meta private:
        """
        return self._action_agent.input_keys

    @property
    def output_keys(self) -> list[str]:
        """Return the singular output key.

        :meta private:
        """
        if self.return_intermediate_steps:
            return [*self._action_agent.return_values, "intermediate_steps"]
        return self._action_agent.return_values

    def lookup_tool(self, name: str) -> BaseTool:
        """Lookup tool by name.

        Args:
            name: Name of tool.

        Returns:
            Tool object.
        """
        return {tool.name: tool for tool in self.tools}[name]

    def _should_continue(self, iterations: int, time_elapsed: float) -> bool:
        if self.max_iterations is not None and iterations >= self.max_iterations:
            return False
        return self.max_execution_time is None or time_elapsed < self.max_execution_time

    def _return(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        if run_manager:
            run_manager.on_agent_finish(output, color="green", verbose=self.verbose)
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    async def _areturn(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        if run_manager:
            await run_manager.on_agent_finish(
                output,
                color="green",
                verbose=self.verbose,
            )
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    def _consume_next_step(
        self,
        values: NextStepOutput,
    ) -> AgentFinish | list[tuple[AgentAction, str]]:
        if isinstance(values[-1], AgentFinish):
            if len(values) != 1:
                msg = "Expected a single AgentFinish output, but got multiple values."
                raise ValueError(msg)
            return values[-1]
        return [(a.action, a.observation) for a in values if isinstance(a, AgentStep)]

    def _take_next_step(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        inputs: dict[str, str],
        intermediate_steps: list[tuple[AgentAction, str]],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> AgentFinish | list[tuple[AgentAction, str]]:
        return self._consume_next_step(
            list(
                self._iter_next_step(
                    name_to_tool_map,
                    color_mapping,
                    inputs,
                    intermediate_steps,
                    run_manager,
                ),
            ),
        )

    def _iter_next_step(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        inputs: dict[str, str],
        intermediate_steps: list[tuple[AgentAction, str]],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> Iterator[AgentFinish | AgentAction | AgentStep]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            # Call the LLM to see what to do.
            output = self._action_agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                msg = (
                    "An output parsing error occurred. "
                    "In order to pass this error back to the agent and have it try "
                    "again, pass `handle_parsing_errors=True` to the AgentExecutor. "
                    f"This is the error: {e!s}"
                )
                raise ValueError(msg) from e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                msg = "Got unexpected type of `handle_parsing_errors`"  # type: ignore[unreachable]
                raise ValueError(msg) from e  # noqa: TRY004
            output = AgentAction("_Exception", observation, text)
            if run_manager:
                run_manager.on_agent_action(output, color="green")
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            yield AgentStep(action=output, observation=observation)
            return

        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            yield output
            return

        actions: list[AgentAction]
        actions = [output] if isinstance(output, AgentAction) else output
        for agent_action in actions:
            yield agent_action
        for agent_action in actions:
            yield self._perform_agent_action(
                name_to_tool_map,
                color_mapping,
                agent_action,
                run_manager,
            )

    def _perform_agent_action(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        agent_action: AgentAction,
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> AgentStep:
        if run_manager:
            run_manager.on_agent_action(agent_action, color="green")
        # Otherwise we lookup the tool
        if agent_action.tool in name_to_tool_map:
            tool = name_to_tool_map[agent_action.tool]
            return_direct = tool.return_direct
            color = color_mapping[agent_action.tool]
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""
            # We then call the tool on the tool input to get an observation
            observation = tool.run(
                agent_action.tool_input,
                verbose=self.verbose,
                color=color,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        else:
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            observation = InvalidTool().run(
                {
                    "requested_tool_name": agent_action.tool,
                    "available_tool_names": list(name_to_tool_map.keys()),
                },
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        return AgentStep(action=agent_action, observation=observation)

    async def _atake_next_step(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        inputs: dict[str, str],
        intermediate_steps: list[tuple[AgentAction, str]],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> AgentFinish | list[tuple[AgentAction, str]]:
        return self._consume_next_step(
            [
                a
                async for a in self._aiter_next_step(
                    name_to_tool_map,
                    color_mapping,
                    inputs,
                    intermediate_steps,
                    run_manager,
                )
            ],
        )

    async def _aiter_next_step(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        inputs: dict[str, str],
        intermediate_steps: list[tuple[AgentAction, str]],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> AsyncIterator[AgentFinish | AgentAction | AgentStep]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            # Call the LLM to see what to do.
            output = await self._action_agent.aplan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                msg = (
                    "An output parsing error occurred. "
                    "In order to pass this error back to the agent and have it try "
                    "again, pass `handle_parsing_errors=True` to the AgentExecutor. "
                    f"This is the error: {e!s}"
                )
                raise ValueError(msg) from e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                msg = "Got unexpected type of `handle_parsing_errors`"  # type: ignore[unreachable]
                raise ValueError(msg) from e  # noqa: TRY004
            output = AgentAction("_Exception", observation, text)
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            observation = await ExceptionTool().arun(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            yield AgentStep(action=output, observation=observation)
            return

        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            yield output
            return

        actions: list[AgentAction]
        actions = [output] if isinstance(output, AgentAction) else output
        for agent_action in actions:
            yield agent_action

        # Use asyncio.gather to run multiple tool.arun() calls concurrently
        result = await asyncio.gather(
            *[
                self._aperform_agent_action(
                    name_to_tool_map,
                    color_mapping,
                    agent_action,
                    run_manager,
                )
                for agent_action in actions
            ],
        )

        # TODO: This could yield each result as it becomes available
        for chunk in result:
            yield chunk

    async def _aperform_agent_action(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        agent_action: AgentAction,
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> AgentStep:
        if run_manager:
            await run_manager.on_agent_action(
                agent_action,
                verbose=self.verbose,
                color="green",
            )
        # Otherwise we lookup the tool
        if agent_action.tool in name_to_tool_map:
            tool = name_to_tool_map[agent_action.tool]
            return_direct = tool.return_direct
            color = color_mapping[agent_action.tool]
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""
            # We then call the tool on the tool input to get an observation
            observation = await tool.arun(
                agent_action.tool_input,
                verbose=self.verbose,
                color=color,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        else:
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            observation = await InvalidTool().arun(
                {
                    "requested_tool_name": agent_action.tool,
                    "available_tool_names": list(name_to_tool_map.keys()),
                },
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        return AgentStep(action=agent_action, observation=observation)

    def _call(
        self,
        inputs: dict[str, str],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools],
            excluded_colors=["green", "red"],
        )
        intermediate_steps: list[tuple[AgentAction, str]] = []
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=run_manager,
            )
            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    next_step_output,
                    intermediate_steps,
                    run_manager=run_manager,
                )

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(
                        tool_return,
                        intermediate_steps,
                        run_manager=run_manager,
                    )
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self._action_agent.return_stopped_response(
            self.early_stopping_method,
            intermediate_steps,
            **inputs,
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)

    async def _acall(
        self,
        inputs: dict[str, str],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        """Async run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools],
            excluded_colors=["green"],
        )
        intermediate_steps: list[tuple[AgentAction, str]] = []
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        try:
            async with asyncio_timeout(self.max_execution_time):
                while self._should_continue(iterations, time_elapsed):
                    next_step_output = await self._atake_next_step(
                        name_to_tool_map,
                        color_mapping,
                        inputs,
                        intermediate_steps,
                        run_manager=run_manager,
                    )
                    if isinstance(next_step_output, AgentFinish):
                        return await self._areturn(
                            next_step_output,
                            intermediate_steps,
                            run_manager=run_manager,
                        )

                    intermediate_steps.extend(next_step_output)
                    if len(next_step_output) == 1:
                        next_step_action = next_step_output[0]
                        # See if tool should return directly
                        tool_return = self._get_tool_return(next_step_action)
                        if tool_return is not None:
                            return await self._areturn(
                                tool_return,
                                intermediate_steps,
                                run_manager=run_manager,
                            )

                    iterations += 1
                    time_elapsed = time.time() - start_time
                output = self._action_agent.return_stopped_response(
                    self.early_stopping_method,
                    intermediate_steps,
                    **inputs,
                )
                return await self._areturn(
                    output,
                    intermediate_steps,
                    run_manager=run_manager,
                )
        except (TimeoutError, asyncio.TimeoutError):
            # stop early when interrupted by the async timeout
            output = self._action_agent.return_stopped_response(
                self.early_stopping_method,
                intermediate_steps,
                **inputs,
            )
            return await self._areturn(
                output,
                intermediate_steps,
                run_manager=run_manager,
            )

    def _get_tool_return(
        self,
        next_step_output: tuple[AgentAction, str],
    ) -> AgentFinish | None:
        """Check if the tool is a returning tool."""
        agent_action, observation = next_step_output
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        return_value_key = "output"
        if len(self._action_agent.return_values) > 0:
            return_value_key = self._action_agent.return_values[0]
        # Invalid tools won't be in the map, so we return False.
        if (
            agent_action.tool in name_to_tool_map
            and name_to_tool_map[agent_action.tool].return_direct
        ):
            return AgentFinish(
                {return_value_key: observation},
                "",
            )
        return None

    def _prepare_intermediate_steps(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
    ) -> list[tuple[AgentAction, str]]:
        if (
            isinstance(self.trim_intermediate_steps, int)
            and self.trim_intermediate_steps > 0
        ):
            return intermediate_steps[-self.trim_intermediate_steps :]
        if callable(self.trim_intermediate_steps):
            return self.trim_intermediate_steps(intermediate_steps)
        return intermediate_steps

    @override
    def stream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[AddableDict]:
        """Enables streaming over steps taken to reach final output.

        Args:
            input: Input to the agent.
            config: Config to use.
            kwargs: Additional arguments.

        Yields:
            Addable dictionary.
        """
        config = ensure_config(config)
        iterator = AgentExecutorIterator(
            self,
            input,
            config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            run_id=config.get("run_id"),
            yield_actions=True,
            **kwargs,
        )
        yield from iterator

    @override
    async def astream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AddableDict]:
        """Async enables streaming over steps taken to reach final output.

        Args:
            input: Input to the agent.
            config: Config to use.
            kwargs: Additional arguments.

        Yields:
            Addable dictionary.
        """
        config = ensure_config(config)
        iterator = AgentExecutorIterator(
            self,
            input,
            config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            run_id=config.get("run_id"),
            yield_actions=True,
            **kwargs,
        )
        async for step in iterator:
            yield step
