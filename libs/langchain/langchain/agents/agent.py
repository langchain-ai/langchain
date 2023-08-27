"""Chain that takes in an input and produces an action and input for the action."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import yaml

from langchain.agents.agent_iterator import AgentExecutorIterator
from langchain.agents.agent_types import AgentType
from langchain.agents.tools import InvalidTool
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForToolRun,
    CallbackManagerForChainRun,
    CallbackManagerForToolRun,
    Callbacks,
)
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseOutputParser,
    BasePromptTemplate,
    OutputParserException,
)
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import BaseMessage
from langchain.tools.base import BaseTool
from langchain.utilities.asyncio import asyncio_timeout
from langchain.utils.input import get_color_mapping

logger = logging.getLogger(__name__)


class BaseSingleActionAgent(BaseModel):
    """A base class for agents that perform a single action."""

    @property
    def return_values(self) -> List[str]:
        """Returns the output of the agent."""
        return ["output"]

    def get_allowed_tools(self) -> Optional[List[str]]:
        """Returns the allowed tools for the agent. Can be overridden in subclass."""
        return None

    @abstractmethod
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Decides what action the agent should take given the current state.

        Args:
            intermediate_steps: A list of tuples containing the actions taken and
                                observations made by the agent so far.
            callbacks: Callbacks to run.
            **kwargs: Additional inputs.

        Returns:
            The next action for the agent to take.
        """

    @abstractmethod
    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Asynchronously decides what action the agent should take given the current state.

        Args:
            intermediate_steps: A list of tuples containing the actions taken and
                                observations made by the agent so far.
            callbacks: Callbacks to run.
            **kwargs: Additional inputs.

        Returns:
            The next action for the agent to take.
        """

    @property
    @abstractmethod
    def input_keys(self) -> List[str]:
        """Returns the list of input keys. To be defined in subclass."""

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> AgentFinish:
        """Returns a response when the agent has been stopped due to reaching the
        maximum number of iterations.

        Args:
            early_stopping_method: Method used for early stopping.
            intermediate_steps: A list of tuples containing the actions taken and
                                observations made by the agent so far.
            **kwargs: Additional inputs.

        Returns:
            The final state of the agent.
        """

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ) -> BaseSingleActionAgent:
        """Creates an instance of the agent using the provided language model and tools.

        Args:
            llm: The language model to use.
            tools: The tools available to the agent.
            callback_manager: The manager for handling callbacks.
            **kwargs: Additional inputs.

        Returns:
            An instance of the agent.
        """
        raise NotImplementedError

    @property
    def _agent_type(self) -> str:
        """Returns the type of agent."""
        raise NotImplementedError

    def dict(self, **kwargs: Any) -> Dict:
        """Returns a dictionary representation of the agent."""
        _dict = super().dict()
        _type = self._agent_type
        if isinstance(_type, AgentType):
            _dict["_type"] = str(_type.value)
        else:
            _dict["_type"] = _type
        return _dict

    def save(self, file_path: Union[Path, str]) -> None:
        """Saves the state of the agent to a file.

        Args:
            file_path: The path to the file where the agent's state should be saved.
        """
        # Convert file to Path object.
        if isinstance(file_path, str):
            save_path = Path(file_path)
        else:
            save_path = file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        # Fetch dictionary to save
        agent_dict = self.dict()

        if save_path.suffix == ".json":
            with open(file_path, "w") as f:
                json.dump(agent_dict, f, indent=4)
        elif save_path.suffix == ".yaml":
            with open(file_path, "w") as f:
                yaml.dump(agent_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"{save_path} must be json or yaml")

    def tool_run_logging_kwargs(self) -> Dict:
        """Returns a dictionary of logging parameters for the tool run."""
        return {}


class BaseMultiActionAgent(BaseModel):
    """A base class for agents that can perform multiple actions."""

    @property
    def return_values(self) -> List[str]:
        """Returns the output of the agent."""
        return ["output"]

    def get_allowed_tools(self) -> Optional[List[str]]:
        """Returns the allowed tools for the agent. Can be overridden in subclass."""
        return None

    @abstractmethod
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[List[AgentAction], AgentFinish]:
        """Decides what actions the agent should take given the current state.

        Args:
            intermediate_steps: A list of tuples containing the actions taken and
                                observations made by the agent so far.
            callbacks: Callbacks to run.
            **kwargs: Additional inputs.

        Returns:
            The next actions for the agent to take.
        """

    @abstractmethod
    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[List[AgentAction], AgentFinish]:
        """Asynchronously decides what actions the agent should take given the current state.

        Args:
            intermediate_steps: A list of tuples containing the actions taken and
                                observations made by the agent so far.
            callbacks: Callbacks to run.
            **kwargs: Additional inputs.

        Returns:
            The next actions for the agent to take.
        """

    @property
    @abstractmethod
    def input_keys(self) -> List[str]:
        """Returns the list of input keys. To be defined in subclass."""

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> AgentFinish:
        """Returns a response when the agent has been stopped due to reaching the
        maximum number of iterations.

        Args:
            early_stopping_method: Method used for early stopping.
            intermediate_steps: A list of tuples containing the actions taken and
                                observations made by the agent so far.
            **kwargs: Additional inputs.

        Returns:
            The final state of the agent.
        """

    @property
    def _agent_type(self) -> str:
        """Returns the type of agent."""
        raise NotImplementedError

    def dict(self, **kwargs: Any) -> Dict:
        """Returns a dictionary representation of the agent."""
        _dict = super().dict()
        _dict["_type"] = str(self._agent_type)
        return _dict

    def save(self, file_path: Union[Path, str]) -> None:
        """Saves the state of the agent to a file.

        Args:
            file_path: The path to the file where the agent's state should be saved.
        """
        # Convert file to Path object.
        if isinstance(file_path, str):
            save_path = Path(file_path)
        else:
            save_path = file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        # Fetch dictionary to save
        agent_dict = self.dict()

        if save_path.suffix == ".json":
            with open(file_path, "w") as f:
                json.dump(agent_dict, f, indent=4)
        elif save_path.suffix == ".yaml":
            with open(file_path, "w") as f:
                yaml.dump(agent_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"{save_path} must be json or yaml")

    def tool_run_logging_kwargs(self) -> Dict:
        """Returns a dictionary of logging parameters for the tool run."""
        return {}


class AgentOutputParser(BaseOutputParser):
    """A base class for parsers that transform agent output into an agent action or finish."""

    @abstractmethod
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Transforms the given text into an agent action or finish.

        Args:
            text: The text to be parsed.

        Returns:
            The agent action or finish represented by the text.
        """


class LLMSingleActionAgent(BaseSingleActionAgent):
    """A base class for agents that perform a single action with a language model."""

    llm_chain: LLMChain
    """The language model chain used by the agent."""
    output_parser: AgentOutputParser
    """The parser used to transform the output of the language model into an agent action or finish."""
    stop: List[str]
    """A list of strings that, if encountered in the output, will stop the agent."""

    @property
    def input_keys(self) -> List[str]:
        """Returns the list of input keys.

        Returns:
            The list of input keys.
        """
        return list(set(self.llm_chain.input_keys) - {"intermediate_steps"})

    def dict(self, **kwargs: Any) -> Dict:
        """Returns a dictionary representation of the agent."""
        _dict = super().dict()
        del _dict["output_parser"]
        return _dict

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Decides what action the agent should take given the current state.

        Args:
            intermediate_steps: A list of tuples containing the actions taken and
                                observations made by the agent so far.
            callbacks: Callbacks to run.
            **kwargs: Additional inputs.

        Returns:
            The next action for the agent to take.
        """

    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Asynchronously decides what action the agent should take given the current state.

        Args:
            intermediate_steps: A list of tuples containing the actions taken and
                                observations made by the agent so far.
            callbacks: Callbacks to run.
            **kwargs: Additional inputs.

        Returns:
            The next action for the agent to take.
        """

    def tool_run_logging_kwargs(self) -> Dict:
        """Returns a dictionary of logging parameters for the tool run."""
        return {
            "llm_prefix": "",
            "observation_prefix": "" if len(self.stop) == 0 else self.stop[0],
        }


class Agent(BaseSingleActionAgent):
    """
    Class to represent an Agent that calls the language model and decides on the action.

    This is driven by an LLMChain. The prompt in the LLMChain MUST include
    a variable called "agent_scratchpad" where the agent can put its
    intermediary work.

    Attributes:
        llm_chain: An instance of LLMChain class.
        output_parser: An instance of AgentOutputParser class.
        allowed_tools: Optional list of allowed tools.

    """

    def dict(self, **kwargs: Any) -> Dict:
        """
        Return the dictionary representation of the agent.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Dictionary representation of the agent.
        """
        _dict = super().dict()
        del _dict["output_parser"]
        return _dict

    def get_allowed_tools(self) -> Optional[List[str]]:
        """
        Get allowed tools for the agent.

        Returns:
            list: List of allowed tools.
        """
        return self.allowed_tools

    @property
    def return_values(self) -> List[str]:
        """
        Return the output values.

        Returns:
            list: List of output values.
        """
        return ["output"]

    def _fix_text(self, text: str) -> str:
        """
        Fix the text.

        Args:
            text: Text to be fixed.

        Raises:
            ValueError: If the method is not implemented for this agent.

        Returns:
            str: Fixed text.
        """
        raise ValueError("fix_text not implemented for this agent.")

    @property
    def _stop(self) -> List[str]:
        """
        Get the stop tokens.

        Returns:
            list: List of stop tokens.
        """
        return [
            f"\n{self.observation_prefix.rstrip()}",
            f"\n\t{self.observation_prefix.rstrip()}",
        ]

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> Union[str, List[BaseMessage]]:
        """
        Construct the scratchpad that lets the agent continue its thought process.

        Args:
            intermediate_steps: List of tuples containing AgentAction and string.

        Returns:
            Union[str, List[BaseMessage]]: Constructed scratchpad.
        """
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
        return thoughts

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """
        Decide the action based on the given input.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Union[AgentAction, AgentFinish]: Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        full_output = self.llm_chain.predict(callbacks=callbacks, **full_inputs)
        return self.output_parser.parse(full_output)

    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """
        Decide the action based on the given input asynchronously.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Union[AgentAction, AgentFinish]: Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        full_output = await self.llm_chain.apredict(callbacks=callbacks, **full_inputs)
        agent_output = await self.output_parser.aparse(full_output)
        return agent_output

    def get_full_inputs(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Create the full inputs for the LLMChain from intermediate steps.

        Args:
            intermediate_steps: List of tuples containing AgentAction and string.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Full inputs for the LLMChain.
        """
        thoughts = self._construct_scratchpad(intermediate_steps)
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        full_inputs = {**kwargs, **new_inputs}
        return full_inputs

    @property
    def input_keys(self) -> List[str]:
        """
        Return the input keys.

        Returns:
            list: List of input keys.
        """
        return list(set(self.llm_chain.input_keys) - {"agent_scratchpad"})

    @root_validator()
    def validate_prompt(cls, values: Dict) -> Dict:
        """
        Validate that prompt matches the format.

        Args:
            values: Dictionary of values.

        Returns:
            dict: Validated dictionary of values.
        """
        prompt = values["llm_chain"].prompt
        if "agent_scratchpad" not in prompt.input_variables:
            logger.warning(
                "`agent_scratchpad` should be a variable in prompt.input_variables."
                " Did not find it, so adding it at the end."
            )
            prompt.input_variables.append("agent_scratchpad")
            if isinstance(prompt, PromptTemplate):
                prompt.template += "\n{agent_scratchpad}"
            elif isinstance(prompt, FewShotPromptTemplate):
                prompt.suffix += "\n{agent_scratchpad}"
            else:
                raise ValueError(f"Got unexpected prompt type {type(prompt)}")
        return values

    @property
    @abstractmethod
    def observation_prefix(self) -> str:
        """
        Get the prefix to append the observation with.

        Returns:
            str: Observation prefix.
        """

    @property
    @abstractmethod
    def llm_prefix(self) -> str:
        """
        Get the prefix to append the LLM call with.

        Returns:
            str: LLM prefix.
        """

    @classmethod
    @abstractmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        """
        Create a prompt for this class.

        Args:
            tools: Sequence of BaseTool instances.

        Returns:
            BasePromptTemplate: Created prompt template.
        """

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        """
        Validate that appropriate tools are passed in.

        Args:
            tools: Sequence of BaseTool instances.
        """
        pass

    @classmethod
    @abstractmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        """
        Get default output parser for this class.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            AgentOutputParser: Default output parser.
        """

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        output_parser: Optional[AgentOutputParser] = None,
        **kwargs: Any,
    ) -> Agent:
        """
        Construct an agent from an LLM and tools.

        Args:
            llm: An instance of BaseLanguageModel class.
            tools: Sequence of BaseTool instances.
            callback_manager: An instance of BaseCallbackManager class.
            output_parser: An instance of AgentOutputParser class.
            **kwargs: Additional keyword arguments.

        Returns:
            Agent: Constructed agent.
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
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> AgentFinish:
        """
        Return response when agent has been stopped due to max iterations.

        Args:
            early_stopping_method: Method used for early stopping.
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            **kwargs: Additional keyword arguments.

        Returns:
            AgentFinish: Response when agent has been stopped.

        Raises:
            ValueError: If the early stopping method is not `force` or `generate`.
        """
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish(
                {"output": "Agent stopped due to iteration limit or time limit."}, ""
            )
        elif early_stopping_method == "generate":
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
            else:
                # If we can extract, but the tool is not the final tool,
                # we just return the full output
                return AgentFinish({"output": full_output}, full_output)
        else:
            raise ValueError(
                "early_stopping_method should be one of `force` or `generate`, "
                f"got {early_stopping_method}"
            )

    def tool_run_logging_kwargs(self) -> Dict:
        """
        Get the logging keywords.

        Returns:
            dict: Dictionary of logging keywords.
        """
        return {
            "llm_prefix": self.llm_prefix,
            "observation_prefix": self.observation_prefix,
        }


class ExceptionTool(BaseTool):
    """
    Class to represent a tool that just returns the query.

    Attributes:
        name: Name of the tool.
        description: Description of the tool.
    """

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Run the tool.

        Args:
            query: Query to be run.
            run_manager: An instance of CallbackManagerForToolRun.

        Returns:
            str: Result of the run.
        """
        return query

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """
        Run the tool asynchronously.

        Args:
            query: Query to be run.
            run_manager: An instance of AsyncCallbackManagerForToolRun.

        Returns:
            str: Result of the run.
        """
        return query


class AgentExecutor(Chain):
    """
    The AgentExecutor class uses a provided agent and a set of tools to create a plan and perform actions.
    It represents the main execution loop for the agent's planning and action system.

    Attributes:
        agent: The agent to run for creating a plan and determining actions
               to take at each step of the execution loop.
        tools: The valid tools the agent can call.
        return_intermediate_steps: Whether to return the agent's trajectory of intermediate steps
                                   at the end in addition to the final output.
        max_iterations: The maximum number of steps to take before ending the execution
                        loop. Setting to 'None' could lead to an infinite loop.
        max_execution_time: The maximum amount of wall clock time to spend in the execution
                            loop.
        early_stopping_method: The method to use for early stopping if the agent never
                               returns `AgentFinish`. Either 'force' or 'generate'.
        handle_parsing_errors: How to handle errors raised by the agent's output parser.
                               Defaults to `False`, which raises the error.
        trim_intermediate_steps: Parameter to determine how to trim intermediate steps if too many
                                 are generated. Defaults to -1, which disables trimming.
    """

    @classmethod
    def from_agent_and_tools(
        cls,
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent],
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ) -> AgentExecutor:
        """
        Class method to create an AgentExecutor instance from a given agent and tools.

        Args:
            agent: The agent to run.
            tools: The valid tools the agent can call.
            callback_manager: An optional manager for callbacks during execution.
            **kwargs: Additional keyword arguments.

        Returns:
            An instance of AgentExecutor.
        """
        return cls(
            agent=agent, tools=tools, callback_manager=callback_manager, **kwargs
        )

    @root_validator()
    def validate_tools(cls, values: Dict) -> Dict:
        """
        Validate that the provided tools are compatible with the agent.

        Args:
            values: A dictionary containing class attribute values.

        Returns:
            The validated dictionary of values.

        Raises:
            ValueError: If the provided tools are not compatible with the agent.
        """
        agent = values["agent"]
        tools = values["tools"]
        allowed_tools = agent.get_allowed_tools()
        if allowed_tools is not None:
            if set(allowed_tools) != set([tool.name for tool in tools]):
                raise ValueError(
                    f"Allowed tools ({allowed_tools}) different than "
                    f"provided tools ({[tool.name for tool in tools]})"
                )
        return values

    @root_validator()
    def validate_return_direct_tool(cls, values: Dict) -> Dict:
        """
        Validate that the tools provided are compatible with the agent.

        Args:
            values: A dictionary containing class attribute values.

        Returns:
            The validated dictionary of values.

        Raises:
            ValueError: If tools that have `return_direct=True` are used with multi-action agents.
        """
        agent = values["agent"]
        tools = values["tools"]
        if isinstance(agent, BaseMultiActionAgent):
            for tool in tools:
                if tool.return_direct:
                    raise ValueError(
                        "Tools that have `return_direct=True` are not allowed "
                        "in multi-action agents"
                    )
        return values

    def save(self, file_path: Union[Path, str]) -> None:
        """
        Raise an error as saving is not supported for Agent Executors.

        Args:
            file_path: The filepath to save to.

        Raises:
            ValueError: Always, as saving is not supported for Agent Executors.
        """
        raise ValueError(
            "Saving not supported for agent executors. "
            "If you are trying to save the agent, please use the "
            "`.save_agent(...)`"
        )

    def save_agent(self, file_path: Union[Path, str]) -> None:
        """
        Save the underlying agent to a file.

        Args:
            file_path: The filepath to save the agent to.
        """
        return self.agent.save(file_path)

    def iter(
        self,
        inputs: Any,
        callbacks: Callbacks = None,
        *,
        include_run_info: bool = False,
        async_: bool = False,
    ) -> AgentExecutorIterator:
        """
        Enable iteration over the steps taken by the agent to reach the final output.

        Args:
            inputs: The inputs for the agent.
            callbacks: Callbacks to call during execution.
            include_run_info: Whether to include run information in the output.
            async_: Whether to run asynchronously.

        Returns:
            An iterator over the steps taken by the agent.
        """
        return AgentExecutorIterator(
            self,
            inputs,
            callbacks,
            tags=self.tags,
            include_run_info=include_run_info,
            async_=async_,
        )

    @property
    def input_keys(self) -> List[str]:
        """
        Return the input keys for the agent.

        Returns:
            A list of input keys.
        """
        return self.agent.input_keys

    @property
    def output_keys(self) -> List[str]:
        """
        Return the output keys for the agent.

        Returns:
            A list of output keys.
        """
        if self.return_intermediate_steps:
            return self.agent.return_values + ["intermediate_steps"]
        else:
            return self.agent.return_values

    def lookup_tool(self, name: str) -> BaseTool:
        """
        Lookup a tool by its name.

        Args:
            name: The name of the tool to lookup.

        Returns:
            The tool with the given name.
        """
        return {tool.name: tool for tool in self.tools}[name]

    def _should_continue(self, iterations: int, time_elapsed: float) -> bool:
        """
        Private method to check whether the execution should continue.

        Args:
            iterations: The current number of iterations.
            time_elapsed: The current time elapsed.

        Returns:
            A boolean indicating whether the execution should continue.
        """
        if self.max_iterations is not None and iterations >= self.max_iterations:
            return False
        if (
            self.max_execution_time is not None
            and time_elapsed >= self.max_execution_time
        ):
            return False

        return True

    def _return(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Private method to prepare the final output.

        Args:
            output: The final output from the agent.
            intermediate_steps: The intermediate steps taken by the agent.
            run_manager: An optional manager for callbacks during execution.

        Returns:
            A dictionary containing the final output and any intermediate steps.
        """
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
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronous version of the _return method.

        Args:
            output: The final output from the agent.
            intermediate_steps: The intermediate steps taken by the agent.
            run_manager: An optional manager for callbacks during execution.

        Returns:
            A dictionary containing the final output and any intermediate steps.
        """
        if run_manager:
            await run_manager.on_agent_finish(
                output, color="green", verbose=self.verbose
            )
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """
        Private method to take a single step in the thought-action-observation loop.

        Args:
            name_to_tool_map: A dictionary mapping tool names to tool instances.
            color_mapping: A dictionary mapping tool names to colors for logging.
            inputs: The inputs for the agent.
            intermediate_steps: The intermediate steps taken so far.
            run_manager: An optional manager for callbacks during execution.

        Returns:
            Either an AgentFinish instance if the agent has finished, or a list
            of tuples containing the action taken and the observation made.
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            # Call the LLM to see what to do.
            output = self.agent.plan(
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
                raise e
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
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            if run_manager:
                run_manager.on_agent_action(output, color="green")
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            return [(output, observation)]
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        result = []
        for agent_action in actions:
            if run_manager:
                run_manager.on_agent_action(agent_action, color="green")
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
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
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
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
            result.append((agent_action, observation))
        return result

    async def _atake_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """
        Asynchronous version of the _take_next_step method.

        Args:
            name_to_tool_map: A dictionary mapping tool names to tool instances.
            color_mapping: A dictionary mapping tool names to colors for logging.
            inputs: The inputs for the agent.
            intermediate_steps: The intermediate steps taken so far.
            run_manager: An optional manager for callbacks during execution.

        Returns:
            Either an AgentFinish instance if the agent has finished, or a list
            of tuples containing the action taken and the observation made.
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            # Call the LLM to see what to do.
            output = await self.agent.aplan(
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
                raise e
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
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = await ExceptionTool().arun(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            return [(output, observation)]
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output

        async def _aperform_agent_action(
            agent_action: AgentAction,
        ) -> Tuple[AgentAction, str]:
            if run_manager:
                await run_manager.on_agent_action(
                    agent_action, verbose=self.verbose, color="green"
                )
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
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
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
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
            return agent_action, observation

        # Use asyncio.gather to run multiple tool.arun() calls concurrently
        result = await asyncio.gather(
            *[_aperform_agent_action(agent_action) for agent_action in actions]
        )

        return list(result)

    async def _aperform_agent_action(self, agent_action):
        """
        Run multiple tool.arun() calls concurrently using asyncio.gather.

        Args:
            agent_action: The agent action to be performed.

        Returns:
            result: List of results after performing the agent actions.
        """
        result = await asyncio.gather(
            *[_aperform_agent_action(agent_action) for agent_action in actions]
        )

        return list(result)

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Process the text and get the response from a specific agent.

        Args:
            inputs: A dictionary of inputs to be processed.
            run_manager: An instance of CallbackManagerForChainRun. Default is None.

        Returns:
            A dictionary of the agent's response.
        """
        ...

    async def _acall(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """
        Asynchronously process the text and get the response from a specific agent.

        Args:
            inputs: A dictionary of inputs to be processed.
            run_manager: An instance of AsyncCallbackManagerForChainRun. Default is None.

        Returns:
            A dictionary of the agent's response.
        """
        ...

    def _get_tool_return(
        self, next_step_output: Tuple[AgentAction, str]
    ) -> Optional[AgentFinish]:
        """
        Check if a specific tool is a returning tool.

        Args:
            next_step_output: A tuple of the next step output.

        Returns:
            An instance of AgentFinish if the tool is a returning tool. None otherwise.
        """
        ...

    def _prepare_intermediate_steps(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> List[Tuple[AgentAction, str]]:
        """
        Prepare intermediate steps based on the trim_intermediate_steps attribute.

        Args:
            intermediate_steps: A list of intermediate steps.

        Returns:
            A list of prepared intermediate steps.
        """
        ...
