"""Chain that takes in an input and produces an action and action input."""
from __future__ import annotations

import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import yaml
from pydantic import BaseModel, root_validator

from langchain.agents.tools import InvalidTool
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.input import get_color_mapping
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import AgentAction, AgentFinish, BaseMessage, BaseOutputParser
from langchain.tools.base import BaseTool

logger = logging.getLogger()


class BaseSingleActionAgent(BaseModel):
    """Base Agent class."""

    @property
    def return_values(self) -> List[str]:
        """Return values of the agent."""
        return ["output"]

    def get_allowed_tools(self) -> Optional[List[str]]:
        return None

    @abstractmethod
    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """

    @abstractmethod
    async def aplan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """

    @property
    @abstractmethod
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> AgentFinish:
        """Return response when agent has been stopped due to max iterations."""
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish({"output": "Agent stopped due to max iterations."}, "")
        else:
            raise ValueError(
                f"Got unsupported early_stopping_method `{early_stopping_method}`"
            )

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        raise NotImplementedError

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of agent."""
        _dict = super().dict()
        _dict["_type"] = self._agent_type
        return _dict

    def save(self, file_path: Union[Path, str]) -> None:
        """Save the agent.

        Args:
            file_path: Path to file to save the agent to.

        Example:
        .. code-block:: python

            # If working with agent executor
            agent.agent.save(file_path="path/agent.yaml")
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
        return {}


class AgentOutputParser(BaseOutputParser):
    @abstractmethod
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Parse text into agent action/finish."""


class LLMSingleActionAgent(BaseSingleActionAgent):
    llm_chain: LLMChain
    output_parser: AgentOutputParser
    stop: List[str]

    @property
    def input_keys(self) -> List[str]:
        return list(set(self.llm_chain.input_keys) - {"intermediate_steps"})

    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        output = self.llm_chain.run(
            intermediate_steps=intermediate_steps, stop=self.stop, **kwargs
        )
        return self.output_parser.parse(output)

    async def aplan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        output = await self.llm_chain.arun(
            intermediate_steps=intermediate_steps, stop=self.stop, **kwargs
        )
        return self.output_parser.parse(output)

    def tool_run_logging_kwargs(self) -> Dict:
        return {
            "llm_prefix": "",
            "observation_prefix": "" if len(self.stop) == 0 else self.stop[0],
        }


class Agent(BaseSingleActionAgent):
    """Class responsible for calling the language model and deciding the action.

    This is driven by an LLMChain. The prompt in the LLMChain MUST include
    a variable called "agent_scratchpad" where the agent can put its
    intermediary work.
    """

    llm_chain: LLMChain
    allowed_tools: Optional[List[str]] = None

    def get_allowed_tools(self) -> Optional[List[str]]:
        return self.allowed_tools

    @property
    def return_values(self) -> List[str]:
        return ["output"]

    @abstractmethod
    def _extract_tool_and_input(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract tool and tool input from llm output."""

    def _fix_text(self, text: str) -> str:
        """Fix the text."""
        raise ValueError("fix_text not implemented for this agent.")

    @property
    def _stop(self) -> List[str]:
        return [
            f"\n{self.observation_prefix.rstrip()}",
            f"\n\t{self.observation_prefix.rstrip()}",
        ]

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> Union[str, List[BaseMessage]]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
        return thoughts

    def _get_next_action(self, full_inputs: Dict[str, str]) -> AgentAction:
        full_output = self.llm_chain.predict(**full_inputs)
        parsed_output = self._extract_tool_and_input(full_output)
        while parsed_output is None:
            full_output = self._fix_text(full_output)
            full_inputs["agent_scratchpad"] += full_output
            output = self.llm_chain.predict(**full_inputs)
            full_output += output
            parsed_output = self._extract_tool_and_input(full_output)
        return AgentAction(
            tool=parsed_output[0], tool_input=parsed_output[1], log=full_output
        )

    async def _aget_next_action(self, full_inputs: Dict[str, str]) -> AgentAction:
        full_output = await self.llm_chain.apredict(**full_inputs)
        parsed_output = self._extract_tool_and_input(full_output)
        while parsed_output is None:
            full_output = self._fix_text(full_output)
            full_inputs["agent_scratchpad"] += full_output
            output = await self.llm_chain.apredict(**full_inputs)
            full_output += output
            parsed_output = self._extract_tool_and_input(full_output)
        return AgentAction(
            tool=parsed_output[0], tool_input=parsed_output[1], log=full_output
        )

    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        action = self._get_next_action(full_inputs)
        if action.tool == self.finish_tool_name:
            return AgentFinish({"output": action.tool_input}, action.log)
        return action

    async def aplan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        action = await self._aget_next_action(full_inputs)
        if action.tool == self.finish_tool_name:
            return AgentFinish({"output": action.tool_input}, action.log)
        return action

    def get_full_inputs(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Create the full inputs for the LLMChain from intermediate steps."""
        thoughts = self._construct_scratchpad(intermediate_steps)
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        full_inputs = {**kwargs, **new_inputs}
        return full_inputs

    @property
    def finish_tool_name(self) -> str:
        """Name of the tool to use to finish the chain."""
        return "Final Answer"

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return list(set(self.llm_chain.input_keys) - {"agent_scratchpad"})

    @root_validator()
    def validate_prompt(cls, values: Dict) -> Dict:
        """Validate that prompt matches format."""
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
        """Prefix to append the observation with."""

    @property
    @abstractmethod
    def llm_prefix(self) -> str:
        """Prefix to append the LLM call with."""

    @classmethod
    @abstractmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        """Create a prompt for this class."""

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        """Validate that appropriate tools are passed in."""
        pass

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLLM,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        llm_chain = LLMChain(
            llm=llm,
            prompt=cls.create_prompt(tools),
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        return cls(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> AgentFinish:
        """Return response when agent has been stopped due to max iterations."""
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish({"output": "Agent stopped due to max iterations."}, "")
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
            parsed_output = self._extract_tool_and_input(full_output)
            if parsed_output is None:
                # If we cannot extract, we just return the full output
                return AgentFinish({"output": full_output}, full_output)
            tool, tool_input = parsed_output
            if tool == self.finish_tool_name:
                # If we can extract, we send the correct stuff
                return AgentFinish({"output": tool_input}, full_output)
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
        return {
            "llm_prefix": self.llm_prefix,
            "observation_prefix": self.observation_prefix,
        }


class AgentExecutor(Chain, BaseModel):
    """Consists of an agent using tools."""

    agent: BaseSingleActionAgent
    tools: Sequence[BaseTool]
    return_intermediate_steps: bool = False
    max_iterations: Optional[int] = 15
    early_stopping_method: str = "force"

    @classmethod
    def from_agent_and_tools(
        cls,
        agent: BaseSingleActionAgent,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ) -> AgentExecutor:
        """Create from agent and tools."""
        return cls(
            agent=agent, tools=tools, callback_manager=callback_manager, **kwargs
        )

    @root_validator()
    def validate_tools(cls, values: Dict) -> Dict:
        """Validate that tools are compatible with agent."""
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

    def save(self, file_path: Union[Path, str]) -> None:
        """Raise error - saving not supported for Agent Executors."""
        raise ValueError(
            "Saving not supported for agent executors. "
            "If you are trying to save the agent, please use the "
            "`.save_agent(...)`"
        )

    def save_agent(self, file_path: Union[Path, str]) -> None:
        """Save the underlying agent."""
        return self.agent.save(file_path)

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return self.agent.input_keys

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        if self.return_intermediate_steps:
            return self.agent.return_values + ["intermediate_steps"]
        else:
            return self.agent.return_values

    def lookup_tool(self, name: str) -> BaseTool:
        """Lookup tool by name."""
        return {tool.name: tool for tool in self.tools}[name]

    def _should_continue(self, iterations: int) -> bool:
        if self.max_iterations is None:
            return True
        else:
            return iterations < self.max_iterations

    def _return(self, output: AgentFinish, intermediate_steps: list) -> Dict[str, Any]:
        self.callback_manager.on_agent_finish(
            output, color="green", verbose=self.verbose
        )
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    async def _areturn(
        self, output: AgentFinish, intermediate_steps: list
    ) -> Dict[str, Any]:
        if self.callback_manager.is_async:
            await self.callback_manager.on_agent_finish(
                output, color="green", verbose=self.verbose
            )
        else:
            self.callback_manager.on_agent_finish(
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
    ) -> Union[AgentFinish, Tuple[AgentAction, str]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        # Call the LLM to see what to do.
        output = self.agent.plan(intermediate_steps, **inputs)
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        self.callback_manager.on_agent_action(
            output, verbose=self.verbose, color="green"
        )
        # Otherwise we lookup the tool
        if output.tool in name_to_tool_map:
            tool = name_to_tool_map[output.tool]
            return_direct = tool.return_direct
            color = color_mapping[output.tool]
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""
            # We then call the tool on the tool input to get an observation
            observation = tool.run(
                output.tool_input, verbose=self.verbose, color=color, **tool_run_kwargs
            )
        else:
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = InvalidTool().run(
                output.tool, verbose=self.verbose, color=None, **tool_run_kwargs
            )
        return output, observation

    async def _atake_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
    ) -> Union[AgentFinish, Tuple[AgentAction, str]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        # Call the LLM to see what to do.
        output = await self.agent.aplan(intermediate_steps, **inputs)
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        if self.callback_manager.is_async:
            await self.callback_manager.on_agent_action(
                output, verbose=self.verbose, color="green"
            )
        else:
            self.callback_manager.on_agent_action(
                output, verbose=self.verbose, color="green"
            )

        # Otherwise we lookup the tool
        if output.tool in name_to_tool_map:
            tool = name_to_tool_map[output.tool]
            return_direct = tool.return_direct
            color = color_mapping[output.tool]
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""
            # We then call the tool on the tool input to get an observation
            observation = await tool.arun(
                output.tool_input, verbose=self.verbose, color=color, **tool_run_kwargs
            )
        else:
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = await InvalidTool().arun(
                output.tool, verbose=self.verbose, color=None, **tool_run_kwargs
            )
            return_direct = False
        return output, observation

    def _call(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # Let's start tracking the iterations the agent has gone through
        iterations = 0
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations):
            next_step_output = self._take_next_step(
                name_to_tool_map, color_mapping, inputs, intermediate_steps
            )
            if isinstance(next_step_output, AgentFinish):
                return self._return(next_step_output, intermediate_steps)

            intermediate_steps.append(next_step_output)
            # See if tool should return directly
            tool_return = self._get_tool_return(next_step_output)
            if tool_return is not None:
                return self._return(tool_return, intermediate_steps)
            iterations += 1
        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(output, intermediate_steps)

    async def _acall(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # Let's start tracking the iterations the agent has gone through
        iterations = 0
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations):
            next_step_output = await self._atake_next_step(
                name_to_tool_map, color_mapping, inputs, intermediate_steps
            )
            if isinstance(next_step_output, AgentFinish):
                return await self._areturn(next_step_output, intermediate_steps)

            intermediate_steps.append(next_step_output)
            # See if tool should return directly
            tool_return = self._get_tool_return(next_step_output)
            if tool_return is not None:
                return await self._areturn(tool_return, intermediate_steps)

            iterations += 1
        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return await self._areturn(output, intermediate_steps)

    def _get_tool_return(
        self, next_step_output: Tuple[AgentAction, str]
    ) -> Optional[AgentFinish]:
        """Check if the tool is a returning tool."""
        agent_action, observation = next_step_output
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # Invalid tools won't be in the map, so we return False.
        if agent_action.tool in name_to_tool_map:
            if name_to_tool_map[agent_action.tool].return_direct:
                return AgentFinish(
                    {self.agent.return_values[0]: observation},
                    "",
                )
        return None
