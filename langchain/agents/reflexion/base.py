# TODO
# - clean up inheritance structure
# - integrate callback manager, run manager, ...
# - create a short default prompt for reflexion?
# - test, document, lint, ...

"""Chain that implements the Reflexion paper from https://arxiv.org/abs/2303.11366."""
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pprint import pprint
from pydantic import Field
from langchain.agents.agent import Agent, AgentExecutor, AgentOutputParser
from langchain.agents.agent_types import AgentType
from langchain.agents.react.base import DocstoreExplorer, ReActChain
from langchain.agents.react.base import ReActDocstoreAgent
from langchain.agents.reflexion.alfworld_prompt import ALFWORLD_PROMPT
from langchain.agents.reflexion.output_parser import ReflexionOutputParser
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import CallbackManagerForChainRun, Callbacks
from langchain.chains.llm import LLMChain
from langchain.docstore.base import Docstore
from langchain.input import get_color_mapping
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools.base import BaseTool, Tool


class ReflexerDocstoreAgent(ReActDocstoreAgent):
    """Agent for the Reflexer chain."""

    reflexion_llm_chain: LLMChain
    reflexion_output_parser: ReflexionOutputParser = Field(default_factory=ReflexionOutputParser)

    @classmethod
    def _get_default_reflexion_output_parser(cls, **kwargs: Any) -> ReflexionOutputParser:
        return ReflexionOutputParser()

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        return AgentType.REFLEXION_DOCSTORE

    @classmethod
    def create_reflexion_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        """Return default prompt."""
        print("ReflexerDocstoreAgent:create_reflexion_prompt")  # DEBUG
        return ALFWORLD_PROMPT

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        output_parser: Optional[AgentOutputParser] = None,
        reflexion_llm: Optional[BaseLanguageModel] = None,
        reflexion_output_parser: Optional[ReflexionOutputParser] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""

        print("ReflexerDocstoreAgent:from_llm_and_tools with kwargs:")  # DEBUG
        pprint(kwargs)

        cls._validate_tools(tools)

        planning_llm_chain = LLMChain(
            llm=llm,
            prompt=cls.create_prompt(tools),
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        _output_parser = output_parser or cls._get_default_output_parser()

        if reflexion_llm is None:
            reflexion_llm = llm

        reflexion_llm_chain = LLMChain(
            llm=llm,
            prompt=cls.create_reflexion_prompt(tools),
            callback_manager=None, # TODO: Integrate callback manager
        )
        _reflexion_output_parser = reflexion_output_parser or cls._get_default_reflexion_output_parser()

        return ReflexerDocstoreAgent(
            llm_chain=planning_llm_chain,    
            reflexion_llm_chain=reflexion_llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            reflexion_output_parser=_reflexion_output_parser,
            **kwargs,
        )

    def _reflect(
            self,
            inputs: Dict[str, str],
            intermediate_steps: List[Tuple[AgentAction, str]],
            next_trial_no: int,
            callbacks: Callbacks = None,
            **kwargs: Any,) -> str:
        """ returns full relection notes  """

        full_inputs = self.get_full_inputs(intermediate_steps, **inputs, **kwargs)
        full_output = self.reflexion_llm_chain.predict(
            callbacks=callbacks, **full_inputs
        )
        reflexion = self.reflexion_output_parser.parse(full_output)

        return (self._construct_scratchpad(intermediate_steps) +
                 "\nSTATUS: FAIL\nNew plan: " + reflexion + f"\n\nTrial {next_trial_no}:\n")

class ReflexionChain(ReActChain):
    """Chain that implements the Reflexion paper.

    Example:
        .. code-block:: python

            TODO
    """
    max_trials: Optional[int] = 3
    max_action_repetition: Optional[int] = 2
    max_iterations_per_trial: Optional[int] = 15
    max_execution_time_per_trial: Optional[float] = None

    def __init__(self, llm: BaseLanguageModel, docstore: Docstore, **kwargs: Any):
        """Initialize with the LLM and a docstore."""

        docstore_explorer = DocstoreExplorer(docstore)
        tools = [
            Tool(
                name="Search",
                func=docstore_explorer.search,
                description="Search for a term in the docstore.",
            ),
            Tool(
                name="Lookup",
                func=docstore_explorer.lookup,
                description="Lookup a term in the docstore.",
            ),
        ]
        agent = ReflexerDocstoreAgent.from_llm_and_tools(llm, tools)
        # Skip ReActChain's init and use AgentExecutor's instead -- TODO Fix hierarchy
        AgentExecutor.__init__(self, agent=agent, tools=tools, **kwargs)

    def _should_continue(
            self,
            trials: int,
            total_iterations: int,
            total_time_elapsed: float) -> bool:
        if self.max_trials is not None and trials > self.max_trials:
            return False
        
        return super()._should_continue(
            iterations=total_iterations,
            time_elapsed=total_time_elapsed
        )

    def _should_reflect(
            self, 
            intermediate_steps: List[Tuple[AgentAction, str]],
            iterations_in_trial: int,
            time_elapsed_in_trial: float,
            _max_action_repetition: Optional[int] = None,
            _max_iterations_per_trial: Optional[int] = None,
            _max_time_elapsed_per_trial: Optional[float] = None,) -> bool:

        # We reflect when ...
        # ... we have too many iterations in current trial, or
        if (_max_iterations_per_trial is not None
            and iterations_in_trial >= _max_iterations_per_trial):
            print("Trial failed due to max_iterations_in_trial reached")
            return True
        # ... current trial took too long, or
        if (_max_time_elapsed_per_trial is not None
            and time_elapsed_in_trial >= _max_time_elapsed_per_trial):
            print("Trial failed due to _max_time_elapsed_per_trial reached")
            return True
        # ... we're stuck in an action loop, or
        if (_max_action_repetition is not None
            and _count_repetitions(intermediate_steps) >= _max_action_repetition):
            print("Trial failed due to _max_action_repetition reached")
            return True
        # ... we're done, but the task was not succesful
        # TODO

        return False


    def _reflect(
            self,
            inputs: Dict[str, str],
            intermediate_steps: List[Tuple[AgentAction, str]],
            run_manager: Optional[CallbackManagerForChainRun] = None,
            **kwargs: Any,) -> str:
        return self.agent.reflexion_llm_chain._reflect()

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # Let's start tracking the number of iterations and time elapsed
        # for total execution, and for current trial
        total_iterations = 0
        total_time_elapsed = 0.0
        total_start_time = time.time()
        trials = 1

        trial_iterations = 0
        trial_time_elapsed = 0.0
        trial_start_time = time.time()
        
        reflexion_notes = "Trial 1:\n"

        # We now enter the agent loop (until it returns something).
        while self._should_continue(trials, total_iterations, total_time_elapsed):

            # DEBUG - BAAAAD HACK: stuff reflection_notes into input
            # TODO - remove hack
            input_with_reflection_notes = inputs.copy()
            input_with_reflection_notes["input"] += "\n" + reflexion_notes

            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                input_with_reflection_notes,
                intermediate_steps,
                run_manager=run_manager,
            )
            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    next_step_output, intermediate_steps, run_manager=run_manager
                )

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(tool_return, intermediate_steps)
                
            total_iterations += 1
            total_time_elapsed = time.time() - total_start_time
        
            trial_iterations += 1
            trial_time_elapsed = time.time() - trial_start_time

            # Check if we trial failed. If yes, we reflect and start a new trial
            trial_failed = self._should_reflect(
                intermediate_steps,
                trial_iterations,
                trial_time_elapsed,
                self.max_action_repetition,
                self.max_iterations_per_trial,
                self.max_execution_time_per_trial
            )

            if trial_failed:
                new_reflection_notes = self.agent._reflect(inputs, intermediate_steps,
                                                           next_trial_no=trials+1)
                reflexion_notes += "\n" + new_reflection_notes

                # TODO: Add to some log / manager that new trial started

                trials += 1
                trial_iterations = 0
                trial_time_elapsed = 0.0
                trial_start_time = time.time()
                intermediate_steps = []


        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(output, intermediate_steps)
    
@staticmethod 
def _count_repetitions(intermediate_steps: List[Tuple[AgentAction, str]]) -> int:
    last_tool = intermediate_steps[-1][0].tool
    last_tool_input = intermediate_steps[-1][0].tool_input

    count = 0

    # Iterate over intermediate_steps in reverse order,
    # starting from second last element
    for action, _ in reversed(intermediate_steps[:-1]):
        if action.tool == last_tool and action.tool_input == last_tool_input:
            count += 1
        else:
            break

    return count
