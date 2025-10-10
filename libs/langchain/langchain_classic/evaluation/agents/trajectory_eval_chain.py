"""A chain for evaluating ReAct style agents.

This chain is used to evaluate ReAct style agents by reasoning about
the sequence of actions taken and their outcomes. It uses a language model
chain (LLMChain) to generate the reasoning and scores.
"""

import re
from collections.abc import Sequence
from typing import (
    Any,
    TypedDict,
    cast,
)

from langchain_core.agents import AgentAction
from langchain_core.callbacks import Callbacks
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field
from typing_extensions import override

from langchain_classic.chains.llm import LLMChain
from langchain_classic.evaluation.agents.trajectory_eval_prompt import (
    EVAL_CHAT_PROMPT,
    TOOL_FREE_EVAL_CHAT_PROMPT,
)
from langchain_classic.evaluation.schema import AgentTrajectoryEvaluator, LLMEvalChain

_MAX_SCORE = 5


class TrajectoryEval(TypedDict):
    """A named tuple containing the score and reasoning for a trajectory."""

    score: float
    """The score for the trajectory, normalized from 0 to 1."""
    reasoning: str
    """The reasoning for the score."""


class TrajectoryOutputParser(BaseOutputParser):
    """Trajectory output parser."""

    @property
    def _type(self) -> str:
        return "agent_trajectory"

    def parse(self, text: str) -> TrajectoryEval:
        """Parse the output text and extract the score and reasoning.

        Args:
            text: The output text to parse.

        Returns:
            A named tuple containing the normalized score and reasoning.

        Raises:
            If the score is not found in the output text or if the LLM's score is not a
            digit in the range 1-5.
        """
        if "Score:" not in text:
            msg = f"Could not find score in model eval output: {text}"
            raise OutputParserException(msg)

        reasoning, score_str = text.split("Score: ", maxsplit=1)

        reasoning, score_str = reasoning.strip(), score_str.strip()

        # Use regex to extract the score.
        # This will get the number in the string, even if it is a float or more than 10.
        # E.g. "Score: 1" will return 1, "Score: 3.5" will return 3.5, and
        # "Score: 10" will return 10.
        # The score should be an integer digit in the range 1-5.
        _score = re.search(r"(\d+(\.\d+)?)", score_str)
        # If the score is not found or is a float, raise an exception.
        if _score is None or "." in _score.group(1):
            msg = f"Score is not an integer digit in the range 1-5: {text}"
            raise OutputParserException(msg)
        score = int(_score.group(1))
        # If the score is not in the range 1-5, raise an exception.
        if not 1 <= score <= _MAX_SCORE:
            msg = f"Score is not a digit in the range 1-5: {text}"
            raise OutputParserException(msg)
        normalized_score = (score - 1) / (_MAX_SCORE - 1)
        return TrajectoryEval(score=normalized_score, reasoning=reasoning)


class TrajectoryEvalChain(AgentTrajectoryEvaluator, LLMEvalChain):
    """A chain for evaluating ReAct style agents.

    This chain is used to evaluate ReAct style agents by reasoning about
    the sequence of actions taken and their outcomes.
    Based on the paper "ReAct: Synergizing Reasoning and Acting in Language Models"
    (https://arxiv.org/abs/2210.03629)

    Example:
    ```python
    from langchain_classic.agents import AgentType, initialize_agent
    from langchain_community.chat_models import ChatOpenAI
    from langchain_classic.evaluation import TrajectoryEvalChain
    from langchain_classic.tools import tool

    @tool
    def geography_answers(country: str, question: str) -> str:
        \"\"\"Very helpful answers to geography questions.\"\"\"
        return f"{country}? IDK - We may never know {question}."

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    agent = initialize_agent(
        tools=[geography_answers],
        llm=model,
        agent=AgentType.OPENAI_FUNCTIONS,
        return_intermediate_steps=True,
    )

    question = "How many dwell in the largest minor region in Argentina?"
    response = agent(question)

    eval_chain = TrajectoryEvalChain.from_llm(
        llm=model, agent_tools=[geography_answers], return_reasoning=True
    )

    result = eval_chain.evaluate_agent_trajectory(
        input=question,
        agent_trajectory=response["intermediate_steps"],
        prediction=response["output"],
        reference="Paris",
    )
    print(result["score"])  # noqa: T201
    # 0

    ```
    """

    agent_tools: list[BaseTool] | None = None
    """A list of tools available to the agent."""
    eval_chain: LLMChain
    """The language model chain used for evaluation."""
    output_parser: TrajectoryOutputParser = Field(
        default_factory=TrajectoryOutputParser,
    )
    """The output parser used to parse the output."""
    return_reasoning: bool = False  # :meta private:
    """DEPRECATED. Reasoning always returned."""

    model_config = ConfigDict(
        extra="ignore",
    )

    @property
    def requires_reference(self) -> bool:
        """Whether this evaluator requires a reference label."""
        return False

    @property
    def _tools_description(self) -> str:
        """Get the description of the agent tools.

        Returns:
            The description of the agent tools.
        """
        if self.agent_tools is None:
            return ""
        return "\n\n".join(
            [
                f"""Tool {i}: {tool.name}
Description: {tool.description}"""
                for i, tool in enumerate(self.agent_tools, 1)
            ],
        )

    @staticmethod
    def get_agent_trajectory(
        steps: str | Sequence[tuple[AgentAction, str]],
    ) -> str:
        """Get the agent trajectory as a formatted string.

        Args:
            steps: The agent trajectory.

        Returns:
            The formatted agent trajectory.
        """
        if isinstance(steps, str):
            return steps

        return "\n\n".join(
            [
                f"""Step {i}:
Tool used: {action.tool}
Tool input: {action.tool_input}
Tool output: {output}"""
                for i, (action, output) in enumerate(steps, 1)
            ],
        )

    @staticmethod
    def _format_reference(reference: str | None) -> str:
        """Format the reference text.

        Args:
            reference: The reference text.

        Returns:
            The formatted reference text.
        """
        if not reference:
            return ""
        return f"""

The following is the expected answer. Use this to measure correctness:
[GROUND_TRUTH]
{reference}
[END_GROUND_TRUTH]
"""

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        agent_tools: Sequence[BaseTool] | None = None,
        output_parser: TrajectoryOutputParser | None = None,
        **kwargs: Any,
    ) -> "TrajectoryEvalChain":
        """Create a TrajectoryEvalChain object from a language model chain.

        Args:
            llm: The language model chain.
            agent_tools: A list of tools available to the agent.
            output_parser : The output parser used to parse the chain output into a
                score.
            **kwargs: Additional keyword arguments.

        Returns:
            The `TrajectoryEvalChain` object.
        """
        if not isinstance(llm, BaseChatModel):
            msg = "Only chat models supported by the current trajectory eval"
            raise NotImplementedError(msg)
        prompt = EVAL_CHAT_PROMPT if agent_tools else TOOL_FREE_EVAL_CHAT_PROMPT
        eval_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(
            agent_tools=agent_tools,
            eval_chain=eval_chain,
            output_parser=output_parser or TrajectoryOutputParser(),
            **kwargs,
        )

    @property
    def input_keys(self) -> list[str]:
        """Get the input keys for the chain.

        Returns:
            The input keys.
        """
        return ["question", "agent_trajectory", "answer", "reference"]

    @property
    def output_keys(self) -> list[str]:
        """Get the output keys for the chain.

        Returns:
            The output keys.
        """
        return ["score", "reasoning"]

    def prep_inputs(self, inputs: dict[str, Any] | Any) -> dict[str, str]:
        """Validate and prep inputs."""
        inputs["reference"] = self._format_reference(inputs.get("reference"))
        return super().prep_inputs(inputs)

    def _call(
        self,
        inputs: dict[str, str],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Run the chain and generate the output.

        Args:
            inputs: The input values for the chain.
            run_manager: The callback manager for the chain run.

        Returns:
            The output values of the chain.
        """
        chain_input = {**inputs}
        if self.agent_tools:
            chain_input["tool_descriptions"] = self._tools_description
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        raw_output = self.eval_chain.run(
            chain_input,
            callbacks=_run_manager.get_child(),
        )
        return cast("dict", self.output_parser.parse(raw_output))

    async def _acall(
        self,
        inputs: dict[str, str],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Run the chain and generate the output.

        Args:
            inputs: The input values for the chain.
            run_manager: The callback manager for the chain run.

        Returns:
            The output values of the chain.
        """
        chain_input = {**inputs}
        if self.agent_tools:
            chain_input["tool_descriptions"] = self._tools_description
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        raw_output = await self.eval_chain.arun(
            chain_input,
            callbacks=_run_manager.get_child(),
        )
        return cast("dict", self.output_parser.parse(raw_output))

    @override
    def _evaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        input: str,
        agent_trajectory: Sequence[tuple[AgentAction, str]],
        reference: str | None = None,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Evaluate a trajectory.

        Args:
            prediction: The final predicted response.
            input: The input to the agent.
            agent_trajectory: The intermediate steps forming the agent trajectory.
            reference: The reference answer.
            callbacks: Callbacks to use for this chain run.
            tags: The tags to apply.
            metadata: The metadata to use.
            include_run_info: Whether to include run info in the output.
            **kwargs: Additional keyword arguments.

        Returns:
            The evaluation result, which includes the score and optionally
                the reasoning for reaching that.
        """
        inputs = {
            "question": input,
            "agent_trajectory": self.get_agent_trajectory(agent_trajectory),
            "answer": prediction,
            "reference": reference,
        }
        return self.__call__(
            inputs=inputs,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
            return_only_outputs=True,
        )

    @override
    async def _aevaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        input: str,
        agent_trajectory: Sequence[tuple[AgentAction, str]],
        reference: str | None = None,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Asynchronously evaluate a trajectory.

        Args:
            prediction: The final predicted response.
            input: The input to the agent.
            agent_trajectory: The intermediate steps forming the agent trajectory.
            reference: The reference answer.
            callbacks: Callbacks to use for this chain run.
            tags: The tags to apply.
            metadata: The metadata to use.
            include_run_info: Whether to include run info in the output.
            **kwargs: Additional keyword arguments.

        Returns:
            The evaluation result, which includes the score and optionally
                the reasoning for reaching that.
        """
        inputs = {
            "question": input,
            "agent_trajectory": self.get_agent_trajectory(agent_trajectory),
            "answer": prediction,
            "reference": reference,
        }
        return await self.acall(
            inputs=inputs,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            include_run_info=include_run_info,
            return_only_outputs=True,
        )
