"""A chain for evaluating ReAct style agents.

This chain is used to evaluate ReAct style agents by reasoning about
the sequence of actions taken and their outcomes. It uses a language model
chain (LLMChain) to generate the reasoning and scores.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from pydantic import Extra, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.evaluation.agents.trajectory_eval_prompt import (
    EVAL_CHAT_PROMPT,
    TOOL_FREE_EVAL_CHAT_PROMPT,
)
from langchain.evaluation.schema import AgentTrajectoryEvaluator, LLMEvalChain
from langchain.schema import AgentAction, BaseOutputParser, OutputParserException
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools.base import BaseTool


class TrajectoryEval(NamedTuple):
    """A named tuple containing the score and reasoning for a trajectory."""

    score: float
    """The score for the trajectory, normalized from 0 to 1.s"""
    reasoning: str
    """The reasoning for the score."""


class TrajectoryOutputParser(BaseOutputParser):
    @property
    def _type(self) -> str:
        return "agent_trajectory"

    def parse(self, text: str) -> TrajectoryEval:
        """Parse the output text and extract the score and reasoning.

        Args:
            text (str): The output text to parse.

        Returns:
            TrajectoryEval: A named tuple containing the normalized score and reasoning.

        Raises:
            OutputParserException: If the score is not found in the output text or
                if the LLM's score is not a digit in the range 1-5.
        """
        if "Score:" not in text:
            raise OutputParserException(
                f"Could not find score in model eval output: {text}"
            )

        reasoning, score_str = text.split("Score: ", maxsplit=1)

        reasoning, score_str = reasoning.strip(), score_str.strip()

        score_str = next(
            (char for char in score_str if char.isdigit()), "0"
        )  # Scan for first digit

        if not 1 <= int(score_str) <= 5:
            raise OutputParserException(
                f"Score is not a digit in the range 1-5: {text}"
            )
        normalized_score = (int(score_str) - 1) / 4
        return TrajectoryEval(score=normalized_score, reasoning=reasoning)


class TrajectoryEvalChain(AgentTrajectoryEvaluator, LLMEvalChain):
    """A chain for evaluating ReAct style agents.

    This chain is used to evaluate ReAct style agents by reasoning about
    the sequence of actions taken and their outcomes.

    Example:

    .. code-block:: python

        from langchain.agents import AgentType, initialize_agent
        from langchain.chat_models import ChatOpenAI
        from langchain.evaluation import TrajectoryEvalChain
        from langchain.tools import tool

        @tool
        def geography_answers(country: str, question: str) -> str:
            \"\"\"Very helpful answers to geography questions.\"\"\"
            return f"{country}? IDK - We may never know {question}."

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        agent = initialize_agent(
            tools=[geography_answers],
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            return_intermediate_steps=True,
        )

        question = "How many dwell in the largest minor region in Argentina?"
        response = agent(question)

        eval_chain = TrajectoryEvalChain.from_llm(
            llm=llm, agent_tools=[geography_answers], return_reasoning=True
        )

        result = eval_chain.evaluate_agent_trajectory(
            input=question,
            agent_trajectory=response["intermediate_steps"],
            prediction=response["output"],
            reference="Paris",
        )
        print(result["score"])
        # 0
    """  # noqa: E501

    agent_tools: Optional[List[BaseTool]] = None
    """A list of tools available to the agent."""
    eval_chain: LLMChain
    """The language model chain used for evaluation."""
    output_parser: TrajectoryOutputParser = Field(
        default_factory=TrajectoryOutputParser
    )
    """The output parser used to parse the output."""
    return_reasoning: bool = False
    """Whether to return the reasoning along with the score."""

    class Config:
        """Configuration for the QAEvalChain."""

        extra = Extra.ignore

    @property
    def _tools_description(self) -> str:
        """Get the description of the agent tools.

        Returns:
            str: The description of the agent tools.
        """
        if self.agent_tools is None:
            return ""
        return "\n\n".join(
            [
                f"""Tool {i}: {tool.name}
Description: {tool.description}"""
                for i, tool in enumerate(self.agent_tools, 1)
            ]
        )

    @staticmethod
    def get_agent_trajectory(
        steps: Union[str, Sequence[Tuple[AgentAction, str]]]
    ) -> str:
        """Get the agent trajectory as a formatted string.

        Args:
            steps (Union[str, List[Tuple[AgentAction, str]]]): The agent trajectory.

        Returns:
            str: The formatted agent trajectory.
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
            ]
        )

    @staticmethod
    def _format_reference(reference: Optional[str]) -> str:
        """Format the reference text.

        Args:
            reference (str): The reference text.

        Returns:
            str: The formatted reference text.
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
        agent_tools: Optional[Sequence[BaseTool]] = None,
        output_parser: Optional[TrajectoryOutputParser] = None,
        return_reasoning: bool = True,
        **kwargs: Any,
    ) -> "TrajectoryEvalChain":
        """Create a TrajectoryEvalChain object from a language model chain.

        Args:
            llm (BaseChatModel): The language model chain.
            agent_tools (Optional[Sequence[BaseTool]]): A list of tools
                available to the agent.
            output_parser (Optional[TrajectoryOutputParser]): The output parser
                used to parse the chain output into a score.
            return_reasoning (bool): Whether to return the
                reasoning along with the score.

        Returns:
            TrajectoryEvalChain: The TrajectoryEvalChain object.
        """
        if not isinstance(llm, BaseChatModel):
            raise NotImplementedError(
                "Only chat models supported by the current trajectory eval"
            )
        if agent_tools:
            prompt = EVAL_CHAT_PROMPT
        else:
            prompt = TOOL_FREE_EVAL_CHAT_PROMPT
        eval_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(
            agent_tools=agent_tools,
            return_reasoning=return_reasoning,
            eval_chain=eval_chain,
            output_parser=output_parser or TrajectoryOutputParser(),
            **kwargs,
        )

    @property
    def input_keys(self) -> List[str]:
        """Get the input keys for the chain.

        Returns:
            List[str]: The input keys.
        """
        return ["question", "agent_trajectory", "answer", "reference"]

    @property
    def output_keys(self) -> List[str]:
        """Get the output keys for the chain.

        Returns:
            List[str]: The output keys.
        """
        if self.return_reasoning:
            return ["score", "reasoning"]
        return ["score"]

    def prep_inputs(self, inputs: Union[Dict[str, Any], Any]) -> Dict[str, str]:
        """Validate and prep inputs."""
        if "reference" not in inputs:
            inputs["reference"] = self._format_reference(inputs.get("reference"))
        return super().prep_inputs(inputs)

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run the chain and generate the output.

        Args:
            inputs (Dict[str, str]): The input values for the chain.
            run_manager (Optional[CallbackManagerForChainRun]): The callback
                manager for the chain run.

        Returns:
            Dict[str, Any]: The output values of the chain.
        """
        chain_input = {**inputs}
        if self.agent_tools:
            chain_input["tool_descriptions"] = self._tools_description
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        raw_output = self.eval_chain.run(
            chain_input, callbacks=_run_manager.get_child()
        )
        parsed_output = self.output_parser.parse(raw_output)

        if self.return_reasoning:
            return {"score": parsed_output.score, "reasoning": parsed_output.reasoning}

        return {"score": parsed_output.score}

    async def _acall(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run the chain and generate the output.

        Args:
            inputs (Dict[str, str]): The input values for the chain.
            run_manager (Optional[CallbackManagerForChainRun]): The callback
                manager for the chain run.

        Returns:
            Dict[str, Any]: The output values of the chain.
        """
        chain_input = {**inputs}
        if self.agent_tools:
            chain_input["tool_descriptions"] = self._tools_description
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        raw_output = await self.eval_chain.arun(
            chain_input, callbacks=_run_manager.get_child()
        )
        parsed_output = self.output_parser.parse(raw_output)

        if self.return_reasoning:
            return {"score": parsed_output.score, "reasoning": parsed_output.reasoning}

        return {"score": parsed_output.score}

    def _evaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        input: str,
        agent_trajectory: Sequence[Tuple[AgentAction, str]],
        reference: Optional[str] = None,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Evaluate a trajectory.

        Args:
            prediction (str): The final predicted response.
            input (str): The input to the agent.
            agent_trajectory (List[Tuple[AgentAction, str]]):
                The intermediate steps forming the agent trajectory.
            reference (Optional[str]): The reference answer.
            callbacks (Callbacks): Callbacks to use for this chain run.

        Returns:
            dict: The evaluation result, which includes the score and optionally
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

    async def _aevaluate_agent_trajectory(
        self,
        *,
        prediction: str,
        input: str,
        agent_trajectory: Sequence[Tuple[AgentAction, str]],
        reference: Optional[str] = None,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Asynchronously evaluate a trajectory.

        Args:
            prediction (str): The final predicted response.
            input (str): The input to the agent.
            agent_trajectory (List[Tuple[AgentAction, str]]):
                The intermediate steps forming the agent trajectory.
            reference (Optional[str]): The reference answer.
            callbacks (Callbacks): Callbacks to use for this chain run.

        Returns:
            dict: The evaluation result, which includes the score and optionally
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
