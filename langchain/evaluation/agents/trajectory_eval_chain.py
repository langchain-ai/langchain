"""A chain for evaluating ReAct style agents."""
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.agents.trajectory_eval_prompt import EVAL_CHAT_PROMPT
from langchain.schema import AgentAction, BaseOutputParser, OutputParserException
from langchain.tools.base import BaseTool


class TrajectoryEval(NamedTuple):
    score: int
    reasoning: str


class TrajectoryOutputParser(BaseOutputParser):
    def parse(self, text: str) -> TrajectoryEval:
        if "Score:" not in text:
            raise OutputParserException(
                f"Could not find score in model eval output: {text}"
            )

        reasoning, score_str = text.split("Score: ")

        reasoning, score_str = reasoning.strip(), score_str.strip()

        score_str = next(
            (char for char in score_str if char.isdigit()), "0"
        )  # Scan for first digit

        if not 1 <= int(score_str) <= 5:
            raise OutputParserException(
                f"Score is not a digit in the range 1-5: {text}"
            )

        return TrajectoryEval(score=int(score_str), reasoning=reasoning)


class TrajectoryEvalChain(Chain):
    agent_tools: List[BaseTool]
    eval_chain: LLMChain
    output_parser: TrajectoryOutputParser
    return_reasoning: bool = False

    @property
    def _tools_description(self) -> str:
        return "\n\n".join(
            [
                f"""Tool {i}: {tool.name}
Description: {tool.description}"""
                for i, tool in enumerate(self.agent_tools, 1)
            ]
        )

    @staticmethod
    def get_agent_trajectory(steps: Union[str, List[Tuple[AgentAction, str]]]) -> str:
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

    @classmethod
    def from_llm(
        cls,
        llm: ChatOpenAI,
        agent_tools: Sequence[BaseTool],
        output_parser: Optional[TrajectoryOutputParser] = None,
        return_reasoning: bool = False,
    ) -> "TrajectoryEvalChain":
        eval_chain = LLMChain(llm=llm, prompt=EVAL_CHAT_PROMPT)
        return cls(
            agent_tools=agent_tools,
            return_reasoning=return_reasoning,
            eval_chain=eval_chain,
            output_parser=output_parser or TrajectoryOutputParser(),
        )

    @property
    def input_keys(self) -> List[str]:
        return ["question", "agent_trajectory", "answer"]

    @property
    def output_keys(self) -> List[str]:
        if self.return_reasoning:
            return ["score", "reasoning"]
        return ["score"]

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        raw_output = self.eval_chain.run(
            {"tool_descriptions": self._tools_description, **inputs}
        )
        parsed_output = self.output_parser.parse(raw_output)

        if self.return_reasoning:
            return {"score": parsed_output.score, "reasoning": parsed_output.reasoning}

        return {"score": parsed_output.score}
