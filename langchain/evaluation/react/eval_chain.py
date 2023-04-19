"""A chain for evaluating ReAct style agents."""
from typing import Sequence, Union

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.react.eval_prompt import EVAL_CHAT_PROMPT
from langchain.schema import AgentAction, OutputParserException
from langchain.tools.base import BaseTool


class ReactEvalChain(Chain):
    llm: ChatOpenAI
    agent_tools: list[BaseTool]
    eval_chain: LLMChain
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
    def get_agent_trajectory(steps: Union[str, list[tuple[AgentAction, str]]]) -> str:
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
        return_reasoning: bool = False,
    ) -> "ReactEvalChain":
        eval_chain = LLMChain(llm=llm, prompt=EVAL_CHAT_PROMPT)
        return cls(
            llm=llm,
            agent_tools=agent_tools,
            return_reasoning=return_reasoning,
            eval_chain=eval_chain,
        )

    @property
    def input_keys(self) -> list[str]:
        return ["question", "agent_trajectory", "answer"]

    @property
    def output_keys(self) -> list[str]:
        if self.return_reasoning:
            return ["score", "reasoning"]
        return ["score"]

    def _call(self, inputs: dict[str, str]) -> dict[str, str]:
        raw_output = self.eval_chain.run(
            dict(
                tool_descriptions=self._tools_description,
                **inputs,
            )
        )

        if "Score:" not in raw_output:
            raise OutputParserException(
                f"Could not find score in model eval output: {raw_output}"
            )

        reasoning, score_str = raw_output.split("Score: ")

        reasoning, score_str = reasoning.strip(), score_str.strip()

        if not score_str.isdigit() and 1 <= int(score_str) <= 5:
            raise OutputParserException(
                f"Score is not a digit in the range 1-5: {raw_output}"
            )

        if self.return_reasoning:
            return {"score": score_str, "reasoning": reasoning}

        return {"score": score_str}
