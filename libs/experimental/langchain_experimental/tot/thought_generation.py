"""
We provide two strategies for generating thoughts in the Tree of Thoughts (ToT)
framework to avoid repetition:

These strategies ensure that the language model generates diverse and
non-repeating thoughts, which are crucial for problem-solving tasks that require
exploration.
"""
from abc import abstractmethod
from typing import Any, Dict, List, Tuple

from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate

from langchain_experimental.pydantic_v1 import Field
from langchain_experimental.tot.prompts import get_cot_prompt, get_propose_prompt


class BaseThoughtGenerationStrategy(LLMChain):
    """
    Base class for a thought generation strategy.
    """

    c: int = 3
    """The number of children thoughts to propose at each step."""

    @abstractmethod
    def next_thought(
        self,
        problem_description: str,
        thoughts_path: Tuple[str, ...] = (),
        **kwargs: Any,
    ) -> str:
        """
        Generate the next thought given the problem description and the thoughts
        generated so far.
        """


class SampleCoTStrategy(BaseThoughtGenerationStrategy):
    """
    Sample thoughts from a Chain-of-Thought (CoT) prompt.

    This strategy works better when the thought space is rich, such as when each
    thought is a paragraph. Independent and identically distributed samples
    lead to diversity, which helps to avoid repetition.
    """

    prompt: BasePromptTemplate = Field(default_factory=get_cot_prompt)

    def next_thought(
        self,
        problem_description: str,
        thoughts_path: Tuple[str, ...] = (),
        **kwargs: Any,
    ) -> str:
        response_text = self.predict_and_parse(
            problem_description=problem_description, thoughts=thoughts_path, **kwargs
        )
        return response_text if isinstance(response_text, str) else ""


class ProposePromptStrategy(BaseThoughtGenerationStrategy):
    """
    Propose thoughts sequentially using a "propose prompt".

    This strategy works better when the thought space is more constrained, such
    as when each thought is just a word or a line. Proposing different thoughts
    in the same prompt completion helps to avoid duplication.
    """

    prompt: BasePromptTemplate = Field(default_factory=get_propose_prompt)
    tot_memory: Dict[Tuple[str, ...], List[str]] = Field(default_factory=dict)

    def next_thought(
        self,
        problem_description: str,
        thoughts_path: Tuple[str, ...] = (),
        **kwargs: Any,
    ) -> str:
        if thoughts_path not in self.tot_memory or not self.tot_memory[thoughts_path]:
            new_thoughts = self.predict_and_parse(
                problem_description=problem_description,
                thoughts=thoughts_path,
                n=self.c,
                **kwargs,
            )
            if not new_thoughts:
                return ""
            if isinstance(new_thoughts, list):
                self.tot_memory[thoughts_path] = new_thoughts[::-1]
            else:
                return ""
        return self.tot_memory[thoughts_path].pop()
