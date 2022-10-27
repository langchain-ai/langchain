"""Implement an LLM driven browser."""
from typing import Dict, List, Optional

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import SelfConsistencyLLMChain
from langchain.chains.self_consistency.prompts.anli_prompt import ANLI_PROMPT
from langchain.chains.self_consistency.prompts.aqua_rat_prompt import AQUA_RAT_PROMPT
from langchain.chains.self_consistency.prompts.arc_prompt import ARC_PROMPT
from langchain.chains.self_consistency.prompts.arithmetic_reasoning_prompt import (
    ARITHMETIC_REASONING_PROMPT,
)
from langchain.chains.self_consistency.prompts.boolq_prompt import BOOLQ_PROMPT
from langchain.chains.self_consistency.prompts.hotpotqa_prompt import HOTPOTQA_PROMPT
from langchain.chains.self_consistency.prompts.esnli_prompt import ESNLI_PROMPT
from langchain.llms.base import LLM
from langchain.llms.openai import OpenAI
from langchain.prompt import Prompt

_CLASS_TO_PROMPT: Dict[str, Prompt] = {
    "anli": ANLI_PROMPT,
    "aqua_rat": AQUA_RAT_PROMPT,
    "arc": ARC_PROMPT,
    "arithmetic_reasoning": ARITHMETIC_REASONING_PROMPT,
    "boolq": BOOLQ_PROMPT,
    "esnli": ESNLI_PROMPT,
    "hotpotqa": HOTPOTQA_PROMPT,
}

# TODO: Add auto-routing and more prompts
_FALLBACK_MAP: Dict[str, str] = {
    "nli": "anli",
    "natural_language_inference": "anli",
    "rte": "anli",
    "math": "aqua_rat",
    "qna": "hotpotqa",
}


class SelfConsistencyChain(Chain, BaseModel):
    """Implement an LLM chain to reason in a self-consistent manner.

    Based on Self-Consistency Improves Chain of Thought Reasoning in
    Language Models

    Example:
        .. code-block:: python

            from langchain import SelfConsistencyChain, OpenAI
            natbot = SelfConsistencyChain(llm=OpenAI(), objective="Buy me a new hat.")
    """

    llm: LLM
    """LLM wrapper to use."""
    default_task: str
    """The default task to run."""
    input_key: str = "prompt_inputs"  #: :meta private:
    output_key: str = "answer"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @classmethod
    def from_default(cls, objective: str) -> "SelfConsistencyChain":
        """Load with default LLM."""
        llm = OpenAI(temperature=0.5, best_of=10, n=3, max_tokens=50)
        return cls(llm=llm, objective=objective)

    @property
    def input_keys(self) -> List[str]:
        """Expect different keys depending on the task.

        :meta private:
        """
        return [self.input_key, "task"]

    @property
    def output_keys(self) -> List[str]:
        """Return command.

        :meta private:
        """
        return [self.output_key]

    def _get_prompt(self, task: Optional[str]) -> Prompt:
        """Get the prompt for the task."""
        
        if task in _CLASS_TO_PROMPT:
            return _CLASS_TO_PROMPT[task]
        if task in _FALLBACK_MAP:
            return _CLASS_TO_PROMPT[_FALLBACK_MAP[task]]
        raise ValueError(f"Unknown task {task}")

    def _run(self, inputs: Dict[str, str]) -> Dict[str, str]:
        task = inputs["task"]
        prompt = self._get_prompt(task)
        llm_executor = SelfConsistencyLLMChain(prompt=prompt, llm=self.llm)
        llm_inputs = inputs[self.input_key]
        if 'choices' in llm_inputs:
            if isinstance(llm_inputs['choices'], list):
                llm_inputs['choices'] = ' '.join([f"({chr(97+i)}) {choice}" for i, choice in enumerate(llm_inputs['choices'])])
        answer = llm_executor.predict(**llm_inputs)
        return {self.output_key: answer}

    def run(self, **kwargs: str) -> str:
        """Figure out next browser command to run."""
        task = kwargs.pop("task", self.default_task)
        _inputs = {
            self.input_key: kwargs,
            "task": task,
        }
        return self(_inputs)[self.output_key]
