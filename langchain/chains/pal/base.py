"""Implements Program-Aided Language Models.

As in https://arxiv.org/pdf/2211.10435.pdf.
"""
from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.pal.colored_object_prompt import COLORED_OBJECT_PROMPT
from langchain.chains.pal.math_prompt import MATH_PROMPT
from langchain.chains.python import PythonChain
from langchain.input import print_text
from langchain.llms.base import LLM
from langchain.prompts.base import BasePromptTemplate


class PALChain(Chain, BaseModel):
    """Implements Program-Aided Language Models."""

    llm: LLM
    prompt: BasePromptTemplate
    stop: str = "\n\n"
    get_answer_expr: str = "print(solution())"
    output_key: str = "result"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
        code = llm_chain.predict(stop=[self.stop], **inputs)
        if self.verbose:
            print_text(code, color="green", end="\n")
        repl = PythonChain()
        res = repl.run(code + f"\n{self.get_answer_expr}")
        return {self.output_key: res.strip()}

    @classmethod
    def from_math_prompt(cls, llm: LLM, **kwargs: Any) -> PALChain:
        """Load PAL from math prompt."""
        return cls(
            llm=llm,
            prompt=MATH_PROMPT,
            stop="\n\n",
            get_answer_expr="print(solution())",
            **kwargs,
        )

    @classmethod
    def from_colored_object_prompt(cls, llm: LLM, **kwargs: Any) -> PALChain:
        """Load PAL from colored object prompt."""
        return cls(
            llm=llm,
            prompt=COLORED_OBJECT_PROMPT,
            stop="\n\n\n",
            get_answer_expr="print(answer)",
            **kwargs,
        )
