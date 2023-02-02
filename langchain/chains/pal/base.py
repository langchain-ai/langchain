"""Implements Program-Aided Language Models.

As in https://arxiv.org/pdf/2211.10435.pdf.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.pal.colored_object_prompt import COLORED_OBJECT_PROMPT
from langchain.chains.pal.math_prompt import MATH_PROMPT
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate
from langchain.python import PythonREPL


class PALChain(Chain, BaseModel):
    """Implements Program-Aided Language Models."""

    llm: BaseLLM
    prompt: BasePromptTemplate
    stop: str = "\n\n"
    get_answer_expr: str = "print(solution())"
    python_globals: Optional[Dict[str, Any]] = None
    python_locals: Optional[Dict[str, Any]] = None
    output_key: str = "result"  #: :meta private:
    return_intermediate_steps: bool = False

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
        if not self.return_intermediate_steps:
            return [self.output_key]
        else:
            return [self.output_key, "intermediate_steps"]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
        code = llm_chain.predict(stop=[self.stop], **inputs)
        self.callback_manager.on_text(
            code, color="green", end="\n", verbose=self.verbose
        )
        repl = PythonREPL(_globals=self.python_globals, _locals=self.python_locals)
        res = repl.run(code + f"\n{self.get_answer_expr}")
        output = {self.output_key: res.strip()}
        if self.return_intermediate_steps:
            output["intermediate_steps"] = code
        return output

    @classmethod
    def from_math_prompt(cls, llm: BaseLLM, **kwargs: Any) -> PALChain:
        """Load PAL from math prompt."""
        return cls(
            llm=llm,
            prompt=MATH_PROMPT,
            stop="\n\n",
            get_answer_expr="print(solution())",
            **kwargs,
        )

    @classmethod
    def from_colored_object_prompt(cls, llm: BaseLLM, **kwargs: Any) -> PALChain:
        """Load PAL from colored object prompt."""
        return cls(
            llm=llm,
            prompt=COLORED_OBJECT_PROMPT,
            stop="\n\n\n",
            get_answer_expr="print(answer)",
            **kwargs,
        )

    @property
    def _chain_type(self) -> str:
        return "pal_chain"
