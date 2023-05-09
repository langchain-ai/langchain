"""LLM Chain specifically for generating examples for question answering."""
from __future__ import annotations

from typing import Any

from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.evaluation.qa.generate_prompt import PROMPT


class QAGenerateChain(LLMChain):
    """LLM Chain specifically for generating examples for question answering."""

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, **kwargs: Any) -> QAGenerateChain:
        """Load QA Generate Chain from LLM."""
        return cls(llm=llm, prompt=PROMPT, **kwargs)
