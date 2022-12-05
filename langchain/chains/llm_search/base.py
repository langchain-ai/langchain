"""Chain that passes query to google and then uses an LLM to parse results."""
from __future__ import annotations

from typing import Any, Dict, List

import requests
from pydantic import BaseModel, Extra, root_validator

from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.chains.llm_search.prompt import PROMPT
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"  # noqa: E501
}


class LLMSearchChain(Chain, BaseModel):
    """Chain that passes query to google and then uses an LLM to parse results."""

    llm_chain: LLMChain
    headers: dict
    question_key: str = "question"  #: :meta private:
    answer_key: str = "answer"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return [self.question_key]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.answer_key]

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        try:
            from bs4 import BeautifulSoup  # noqa: F401

        except ImportError:
            raise ValueError(
                "Could not import bs4 python package. "
                "Please it install it with `pip install bs4`."
            )
        return values

    @classmethod
    def from_llm(
        cls,
        llm: LLM,
        prompt: PromptTemplate = PROMPT,
        headers: dict = DEFAULT_HEADERS,
        **kwargs: Any,
    ) -> LLMSearchChain:
        """Load search chain from LLM."""
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, headers=headers, **kwargs)

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        from bs4 import BeautifulSoup

        question = inputs[self.question_key]
        question_url = "https://www.google.com/search?q=" + question.replace(" ", "+")
        r = requests.get(question_url, headers=self.headers)
        # extract the text from the html
        soup = BeautifulSoup(r.text, "html.parser")
        result = self.llm_chain.predict(
            query=question, search_results=soup.get_text()[:8000]
        )
        return {self.answer_key: result}
