"""Chain that hits a URL and then uses an LLM to parse results."""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Extra, Field, root_validator

from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.requests import RequestsWrapper

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"  # noqa: E501
}


class LLMRequestsChain(Chain, BaseModel):
    """Chain that hits a URL and then uses an LLM to parse results."""

    llm_chain: LLMChain
    requests_wrapper: RequestsWrapper = Field(
        default_factory=RequestsWrapper, exclude=True
    )
    text_length: int = 8000
    requests_key: str = "requests_result"  #: :meta private:
    input_key: str = "url"  #: :meta private:
    output_key: str = "output"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

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

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        from bs4 import BeautifulSoup

        # Other keys are assumed to be needed for LLM prediction
        other_keys = {k: v for k, v in inputs.items() if k != self.input_key}
        url = inputs[self.input_key]
        res = self.requests_wrapper.get(url)
        # extract the text from the html
        soup = BeautifulSoup(res, "html.parser")
        other_keys[self.requests_key] = soup.get_text()[: self.text_length]
        result = self.llm_chain.predict(**other_keys)
        return {self.output_key: result}

    @property
    def _chain_type(self) -> str:
        return "llm_requests_chain"
