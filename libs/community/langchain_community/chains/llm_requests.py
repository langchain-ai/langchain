"""Chain that hits a URL and then uses an LLM to parse results."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from pydantic import ConfigDict, Field, model_validator

from langchain_community.utilities.requests import TextRequestsWrapper

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"  # noqa: E501
}


class LLMRequestsChain(Chain):
    """Chain that requests a URL and then uses an LLM to parse results.

    **Security Note**: This chain can make GET requests to arbitrary URLs,
        including internal URLs.

        Control access to who can run this chain and what network access
        this chain has.

        See https://python.langchain.com/docs/security for more information.
    """

    llm_chain: LLMChain
    requests_wrapper: TextRequestsWrapper = Field(
        default_factory=lambda: TextRequestsWrapper(headers=DEFAULT_HEADERS),
        exclude=True,
    )
    text_length: int = 8000
    requests_key: str = "requests_result"  #: :meta private:
    input_key: str = "url"  #: :meta private:
    output_key: str = "output"  #: :meta private:

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

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

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and python package exists in environment."""
        try:
            from bs4 import BeautifulSoup  # noqa: F401

        except ImportError:
            raise ImportError(
                "Could not import bs4 python package. "
                "Please install it with `pip install bs4`."
            )
        return values

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        from bs4 import BeautifulSoup

        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        # Other keys are assumed to be needed for LLM prediction
        other_keys = {k: v for k, v in inputs.items() if k != self.input_key}
        url = inputs[self.input_key]
        res = self.requests_wrapper.get(url)
        # extract the text from the html
        soup = BeautifulSoup(res, "html.parser")
        other_keys[self.requests_key] = soup.get_text()[: self.text_length]
        result = self.llm_chain.predict(
            callbacks=_run_manager.get_child(), **other_keys
        )
        return {self.output_key: result}

    @property
    def _chain_type(self) -> str:
        return "llm_requests_chain"
