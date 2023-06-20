"""Chain that makes API calls and summarizes the responses to answer a question."""
import re
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field, root_validator

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.api.prompt import API_RESPONSE_PROMPT, API_URL_PROMPT
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts import BasePromptTemplate
from langchain.requests import TextRequestsWrapper

class APIChain(Chain):
    """Chain that makes API calls and summarizes the responses to answer a question."""

    api_request_chain: LLMChain
    api_answer_chain: LLMChain
    requests_wrapper: TextRequestsWrapper = Field(exclude=True)
    api_docs: str
    allow_unverified_urls: bool = False
    question_key: str = "question"  #: :meta private:
    output_key: str = "output"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.question_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    @root_validator(pre=True)
    def validate_api_request_prompt(cls, values: Dict) -> Dict:
        """Check that api request prompt expects the right variables."""
        input_vars = values["api_request_chain"].prompt.input_variables
        expected_vars = {"question", "api_docs"}
        if set(input_vars) != expected_vars:
            raise ValueError(
                f"Input variables should be {expected_vars}, got {input_vars}"
            )
        return values

    @root_validator(pre=True)
    def validate_api_answer_prompt(cls, values: Dict) -> Dict:
        """Check that api answer prompt expects the right variables."""
        input_vars = values["api_answer_chain"].prompt.input_variables
        expected_vars = {"question", "api_docs", "api_url", "api_response"}
        if set(input_vars) != expected_vars:
            raise ValueError(
                f"Input variables should be {expected_vars}, got {input_vars}"
            )
        return values

    def verify_api_url_is_legit(self, parsed_url: str) -> None:
        """Verify that the parsed URL corresponds to one of the URLs in the API spec"""

        base_url_re = re.compile("http[s]?://((?:[-\w.]|(?:%[\da-fA-F]{2}))+)")
        base_parsed_url = base_url_re.findall(parsed_url)[0].strip().lower()
        api_doc_allowed_urls = base_url_re.findall(self.api_docs)
        for url in api_doc_allowed_urls:
            if base_parsed_url == url.strip().lower():
                return
        raise ValueError(
            f"Parsed URL {parsed_url} is not covered by the API specification"
        )

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.question_key]
        api_url = self.api_request_chain.predict(
            question=question,
            api_docs=self.api_docs,
            callbacks=_run_manager.get_child(),
        )
        _run_manager.on_text(api_url, color="green", end="\n", verbose=self.verbose)
        api_url = api_url.strip()
        if self.allow_unverified_urls is False:
            self.verify_api_url_is_legit(api_url)
        api_response = self.requests_wrapper.get(api_url)
        _run_manager.on_text(
            api_response, color="yellow", end="\n", verbose=self.verbose
        )
        answer = self.api_answer_chain.predict(
            question=question,
            api_docs=self.api_docs,
            api_url=api_url,
            api_response=api_response,
            callbacks=_run_manager.get_child(),
        )

        return {self.output_key: answer}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.question_key]
        api_url = await self.api_request_chain.apredict(
            question=question,
            api_docs=self.api_docs,
            callbacks=_run_manager.get_child(),
        )
        await _run_manager.on_text(
            api_url, color="green", end="\n", verbose=self.verbose
        )
        api_url = api_url.strip()
        api_response = await self.requests_wrapper.aget(api_url)
        await _run_manager.on_text(
            api_response, color="yellow", end="\n", verbose=self.verbose
        )
        answer = await self.api_answer_chain.apredict(
            question=question,
            api_docs=self.api_docs,
            api_url=api_url,
            api_response=api_response,
            callbacks=_run_manager.get_child(),
        )
        return {self.output_key: answer}

    @classmethod
    def from_llm_and_api_docs(
        cls,
        llm: BaseLanguageModel,
        api_docs: str,
        headers: Optional[dict] = None,
        allow_unverified_urls: bool = False,
        api_url_prompt: BasePromptTemplate = API_URL_PROMPT,
        api_response_prompt: BasePromptTemplate = API_RESPONSE_PROMPT,
        **kwargs: Any,
    ) -> APIChain:
        """Load chain from just an LLM and the api docs."""
        get_request_chain = LLMChain(llm=llm, prompt=api_url_prompt)
        requests_wrapper = TextRequestsWrapper(headers=headers)
        get_answer_chain = LLMChain(llm=llm, prompt=api_response_prompt)
        return cls(
            api_request_chain=get_request_chain,
            api_answer_chain=get_answer_chain,
            requests_wrapper=requests_wrapper,
            api_docs=api_docs,
            allow_unverified_urls=allow_unverified_urls,
            **kwargs,
        )

    @property
    def _chain_type(self) -> str:
        return "api_chain"
