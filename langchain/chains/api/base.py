"""Chain that makes API calls and summarizes the responses to answer a question."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, root_validator

from langchain.chains.api.prompt import API_RESPONSE_PROMPT, API_URL_PROMPT
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts import BasePromptTemplate
from langchain.requests import RequestsWrapper


class APIChain(Chain, BaseModel):
    """Chain that makes API calls and summarizes the responses to answer a question."""

    api_request_chain: LLMChain
    api_answer_chain: LLMChain
    requests_wrapper: RequestsWrapper
    api_docs: str
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

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        question = inputs[self.question_key]
        api_url = self.api_request_chain.predict(
            question=question, api_docs=self.api_docs
        )
        if self.verbose:
            self.callback_manager.on_text(api_url, color="green", end="\n")
        api_response = self.requests_wrapper.run(api_url)
        if self.verbose:
            self.callback_manager.on_text(api_response, color="yellow", end="\n")
        answer = self.api_answer_chain.predict(
            question=question,
            api_docs=self.api_docs,
            api_url=api_url,
            api_response=api_response,
        )
        return {self.output_key: answer}

    @classmethod
    def from_llm_and_api_docs(
        cls,
        llm: BaseLLM,
        api_docs: str,
        headers: Optional[dict] = None,
        api_url_prompt: BasePromptTemplate = API_URL_PROMPT,
        api_response_prompt: BasePromptTemplate = API_RESPONSE_PROMPT,
        **kwargs: Any,
    ) -> APIChain:
        """Load chain from just an LLM and the api docs."""
        get_request_chain = LLMChain(llm=llm, prompt=api_url_prompt)
        requests_wrapper = RequestsWrapper(headers=headers)
        get_answer_chain = LLMChain(llm=llm, prompt=api_response_prompt)
        return cls(
            api_request_chain=get_request_chain,
            api_answer_chain=get_answer_chain,
            requests_wrapper=requests_wrapper,
            api_docs=api_docs,
            **kwargs,
        )
