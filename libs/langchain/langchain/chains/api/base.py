"""Chain that makes API calls and summarizes the responses to answer a question."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.api.prompt import API_RESPONSE_PROMPT, API_URL_PROMPT
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.pydantic_v1 import Field, root_validator
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.utilities.requests import TextRequestsWrapper


class APIChain(Chain):
    """Chain that makes API calls and summarizes the responses to answer a question.

    *Security Note*: This API chain uses the requests toolkit
        to make GET, POST, PATCH, PUT, and DELETE requests to an API.

        Exercise care in who is allowed to use this chain. If exposing
        to end users, consider that users will be able to make arbitrary
        requests on behalf of the server hosting the code. For example,
        users could ask the server to make a request to a private API
        that is only accessible from the server.

        Control access to who can submit issue requests using this toolkit and
        what network access it has.

        See https://python.langchain.com/docs/security for more information.
    """

    api_request_chain: LLMChain
    api_answer_chain: LLMChain
    requests_wrapper: TextRequestsWrapper = Field(exclude=True)
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
            **kwargs,
        )

    @property
    def _chain_type(self) -> str:
        return "api_chain"
