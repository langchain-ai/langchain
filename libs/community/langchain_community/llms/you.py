import os
from typing import Any, Dict, Generator, Iterator, List, Literal, Optional

import requests
import sseclient
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from pydantic import Field

SMART_ENDPOINT = "https://chat-api.you.com/smart"
RESEARCH_ENDPOINT = "https://chat-api.you.com/research"


def _request(base_url: str, api_key: str, **kwargs) -> Dict[str, Any]:
    """
    NOTE: This function can be replaced by a OpenAPI-generated Python SDK in the future,
    for better input/output typing support.
    """
    headers = {"x-api-key": api_key}
    response = requests.post(base_url, headers=headers, json=kwargs)
    response.raise_for_status()
    return response.json()


def _request_stream(
    base_url: str, api_key: str, **kwargs
) -> Generator[str, None, None]:
    headers = {"x-api-key": api_key}
    params = dict(**kwargs, stream=True)
    response = requests.post(base_url, headers=headers, stream=True, json=params)
    response.raise_for_status()

    client = sseclient.SSEClient(response)
    for event in client.events():
        if event.event in ("search_results", "done"):
            pass
        elif event.event == "token":
            yield event.data
        elif event.event == "error":
            raise ValueError(f"Error in response: {event.data}")
        else:
            raise NotImplementedError(f"Unknown event type {event.event}")


class You(LLM):
    """Wrapper around You.com's conversational Smart and Research APIs.

    Each API endpoint is designed to generate conversational
    responses to a variety of query types, including inline citations
    and web results when relevant.

    Smart Mode:
    - Quick, reliable answers for a variety of questions
    - Cites the entire web page URL

    Research Mode:
    - In-depth answers with extensive citations for a variety of questions
    - Cites the specific web page snippet relevant to the claim

    To connect to the You.com api requires an API key which
    you can get at https://api.you.com.

    For more information, check out the documentations at
    https://documentation.you.com/api-reference/.

    Args:
        mode: You.com conversational endpoints. Choose from "smart" or "research"
        ydc_api_key: You.com API key, if `YDC_API_KEY` is not set in the environment
    """

    mode: Literal["smart", "research"] = Field(
        "smart",
        description='You.com conversational endpoints. Choose from "smart" or "research"',
    )
    ydc_api_key: Optional[str] = Field(
        None,
        description="You.com API key, if `YDC_API_KEY` is not set in the envrioment",
    )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop:
            raise NotImplementedError(
                "Stop words are not implemented for You.com endpoints."
            )
        params = {"query": prompt}
        response = _request(self.endpoint, api_key=self.api_key, **params)
        return response["answer"]

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        if stop:
            raise NotImplementedError(
                "Stop words are not implemented for You.com endpoints."
            )
        params = {"query": prompt}
        for token in _request_stream(self.endpoint, api_key=self.api_key, **params):
            yield GenerationChunk(text=token)

    @property
    def endpoint(self) -> str:
        if self.mode == "smart":
            return SMART_ENDPOINT
        return RESEARCH_ENDPOINT

    @property
    def api_key(self) -> str:
        return os.environ["YDC_API_KEY"]

    @property
    def _llm_type(self) -> str:
        return "you.com"
