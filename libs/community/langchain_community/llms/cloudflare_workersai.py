import json
import logging
from typing import Any, Dict, Iterator, List, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

logger = logging.getLogger(__name__)


class CloudflareWorkersAI(LLM):
    """Cloudflare Workers AI service.

    To use, you must provide an API token and
    account ID to access Cloudflare Workers AI, and
    pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.llms.cloudflare_workersai import CloudflareWorkersAI

            my_account_id = "my_account_id"
            my_api_token = "my_secret_api_token"
            llm_model =  "@cf/meta/llama-2-7b-chat-int8"

            cf_ai = CloudflareWorkersAI(
                account_id=my_account_id,
                api_token=my_api_token,
                model=llm_model
            )
    """  # noqa: E501

    account_id: str
    api_token: str
    model: str = "@cf/meta/llama-2-7b-chat-int8"
    base_url: str = "https://api.cloudflare.com/client/v4/accounts"
    streaming: bool = False
    endpoint_url: str = ""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Cloudflare Workers AI class."""
        super().__init__(**kwargs)

        self.endpoint_url = f"{self.base_url}/{self.account_id}/ai/run/{self.model}"

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "cloudflare"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Default parameters"""
        return {}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Identifying parameters"""
        return {
            "account_id": self.account_id,
            "api_token": self.api_token,
            "model": self.model,
            "base_url": self.base_url,
        }

    def _call_api(self, prompt: str, params: Dict[str, Any]) -> requests.Response:
        """Call Cloudflare Workers API"""
        headers = {"Authorization": f"Bearer {self.api_token}"}
        data = {"prompt": prompt, "stream": self.streaming, **params}
        response = requests.post(
            self.endpoint_url, headers=headers, json=data, stream=self.streaming
        )
        return response

    def _process_response(self, response: requests.Response) -> str:
        """Process API response"""
        if response.ok:
            data = response.json()
            return data["result"]["response"]
        else:
            raise ValueError(f"Request failed with status {response.status_code}")

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Streaming prediction"""
        original_steaming: bool = self.streaming
        self.streaming = True
        _response_prefix_count = len("data: ")
        _response_stream_end = b"data: [DONE]"
        for chunk in self._call_api(prompt, kwargs).iter_lines():
            if chunk == _response_stream_end:
                break
            if len(chunk) > _response_prefix_count:
                try:
                    data = json.loads(chunk[_response_prefix_count:])
                except Exception as e:
                    logger.debug(chunk)
                    raise e
                if data is not None and "response" in data:
                    if run_manager:
                        run_manager.on_llm_new_token(data["response"])
                    yield GenerationChunk(text=data["response"])
        logger.debug("stream end")
        self.streaming = original_steaming

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Regular prediction"""
        if self.streaming:
            return "".join(
                [c.text for c in self._stream(prompt, stop, run_manager, **kwargs)]
            )
        else:
            response = self._call_api(prompt, kwargs)
            return self._process_response(response)
