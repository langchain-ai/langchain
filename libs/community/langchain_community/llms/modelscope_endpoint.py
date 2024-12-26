import json
from collections.abc import Mapping
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import httpx
import requests
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LLM
from langchain_core.outputs.generation import GenerationChunk
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)

from langchain_community.llms.utils import enforce_stop_tokens

MODELSCOPE_SERVICE_URL_BASE = "https://api-inference.modelscope.cn/v1"


def _convert_chunk_to_str(chunk: str) -> str:
    if chunk == "":
        return ""
    chunk = chunk.lstrip("data: ")
    if chunk == "[DONE]":
        return ""
    data = json.loads(chunk)
    text = data["choices"][0]["delta"]["content"]
    return text


class ModelScopeClient(BaseModel):
    """An API client that talks to the Modelscope api inference server."""

    api_key: SecretStr
    """The API key to use for authentication."""
    base_url: str = MODELSCOPE_SERVICE_URL_BASE
    timeout: int = 60

    def completion(self, request: Any) -> str:
        headers = {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=request,
            timeout=self.timeout,
        )
        if not response.ok:
            raise ValueError(f"HTTP {response.status_code} error: {response.text}")
        return response.json()["choices"][0]["message"]["content"]

    async def acompletion(self, request: Any) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            headers = {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=request,
            )
            if not response.status_code == 200:
                raise ValueError(f"HTTP {response.status_code} error: {response.text}")
            return response.json()["choices"][0]["message"]["content"]

    def stream(self, request: Any) -> Iterator[str]:
        headers = {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}
        with requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=request,
            timeout=self.timeout,
            stream=True,
        ) as response:
            if not response.ok:
                raise ValueError(f"HTTP {response.status_code} error: {response.text}")
            for line in response.iter_lines(decode_unicode=True):
                text = _convert_chunk_to_str(line)
                if text:
                    yield text

    async def astream(self, request: Any) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            headers = {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=request,
            ) as response:
                if not response.status_code == 200:
                    raise ValueError(
                        f"HTTP {response.status_code} error: {response.text}"
                    )
                async for line in response.aiter_lines():
                    text = _convert_chunk_to_str(line)
                    if text:
                        yield text


class ModelScopeCommon(BaseModel):
    """Common parameters for Modelscope LLMs."""

    client: Any
    base_url: str = MODELSCOPE_SERVICE_URL_BASE
    modelscope_sdk_token: Optional[SecretStr] = Field(default=None, alias="api_key")
    model_name: str = Field(default="Qwen/Qwen2.5-Coder-32B-Instruct", alias="model")
    """Model name. Available models listed here: https://modelscope.cn/docs/model-service/API-Inference/intro """
    max_tokens: int = 1024
    """Maximum number of tokens to generate."""
    temperature: float = 0.3
    """Temperature parameter (higher values make the model more creative)."""
    timeout: int = 60
    """Timeout for the request."""

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    @property
    def lc_secrets(self) -> dict:
        """A map of constructor argument names to secret ids.

        For example,
            {"modelscope_sdk_token": "MODELSCOPE_SDK_TOKEN"}
        """
        return {"modelscope_sdk_token": "MODELSCOPE_SDK_TOKEN"}

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        return {**self._default_params}

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
        """Build extra parameters.
        Override the superclass method, prevent the model parameter from being
        overridden.
        """
        return values

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["modelscope_sdk_token"] = convert_to_secret_str(
            get_from_dict_or_env(
                values, ["modelscope_sdk_token", "api_key"], "MODELSCOPE_SDK_TOKEN"
            )
        )

        values["client"] = ModelScopeClient(
            api_key=values["modelscope_sdk_token"],
            base_url=values["base_url"] if "base_url" in values else MODELSCOPE_SERVICE_URL_BASE,
            timeout=values["timeout"] if "timeout" in values else 60,
        )
        return values


class ModelScopeEndpoint(ModelScopeCommon, LLM):
    """Modelscope model inference API endpoint.

    To use, you should have a modelscope account and the environment variable ``MODELSCOPE_SDK_TOKEN`` set with your
    API key. Refer to https://modelscope.cn/docs/model-service/API-Inference/intro for more details.

    Example:
        .. code-block:: python

            from langchain_community.llms.modelscope_endpoint import ModelscopeEndpoint

            llm = ModelscopeEndpoint(model="Qwen/Qwen2.5-Coder-32B-Instruct")

            # invoke
            llm.invoke("write a quick sort in python")
            # stream
            for chunk in llm.stream("write a quick sort in python"):
                print(chunk, end='', flush=True)
            # ainvoke
            asyncio.run(llm.ainvoke("write a quick sort in python"))
            # astream
            async for chunk in llm.astream("write a quick sort in python"):
                print(chunk, end='', flush=True)

    """

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "modelscope_endpoint"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "base_url": self.base_url,
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        request = self._invocation_params
        request["messages"] = [{"role": "user", "content": prompt}]
        request.update(kwargs)
        text = self.client.completion(request)
        if stop is not None:
            # This is required since the stop tokens
            # are not enforced by the model parameters
            text = enforce_stop_tokens(text, stop)

        return text

    async def _acall(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        request = self._invocation_params
        request["messages"] = [{"role": "user", "content": prompt}]
        request.update(kwargs)
        text = await self.client.acompletion(request)
        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text

    def _stream(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        request = self._invocation_params
        request["messages"] = [{"role": "user", "content": prompt}]
        request.update(kwargs)
        request["stream"] = True
        for text in self.client.stream(request):
            yield GenerationChunk(text=text)

    async def _astream(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        request = self._invocation_params
        request["messages"] = [{"role": "user", "content": prompt}]
        request.update(kwargs)
        request["stream"] = True
        async for text in self.client.astream(request):
            yield GenerationChunk(text=text)
