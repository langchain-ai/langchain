from __future__ import annotations

import copy
import json
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    TypedDict,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM, BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, PrivateAttr, root_validator

if TYPE_CHECKING:
    import bentoml
    import openllm


class IdentifyingParams(TypedDict):
    """Parameters for identifying a model as a typed dict."""

    model_id: Optional[str]
    quantize: Optional[Literal["awq", "gptq", "squeezellm"]]
    serialization: Literal["safetensors", "legacy"]
    trust_remote_code: bool
    llm_kwargs: Dict[str, Any]


logger = logging.getLogger(__name__)


class OpenLLMAPI(LLM):
    """OpenLLMAPI supports interacting with OpenLLM Server.

    To use, you should have the openllm-client library installed:

    .. code-block:: bash

        pip install openllm-client

    Learn more at: https://github.com/bentoml/openllm

    Example running an LLM model locally managed by OpenLLM:
        .. code-block:: python

            from langchain_community.llms import OpenLLMAPI
            llm = OpenLLMAPI(server_url='localhost:3000')
            llm.invoke("What is the difference between a duck and a goose?")
    """

    server_url: Optional[str] = None
    """Optional server URL that currently runs a LLMServer with 'openllm start'."""
    timeout: int = 30
    """"Time out for the openllm client"""
    llm_kwargs: Dict[str, Any]
    """Keyword arguments to be passed to openllm.LLM"""

    _sync_client: openllm.HTTPClient = PrivateAttr(default=None)  #: :meta private:
    _async_client: openllm.AsyncHTTPClient = PrivateAttr(
        default=None
    )  #: :meta private:

    class Config:
        extra = "forbid"

    def __init__(
        self,
        server_url: str,
        timeout: int = 30,
        **llm_kwargs: Any,
    ):
        try:
            import openllm_client
        except ImportError as e:
            raise ImportError(
                "Could not import openllm-client. Make sure to install it with "
                "'pip install openllm-client.'"
            ) from e

        llm_kwargs = llm_kwargs or {}

        super().__init__(  # type: ignore
            server_url=server_url,
            timeout=timeout,
            llm_kwargs=llm_kwargs,
        )

        self._sync_client = openllm_client.HTTPClient(
            address=server_url, timeout=timeout
        )
        self._async_client = openllm_client.AsyncHTTPClient(
            address=server_url, timeout=timeout
        )

    @property
    def _identifying_params(self) -> IdentifyingParams:
        """Get the identifying parameters."""
        if self._sync_client is None:
            raise ValueError("OpenLLMAPI is not initialized correctly.")
        self.llm_kwargs.update(self._sync_client._config)
        return IdentifyingParams(
            llm_kwargs=self.llm_kwargs,
            model_id=self._sync_client._metadata.model_id,
            trust_remote_code=self._sync_client._metadata.trust_remote_code,
            quantize=self._sync_client._metadata.quantise,
            serialization=self._sync_client._metadata.serialisation,
        )

    @property
    def _llm_type(self) -> str:
        return "openllm_client"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            import openllm_core as core
        except ImportError as e:
            raise ImportError(
                "Could not import openllm-client. Make sure to install it with "
                "'pip install openllm-client'."
            ) from e

        copied = copy.deepcopy(self.llm_kwargs)
        copied.update(kwargs)
        if stop:
            copied.update(stop=stop)
        config = core.AutoConfig.from_id(self._identifying_params["model_id"], **copied)
        return self._sync_client.generate(prompt, **config.model_dump()).outputs[0].text

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            import openllm_core as core
        except ImportError as e:
            raise ImportError(
                "Could not import openllm-client. Make sure to install it with "
                "'pip install openllm-client'."
            ) from e

        copied = copy.deepcopy(self.llm_kwargs)
        copied.update(kwargs)
        if stop:
            copied.update(stop=stop)
        config = core.AutoConfig.from_id(self._identifying_params["model_id"], **copied)
        return (
            (await self._async_client.generate(prompt, **config.model_dump()))
            .outputs[0]
            .text
        )

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        try:
            import openllm_core as core
        except ImportError as e:
            raise ImportError(
                "Could not import openllm-client. Make sure to install it with "
                "'pip install openllm-client'."
            ) from e

        copied = copy.deepcopy(self.llm_kwargs)
        copied.update(kwargs)
        if stop:
            copied.update(stop=stop)
        config = core.AutoConfig.from_id(self._identifying_params["model_id"], **copied)

        for res in self._sync_client.generate_stream(
            prompt, llm_config=config.model_dump(), stop=stop
        ):
            yield GenerationChunk(
                text=res.text,
                generation_info=dict(
                    request_id=[res.request_id],
                    index=[res.index],
                    token_ids=[res.token_ids],
                ),
            )

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        try:
            import openllm_core as core
        except ImportError as e:
            raise ImportError(
                "Could not import openllm-client. Make sure to install it with "
                "'pip install openllm-client'."
            ) from e

        copied = copy.deepcopy(self.llm_kwargs)
        copied.update(kwargs)
        if stop:
            copied.update(stop=stop)
        config = core.AutoConfig.from_id(self._identifying_params["model_id"], **copied)

        async for res in self._async_client.generate_stream(
            prompt, llm_config=config.model_dump(), stop=stop
        ):
            yield GenerationChunk(
                text=res.text,
                generation_info=dict(
                    request_id=[res.request_id],
                    index=[res.index],
                    token_ids=[res.token_ids],
                ),
            )


class OpenLLM(BaseLLM):
    """OpenLLM class supports running localized LLM.

    Note that since openllm>=0.5, you will be required to have GPU
    in order to use this class. If you have a OpenLLM server running
    elsewhere, you should use OpenLLMAPI instead.

    To use, you should have the openllm library installed:

    .. code-block:: bash

        pip install openllm

    Learn more at: https://github.com/bentoml/openllm

    Example running an LLM model locally managed by OpenLLM:
        .. code-block:: python

            from langchain_community.llms import OpenLLM
            llm = OpenLLM(
                model_id='microsoft/Phi-3-mini-4k-instruct',
                trust_remote_code=True,
            )
            llm.invoke("What is the difference between a duck and a goose?")
    """

    model_id: Optional[str] = None
    """Model id from HuggingFace"""
    bentomodel: Optional[bentoml.Model] = None
    """Private model saved under BentoML model store."""
    dtype: str = "auto"
    """Configure dtype for this given model. Default to auto."""
    quantize: Optional[Literal["awq", "gptq", "squeezellm"]] = None
    """Optional quantization methods to use with this LLM.
    See OpenLLM's --quantize options from `openllm start` for more information."""
    serialization: Literal["safetensors", "legacy"] = "safetensors"
    """Optional serialization methods for this LLM to be save as.
    Default to 'safetensors', but will fallback to PyTorch pickle `.bin`
    on some models."""
    trust_remote_code: bool = False
    """If the model requires external code execution, then
    pass 'trust_remote_code=True'. Synonymous to HF's trust_remote_code."""
    llm_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to be passed to openllm.LLM"""

    llm: Any  #: :meta private:

    class Config:
        extra = "forbid"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        try:
            import openllm
        except ImportError as e:
            raise ImportError(
                "Could not import openllm. Make sure to install it with "
                "'pip install openllm.'"
            ) from e

        values["llm"] = openllm.LLM.from_model(
            values["model_id"],
            dtype=values["dtype"],
            bentomodel=values["bentomodel"],
            quantise=values["quantize"],
            serialisation=values["serialization"],
            trust_remote_code=values["trust_remote_code"],
            mode="batch",
            **values["llm_kwargs"],
        )
        return values

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        model_id = self.model_id
        try:
            self.llm_kwargs.update(
                json.loads(self.llm.identifying_params["configuration"])
            )
        except (TypeError, json.JSONDecodeError):
            pass
        return dict(
            llm_kwargs=self.llm_kwargs,
            model_id=model_id,
            dtype=self.dtype,
            quantize=self.quantize,
            serialization=self.serialization,
            trust_remote_code=self.trust_remote_code,
        )

    @property
    def _llm_type(self) -> str:
        return "openllm"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        copied = copy.deepcopy(self.llm_kwargs)
        copied.update(kwargs)
        if stop:
            copied.update(stop=stop)
        outputs = self.llm.batch(prompts, copied)

        generations = []
        for output in outputs:
            text = output.outputs[0].text
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)
