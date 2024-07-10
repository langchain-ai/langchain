from __future__ import annotations

import logging
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Field
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init

logger = logging.getLogger(__name__)


class QianfanLLMEndpoint(LLM):
    """Baidu Qianfan hosted open source or customized models.

    To use, you should have the ``qianfan`` python package installed, and
    the environment variable ``qianfan_ak`` and ``qianfan_sk`` set with
    your API key and Secret Key.

    ak, sk are required parameters which you could get from
    https://cloud.baidu.com/product/wenxinworkshop

    Example:
        .. code-block:: python

            from langchain_community.llms import QianfanLLMEndpoint
            qianfan_model = QianfanLLMEndpoint(model="ERNIE-Bot",
                endpoint="your_endpoint", qianfan_ak="your_ak", qianfan_sk="your_sk")
    """

    init_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """init kwargs for qianfan client init, such as `query_per_second` which is 
        associated with qianfan resource object to limit QPS"""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """extra params for model invoke using with `do`."""

    client: Any

    qianfan_ak: Optional[str] = None
    qianfan_sk: Optional[str] = None

    streaming: Optional[bool] = False
    """Whether to stream the results or not."""

    model: str = "ERNIE-Bot-turbo"
    """Model name. 
    you could get from https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Nlks5zkzu
    
    preset models are mapping to an endpoint.
    `model` will be ignored if `endpoint` is set
    """

    endpoint: Optional[str] = None
    """Endpoint of the Qianfan LLM, required if custom model used."""

    request_timeout: Optional[int] = 60
    """request timeout for chat http requests"""

    top_p: Optional[float] = 0.8
    temperature: Optional[float] = 0.95
    penalty_score: Optional[float] = 1
    """Model params, only supported in ERNIE-Bot and ERNIE-Bot-turbo.
    In the case of other model, passing these params will not affect the result.
    """

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        values["qianfan_ak"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "qianfan_ak",
                "QIANFAN_AK",
                default="",
            )
        )
        values["qianfan_sk"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "qianfan_sk",
                "QIANFAN_SK",
                default="",
            )
        )

        params = {
            **values.get("init_kwargs", {}),
            "model": values["model"],
        }
        if values["qianfan_ak"].get_secret_value() != "":
            params["ak"] = values["qianfan_ak"].get_secret_value()
        if values["qianfan_sk"].get_secret_value() != "":
            params["sk"] = values["qianfan_sk"].get_secret_value()
        if values["endpoint"] is not None and values["endpoint"] != "":
            params["endpoint"] = values["endpoint"]
        try:
            import qianfan

            values["client"] = qianfan.Completion(**params)
        except ImportError:
            raise ImportError(
                "qianfan package not found, please install it with "
                "`pip install qianfan`"
            )
        return values

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            **{"endpoint": self.endpoint, "model": self.model},
            **super()._identifying_params,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "baidu-qianfan-endpoint"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Qianfan API."""
        normal_params = {
            "model": self.model,
            "endpoint": self.endpoint,
            "stream": self.streaming,
            "request_timeout": self.request_timeout,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "penalty_score": self.penalty_score,
        }

        return {**normal_params, **self.model_kwargs}

    def _convert_prompt_msg_params(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> dict:
        if "streaming" in kwargs:
            kwargs["stream"] = kwargs.pop("streaming")
        return {
            **{"prompt": prompt, "model": self.model},
            **self._default_params,
            **kwargs,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to an qianfan models endpoint for each generation with a prompt.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python
                response = qianfan_model.invoke("Tell me a joke.")
        """
        if self.streaming:
            completion = ""
            for chunk in self._stream(prompt, stop, run_manager, **kwargs):
                completion += chunk.text
            return completion
        params = self._convert_prompt_msg_params(prompt, **kwargs)
        params["stop"] = stop
        response_payload = self.client.do(**params)

        return response_payload["result"]

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.streaming:
            completion = ""
            async for chunk in self._astream(prompt, stop, run_manager, **kwargs):
                completion += chunk.text
            return completion

        params = self._convert_prompt_msg_params(prompt, **kwargs)
        params["stop"] = stop
        response_payload = await self.client.ado(**params)

        return response_payload["result"]

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        params = self._convert_prompt_msg_params(prompt, **{**kwargs, "stream": True})
        params["stop"] = stop
        for res in self.client.do(**params):
            if res:
                chunk = GenerationChunk(text=res["result"])
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text)
                yield chunk

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        params = self._convert_prompt_msg_params(prompt, **{**kwargs, "stream": True})
        params["stop"] = stop
        async for res in await self.client.ado(**params):
            if res:
                chunk = GenerationChunk(text=res["result"])
                if run_manager:
                    await run_manager.on_llm_new_token(chunk.text)
                yield chunk
