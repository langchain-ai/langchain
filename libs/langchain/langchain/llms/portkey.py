from __future__ import annotations
from langchain.llms.base import LLM
from langchain.callbacks.manager import (
    CallbackManagerForLLMRun
)
from langchain.schema.output import GenerationChunk
from langchain.schema import Generation, LLMResult
from typing import List, Optional, Union, TYPE_CHECKING, Dict, Any

from langchain.pydantic_v1 import Field, PrivateAttr

import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import portkey
    from portkey import (
        LLMBase,
        PortkeyModes,
        PortkeyCacheType,
        PortkeyCacheLiteral,
        PortkeyResponse,
        PortkeyModesLiteral,
    )


IMPORT_ERROR_MESSAGE = (
    "Portkey is not installed.Please install it with `pip install portkey-ai`."
)


class Portkey(LLM):
    mode: Optional[Union["PortkeyModes", "PortkeyModesLiteral"]] = Field(
        description="The mode for using the Portkey integration"
    )

    model: Optional[str] = Field(default="gpt-3.5-turbo")
    llm: "LLMBase" = Field(description="LLM parameter", default_factory=dict)

    llms: List["LLMBase"] = Field(description="LLM parameters", default_factory=list)

    _client: "portkey" = PrivateAttr()
    stream: Optional[bool] = Field(default=False)

    def __init__(
        self,
        *,
        mode: Optional[Union["PortkeyModes", "PortkeyModesLiteral"]] = None,
        api_key: str = "",
        cache_status: Optional[Union["PortkeyCacheType", "PortkeyCacheLiteral"]] = None,
        trace_id: Optional[str] = "",
        cache_age: Optional[int] = None,
        cache_force_refresh: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        base_url: Optional[str] = None,
        stream: Optional[bool] = False,
        **kwargs
    ) -> None:
        try:
            import portkey
            from portkey import Params
        except ImportError:
            raise ImportError(IMPORT_ERROR_MESSAGE) from None
        super().__init__()
        self._client = portkey
        self._client.api_key = api_key
        self._client.mode = mode
        if base_url is not None:
            self._client.base_url = base_url
        self.model = None
        self._client.params = Params(
            cache_status=cache_status,
            trace_id=trace_id,
            cache_age=cache_age,
            metadata=metadata,
            cache_force_refresh=cache_force_refresh,
            **kwargs
        )
        self.stream = stream

    def add_llms(self, llm_params: Union[LLMBase, List[LLMBase]]) -> "Portkey":
        try:
            from portkey import LLMBase, Config
        except ImportError as exc:
            raise ImportError(IMPORT_ERROR_MESSAGE) from exc
        if isinstance(llm_params, LLMBase):
            llm_params = [llm_params]
        self.llms.extend(llm_params)
        if self.model is None:
            self.model = self.llms[0].model
        self._client.config = Config(
            llms=self.llms
        )
        return self

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        if self.streaming:
            generation: Optional[GenerationChunk] = None
            for chunk in self._stream(prompts[0], stop, run_manager, **kwargs):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return LLMResult(generations=[[generation]])

        messages, params = self._get_chat_params(prompts, stop)
        params = {**params, **kwargs}
        full_response = completion_with_retry(
            self, messages=messages, run_manager=run_manager, **params
        )
        llm_output = {
            "token_usage": full_response["usage"],
            "model_name": self.model_name,
        }
        return LLMResult(
            generations=[
                [Generation(text=full_response["choices"][0]["message"]["content"])]
            ],
            llm_output=llm_output,
        )

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "portkey-ai-gateway"
